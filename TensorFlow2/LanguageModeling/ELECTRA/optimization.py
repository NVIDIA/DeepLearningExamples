# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions and classes related to optimization (weight updates)."""

import re
import collections
import tensorflow as tf
import tensorflow_addons.optimizers as tfa_optimizers

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops
from utils import log


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applys a warmup schedule on a given learning rate decay schedule."""

    def __init__(self, initial_learning_rate, decay_schedule_fn, warmup_steps, power=1.0, name=None):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


def create_optimizer(init_lr, num_train_steps, num_warmup_steps, weight_decay_rate=0.01,
                     layerwise_lr_decay=-1, n_transformer_layers=None, clip_norm=1.0,
                     optimizer="adam", skip_adaptive=False, power=1.0, beta_1=0.9, beta_2=0.999, end_lr=0.0):
    """Creates an optimizer with learning rate schedule."""
    # Implements linear decay of the learning rate.
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr, decay_steps=num_train_steps - num_warmup_steps, end_learning_rate=end_lr, power=power
    )
    if num_warmup_steps:
        learning_rate_fn = WarmUp(
            initial_learning_rate=init_lr, decay_schedule_fn=learning_rate_fn, warmup_steps=num_warmup_steps
        )
    layer_decay = None
    if layerwise_lr_decay > 0 and n_transformer_layers is not None:
        layer_decay = _get_layer_decay(layerwise_lr_decay, n_transformer_layers)

    if optimizer == "adam":
        optimizer = AdamWeightDecay(
            learning_rate=learning_rate_fn,
            weight_decay_rate=weight_decay_rate,
            layer_decay=layer_decay,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=1e-6,
            exclude_from_weight_decay=["layer_norm", "bias", "LayerNorm"],
            clip_norm=clip_norm,
        )
    else:
        if skip_adaptive:
            skip_list = ["layer_norm", "bias", "LayerNorm"]
        else:
            skip_list = ["None"]
        log("Skip list for LAMB {}".format(skip_list))
        
        optimizer = tfa_optimizers.LAMB(
            learning_rate=learning_rate_fn,
            weight_decay_rate=weight_decay_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=1e-6,
            exclude_from_weight_decay=["layer_norm", "bias", "LayerNorm"],
            exclude_from_layer_adaptation=skip_list,
        )

    return optimizer


class AdamWeightDecay(tf.keras.optimizers.Adam):
    """Adam enables L2 weight decay and clip_by_global_norm on gradients.

  Just adding the square of the weights to the loss function is *not* the
  correct way of using L2 regularization/weight decay with Adam, since that will
  interact with the m and v parameters in strange ways.

  Instead we want ot decay the weights in a manner that doesn't interact with
  the m/v parameters. This is equivalent to adding the square of the weights to
  the loss with plain (non-momentum) SGD.
  """

    def __init__(
            self,
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False,
            weight_decay_rate=0.0,
            include_in_weight_decay=None,
            exclude_from_weight_decay=None,
            layer_decay=None,
            clip_norm=1.0,
            name="AdamWeightDecay",
            **kwargs
    ):
        super().__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay
        self.layer_decay = layer_decay
        self.clip_norm = clip_norm

    @classmethod
    def from_config(cls, config):
        """Creates an optimizer from its config with WarmUp custom object."""
        custom_objects = {"WarmUp": WarmUp}
        return super().from_config(config, custom_objects=custom_objects)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        apply_state["weight_decay_rate"] = tf.constant(self.weight_decay_rate, name="adam_weight_decay_rate")

    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(
                learning_rate * var * apply_state["weight_decay_rate"], use_locking=self._use_locking
            )
        return tf.no_op()

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        grads, tvars = list(zip(*grads_and_vars))
        # Being done in train_step
        ##(grads, _) = tf.clip_by_global_norm(grads, clip_norm=self.clip_norm)
        return super().apply_gradients(zip(grads, tvars), name=name,
                                       experimental_aggregate_gradients=experimental_aggregate_gradients)

    def _get_lr(self, var, apply_state):
        """Retrieves the learning rate with the given state."""
        # if apply_state is None:
        #     return self._decayed_lr_t[var_dtype], {}
        var_name, var_device, var_dtype = var.name, var.device, var.dtype.base_dtype

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients
        lr_t = coefficients["lr_t"]
        lr = coefficients["lr"]

        if self.layer_decay is not None:
            update_for_var = False
            for key in self.layer_decay:
                if key in var_name:
                    update_for_var = True
                    lr_t *= self.layer_decay[key]
                    lr *= self.layer_decay[key]
                    break
            if not update_for_var:
                raise ValueError("No learning rate specified for variable", var)

        return lr_t, lr, coefficients, dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # print("Dense: {} {} {}".format(var.name, var.device, var.dtype.base_dtype))
        lr_t, _, coefficients, kwargs = self._get_lr(var, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            m = self.get_slot(var, 'm')
            v = self.get_slot(var, 'v')

            if not self.amsgrad:
                return training_ops.resource_apply_adam(
                    var.handle,
                    m.handle,
                    v.handle,
                    coefficients['beta_1_power'],
                    coefficients['beta_2_power'],
                    lr_t,
                    coefficients['beta_1_t'],
                    coefficients['beta_2_t'],
                    coefficients['epsilon'],
                    grad,
                    use_locking=self._use_locking)
            else:
                vhat = self.get_slot(var, 'vhat')
                return training_ops.resource_apply_adam_with_amsgrad(
                    var.handle,
                    m.handle,
                    v.handle,
                    vhat.handle,
                    coefficients['beta_1_power'],
                    coefficients['beta_2_power'],
                    lr_t,
                    coefficients['beta_1_t'],
                    coefficients['beta_2_t'],
                    coefficients['epsilon'],
                    grad,
                    use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # print("Sparse: {} {} {}".format(var.name, var.device, var.dtype.base_dtype))
        lr_t, lr, coefficients, kwargs = self._get_lr(var, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            # m_t = beta1 * m + (1 - beta1) * g_t
            m = self.get_slot(var, 'm')
            m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
            m_t = state_ops.assign(m, m * coefficients['beta_1_t'],
                                   use_locking=self._use_locking)
            with tf.control_dependencies([m_t]):
                m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

            # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
            v = self.get_slot(var, 'v')
            v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
            v_t = state_ops.assign(v, v * coefficients['beta_2_t'],
                                   use_locking=self._use_locking)
            with tf.control_dependencies([v_t]):
                v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

            if not self.amsgrad:
                v_sqrt = math_ops.sqrt(v_t)
                var_update = state_ops.assign_sub(
                    var, lr * m_t / (v_sqrt + coefficients['epsilon']),
                    use_locking=self._use_locking)
                return control_flow_ops.group(*[var_update, m_t, v_t])
            else:
                v_hat = self.get_slot(var, 'vhat')
                v_hat_t = math_ops.maximum(v_hat, v_t)
                with tf.control_dependencies([v_hat_t]):
                    v_hat_t = state_ops.assign(
                        v_hat, v_hat_t, use_locking=self._use_locking)
                v_hat_sqrt = math_ops.sqrt(v_hat_t)
                var_update = state_ops.assign_sub(
                    var,
                    lr * m_t / (v_hat_sqrt + coefficients['epsilon']),
                    use_locking=self._use_locking)
                return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])

    def get_config(self):
        config = super().get_config()
        config.update({"weight_decay_rate": self.weight_decay_rate})
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.weight_decay_rate == 0:
            return False

        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


# Inspired from https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/optimizers/utils.py
class GradientAccumulator(object):
    """Distribution strategies-aware gradient accumulation utility."""

    def __init__(self):
        """Initializes the accumulator."""
        self._gradients = []
        self._accum_steps = tf.Variable(
            initial_value=0, dtype=tf.int64, trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
        )

    @property
    def step(self):
        """Number of accumulated steps."""
        return self._accum_steps.value()

    @property
    def gradients(self):
        """The accumulated gradients."""
        return list(
            gradient.value() if gradient is not None else gradient for gradient in self._get_replica_gradients()
        )

    def __call__(self, gradients):
        """Accumulates :obj:`gradients`."""
        if not self._gradients:
            self._gradients.extend(
                [
                    tf.Variable(tf.zeros_like(gradient), trainable=False) if gradient is not None else gradient
                    for gradient in gradients
                ]
            )

        if len(gradients) != len(self._gradients):
            raise ValueError("Expected %s gradients, but got %d" % (len(self._gradients), len(gradients)))

        for accum_gradient, gradient in zip(self._get_replica_gradients(), gradients):
            if accum_gradient is not None and gradient is not None:
                accum_gradient.assign_add(gradient)

        self._accum_steps.assign_add(1)

    def reset(self):
        """Resets the accumulated gradients."""
        if self._gradients:
            self._accum_steps.assign(0)

        for gradient in self._get_replica_gradients():
            if gradient is not None:
                gradient.assign(tf.zeros_like(gradient))

    def _get_replica_gradients(self):
        if tf.distribute.has_strategy():
            # In a replica context, we want to accumulate gradients on each replica
            # without synchronization, so we directly assign the value of the
            # current replica.
            replica_context = tf.distribute.get_replica_context()

            if replica_context is None or tf.distribute.get_strategy().num_replicas_in_sync == 1:
                return self._gradients

            return (
                gradient.device_map.select_for_current_replica(gradient.values, replica_context)
                for gradient in self._gradients
                if gradient is not None
            )
        else:
            return self._gradients


def _get_layer_decay(layer_decay, n_layers):
    """Have lower learning rates for layers closer to the input."""
    key_to_depths = collections.OrderedDict({
        "/embeddings/": 0,
        "/embeddings_project/": 0,
        "/start_logits/": n_layers + 2,
        "/end_logits/": n_layers + 2,
        "/answer_class/": n_layers + 2,
        "/qa_outputs/": n_layers + 2,
    })
    for layer in range(n_layers):
        key_to_depths["encoder/layer_._" + str(layer) + "/"] = layer + 1
    return {
        key: layer_decay ** (n_layers + 2 - depth)
        for key, depth in key_to_depths.items()
    }
