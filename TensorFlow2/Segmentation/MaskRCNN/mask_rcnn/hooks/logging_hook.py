#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import copy
import operator
import time

import numpy as np
import tensorflow as tf

from distutils.version import LooseVersion

from mask_rcnn.utils.logging_formatter import logging

from mask_rcnn.utils import meters

from mask_rcnn.utils.decorators import atexit_hook

from mask_rcnn.utils.distributed_utils import MPI_is_distributed
from mask_rcnn.utils.distributed_utils import MPI_rank_and_size
from mask_rcnn.utils.distributed_utils import MPI_size

from mask_rcnn.utils.logging_backend import LoggingBackend
from mask_rcnn.utils.logging_backend import RuntimeMode

from mask_rcnn.utils.metric_tracking import clear_registered_metrics

from mask_rcnn.utils.metric_tracking import TF_METRICS
from mask_rcnn.utils.metric_tracking import KERAS_MODELS

from mask_rcnn.utils.lazy_imports import LazyImport
hvd = LazyImport("horovod.tensorflow")

__all__ = ["AutoLoggingHook"]


@atexit_hook
class _AutoLoggingHook(tf.estimator.SessionRunHook):

    def __init__(self, log_every_n_steps=200, warmup_steps=500, is_training=True):
        """
        AutoLogging Hook for Tensorflow

        :param log_every_n_steps: log will be output on the console every N steps
        :param warmup_steps: integers, numbers of steps considered as warmup
        :param is_training: boolean
        """

        self._logging_proxy = LoggingBackend()

        self._initialized = False
        self._metrics = copy.copy(TF_METRICS)

        self._batch_size_tensor = None

        self._AMP_steps_since_last_loss_scale = None
        self._AMP_loss_scale_tensor = None

        self._current_step = None
        self._amp_steps_non_skipped = None

        self._warmup_steps = warmup_steps

        self._log_every_n_steps = log_every_n_steps

        self._step_t0 = None
        self._session_t0 = None
        self._session_run_times = list()

        self._global_step_tensor = None

        self._is_training = is_training
        self._runtime_mode = RuntimeMode.TRAIN if is_training else RuntimeMode.VALIDATION

        self._model_throughput = meters.MovingAverageMeter(window_size=1000)
        self._model_stats = None

        self._n_gpus = None

    def __atexit__(self):

        if self._initialized:

            total_processing_time = int(np.sum(self._session_run_times))

            try:
                avg_throughput = self._model_throughput.read()
            except ValueError:
                avg_throughput = -1

            self._logging_proxy.log_summary(
                is_train=self._is_training,
                total_steps=self._current_step,
                total_processing_time=total_processing_time,
                avg_throughput=avg_throughput
            )

            metric_data = dict()

            for key, value in self._metrics.items():
                try:
                    metric_data[key] = value["aggregator"].read()

                except ValueError:
                    pass

            self._logging_proxy.log_final_metrics(metric_data=metric_data, runtime_mode=self._runtime_mode)

    def begin(self):
        """Called once before using the session.
        When called, the default graph is the one that will be launched in the
        session.  The hook can modify the graph by adding new operations to it.
        After the `begin()` call the graph will be finalized and the other callbacks
        can not modify the graph anymore. Second call of `begin()` on the same
        graph, should not change the graph.
        """

        from tensorflow.python.keras.backend import get_graph
        _graph = get_graph()

        try:
            self._batch_size_tensor = None

            for tensor in _graph.as_graph_def().node:
                if "IteratorGetNext" in tensor.name:
                    _input_tensor = _graph.get_tensor_by_name(tensor.name + ":0")
                    try:
                        self._batch_size_tensor = tf.shape(input=_input_tensor)[0]
                    except TypeError:  # Ragged Tensor
                        self._batch_size_tensor = _input_tensor.bounding_shape()[0]
                    break
            else:
                raise RuntimeError(
                    "Tensor `{}` could not be found. "
                    "Make sure you are using tf.data API".format("IteratorGetNext")
                )

        except RuntimeError:
            raise

        except Exception as e:
            raise RuntimeError(
                "Impossible to fetch the tensor: `IteratorGetNext`. Make sure you are using tf.data API."
            ) from e

        self._global_step_tensor = tf.compat.v1.train.get_or_create_global_step()

        try:
            self._AMP_loss_scale_tensor = _graph.get_tensor_by_name("current_loss_scale/Read/ReadVariableOp:0")
            self._AMP_steps_since_last_loss_scale = _graph.get_tensor_by_name("current_loss_scale/Read/ReadVariableOp:0")

        except RuntimeError:
            raise

        # TF-AMP is not activated
        except Exception:
            pass

        # if self._is_training:
        #     self.runtime_data["params_count"] = tf.reduce_sum(
        #         [tf.reduce_prod(v.shape) for v in tf.trainable_variables()]
        #     )

    def end(self, session):  # pylint: disable=unused-argument
        """Called at the end of session.
        The `session` argument can be used in case the hook wants to run final ops,
        such as saving a last checkpoint.
        If `session.run()` raises exception other than OutOfRangeError or
        StopIteration then `end()` is not called.
        Note the difference between `end()` and `after_run()` behavior when
        `session.run()` raises OutOfRangeError or StopIteration. In that case
        `end()` is called but `after_run()` is not called.
        Args:
          session: A TensorFlow Session that will be soon closed.
        """

        self._session_run_times.append(time.time() - self._session_t0)

    def after_create_session(self, session, coord):  # pylint: disable=unused-argument3
        """Called when new TensorFlow session is created.
        This is called to signal the hooks that a new session has been created. This
        has two essential differences with the situation in which `begin` is called:
        * When this is called, the graph is finalized and ops can no longer be added
            to the graph.
        * This method will also be called as a result of recovering a wrapped
            session, not only at the beginning of the overall session.
        Args:
          session: A TensorFlow Session that has been created.
          coord: A Coordinator object which keeps track of all threads.
        """

        # ========= Collect the number of GPUs ======== #
        if self._is_training:

            if MPI_is_distributed():
                self._n_gpus = MPI_size()

            elif tf.distribute.has_strategy():
                self._n_gpus = tf.distribute.get_strategy().num_replicas_in_sync

            else:
                self._n_gpus = 1

        else:
            self._n_gpus = 1

        # =========== TensorFlow Hook Setup =========== #
        _global_step, _metrics = setup_tensorflow_hook(
            sess=session,
            logging_proxy=self._logging_proxy,
            is_training=self._is_training,
            is_initialized=self._initialized
        )

        if _global_step >= 0:
            self._current_step = self._amp_steps_non_skipped = _global_step

        self._metrics.update(_metrics)

        if not self._is_training:

            for metric_name in self._metrics.keys():
                self._metrics[metric_name]["aggregator"].reset()

        self._initialized = True
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

        self._session_t0 = time.time()

    def before_run(self, run_context):  # pylint: disable=unused-argument
        """Called before each call to run().
        You can return from this call a `SessionRunArgs` object indicating ops or
        tensors to add to the upcoming `run()` call.  These ops/tensors will be run
        together with the ops/tensors originally passed to the original run() call.
        The run args you return can also contain feeds to be added to the run()
        call.
        The `run_context` argument is a `SessionRunContext` that provides
        information about the upcoming `run()` call: the originally requested
        op/tensors, the TensorFlow Session.
        At this point graph is finalized and you can not add ops.
        Args:
          run_context: A `SessionRunContext` object.
        Returns:
          None or a `SessionRunArgs` object.
        """

        self._current_step += 1

        request_fetches = {
            "global_step": self._global_step_tensor, "metrics": dict(), "batch_size": self._batch_size_tensor
        }

        if self._is_training and self._AMP_steps_since_last_loss_scale is not None:
            request_fetches["AMP"] = {
                "steps_since_last_loss_scale": self._AMP_steps_since_last_loss_scale,
                "current_loss_scale": self._AMP_loss_scale_tensor,
            }

        if self._current_step % self._log_every_n_steps == 0:
            for key, value in self._metrics.items():
                request_fetches["metrics"][key] = value["tensor"]

        self._step_t0 = time.time()

        return tf.estimator.SessionRunArgs(request_fetches)

    def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
        """Called after each call to run().
        The `run_values` argument contains results of requested ops/tensors by
        `before_run()`.
        The `run_context` argument is the same one send to `before_run` call.
        `run_context.request_stop()` can be called to stop the iteration.
        If `session.run()` raises any exceptions then `after_run()` is not called.
        Args:
          run_context: A `SessionRunContext` object.
          run_values: A SessionRunValues object.
        """

        batch_time = time.time() - self._step_t0

        _global_step = run_values.results["global_step"]

        if self._is_training and self._AMP_steps_since_last_loss_scale is not None:

            try:
                AMP_steps_since_last_loss_scale = run_values.results["AMP"]["steps_since_last_loss_scale"]
                AMP_loss_scale = run_values.results["AMP"]["current_loss_scale"]

            except KeyError:
                AMP_steps_since_last_loss_scale = None
                AMP_loss_scale = None

            if AMP_steps_since_last_loss_scale is not None:

                # Step has been skipped
                if _global_step != (self._amp_steps_non_skipped + 1):
                    logging.warning(
                        "AMP - Training iteration `#{step}` has been skipped and loss rescaled. "
                        "New Loss Scale: {loss_scale}\n".format(step=self._current_step, loss_scale=AMP_loss_scale)
                    )

                else:
                    self._amp_steps_non_skipped += 1

                    if AMP_steps_since_last_loss_scale == 0:
                        logging.warning(
                            "AMP - Training iteration `#{step}` - Loss scale has been automatically increased. "
                            "New Loss Scale: {loss_scale}\n".format(step=self._current_step, loss_scale=AMP_loss_scale)
                        )

        else:
            AMP_steps_since_last_loss_scale = None
            AMP_loss_scale = None

        def get_model_throughput():
            gpu_batch_size = run_values.results["batch_size"]
            return gpu_batch_size / batch_time * self._n_gpus

        # def get_model_stats():
        #     return get_tf_model_statistics(batch_size=run_values.results["batch_size"], scope_name=None)
        #
        # if self._model_stats is None:
        #     self._model_stats = get_model_stats()

        is_log_step = self._current_step % self._log_every_n_steps == 0

        if is_log_step:

            if self._current_step > self._warmup_steps:

                try:
                    model_throughput = self._model_throughput.read()
                except ValueError:
                    model_throughput = get_model_throughput()

            else:
                model_throughput = get_model_throughput()

            self._logging_proxy.log_step(iteration=self._current_step, throughput=model_throughput, gpu_stats=[])

            self._logging_proxy.log_amp_runtime(
                current_loss_scale=AMP_loss_scale,
                steps_non_skipped=_global_step,
                steps_since_last_scale=AMP_steps_since_last_loss_scale,
            )

            metric_data = dict()

            for name, value in sorted(run_values.results["metrics"].items(), key=operator.itemgetter(0)):
                self._metrics[name]["aggregator"].record(value)

                metric_data[name] = self._metrics[name]["aggregator"].read()

            self._logging_proxy.log_metrics(
                metric_data=metric_data, iteration=self._current_step, runtime_mode=self._runtime_mode
            )

            print()  # Visual Spacing

        elif self._current_step > self._warmup_steps:
            # Do not store speed for log step due to additional fetches
            self._model_throughput.record(get_model_throughput())


class _SlaveGPUsHook(tf.estimator.SessionRunHook):

    def after_create_session(self, session, coord):

        with logging.temp_verbosity(logging.INFO):  # Do not warn user about metric cleaning
            clear_registered_metrics()


def real_autologging_hook(*args, **kwargs):

    replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group

    # Do not set a logging hook for GPUs != 0
    if MPI_rank_and_size()[0] != 0 or (isinstance(replica_id, tf.Tensor) and tf.get_static_value(replica_id) != 0):
        return _SlaveGPUsHook()

    else:
        _ = LoggingBackend()  # Making sure the backend is defined before any hook due to __atexit__ hook
        return _AutoLoggingHook(*args, **kwargs)


def collect_registered_metrics():

    if TF_METRICS:  # if not empty

        metrics = copy.copy(TF_METRICS)

        # Do not warn user about metric cleaning
        with logging.temp_verbosity(logging.INFO):
            clear_registered_metrics()

        return metrics

    else:
        return dict()


def get_model_variables():
    """return model variables: global variables without optimizer's variables"""

    return [
        # yapf: disable
        var for var in tf.compat.v1.global_variables() if (
            var.name[-11:] not in "/Momentum:0" and
            var.name[-11:] not in "/Adadelta:0" and
            var.name[-13:] not in "/Adadelta_1:0" and
            var.name[-7:] not in "/Adam:0" and
            var.name[-9:] not in "/Adam_1:0" and
            var.name[-10:] not in "/Adagrad:0" and
            var.name[10:] not in "/RMSProp:0" and
            var.name[-12:] not in "/RMSProp_1:0" and
            var.name[-16:] not in "/LARSOptimizer:0"
        )
        # yapf: enable
    ]


def get_trainable_variables():
    """Get a list of trainable TensorFlow variables.

    Parameters
    ----------
    train_only : boolean
        If True, only get the trainable variables.

    Returns
    -------
    list of Tensor
        A list of trainable TensorFlow variables

    Examples
    --------

    """
    if KERAS_MODELS or LooseVersion(tf.__version__) >= LooseVersion("2.0.0"):
        logging.warning(
            "In TF2.x, only trainable variables created with Keras Models are captured for logging.\n"
            "In TF1.x, if any keras model is defined. Only variables created inside Keras Models will be logged."
        )

        var_list = list()

        for model in KERAS_MODELS:
            var_list.extend(model.trainable_variables)

        # Keep only a list of unique variables (remove potential duplicates)
        var_list = list(set(var_list))

        # clearing the list of Keras Model to avoid memory leaks
        KERAS_MODELS.clear()

        return [var for var in sorted(var_list, key=lambda v: v.name)]

    else:
        # return tf.trainable_variables()  # deprecated in TF2.x
        from tensorflow.python.keras.backend import get_graph
        return get_graph().get_collection('trainable_variables')


def setup_tensorflow_hook(sess, logging_proxy, is_training, is_initialized):

    global_step = -1

    if is_training:

        if not is_initialized:

            _global_step_tensor = tf.compat.v1.train.get_or_create_global_step()

            global_step = sess.run(_global_step_tensor)

            trainable_variables = get_trainable_variables()

            def count_weights_in_varlist(var_list):
                return np.sum([np.prod(s.get_shape()) for s in var_list])

            logging_proxy.log_git_status()

            logging_proxy.log_model_statistics(
                model_statistics={
                    "# Trainable Weights": "{:,}".format(int(count_weights_in_varlist(trainable_variables))),
                    "# Model Weights": "{:,}".format(int(count_weights_in_varlist(get_model_variables()))),
                }
            )

            logging_proxy.log_trainable_variables([(var.name, var.get_shape()) for var in trainable_variables])

    else:

        if not is_initialized:
            global_step = 0

    metrics = collect_registered_metrics()

    logging_proxy.log_runtime(is_train=is_training)

    return global_step, metrics


AutoLoggingHook = lambda *args, **kwargs: real_autologging_hook(*args, **kwargs)
