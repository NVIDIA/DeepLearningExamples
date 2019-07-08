import tensorflow as tf
import re

class MixedPrecisionOptimizer(tf.train.Optimizer):
    """An optimizer that wraps another tf.Optimizer, maintaining shadow fp32
    weights for mixed precision training."""

    def __init__(self, optimizer, name=None, use_locking=False):
        """Construct a new MixedPrecisionOptimizer, which uses another optimizer
        under the hood for computing single-process gradient values and
        applying gradient updates to the shadow fp32 weights.

        Args:
          optimizer:
            Optimizer to use for computing gradients and applying updates.
          name:
            Optional name prefix for the operations created when applying
            gradients. Defaults to "Distributed" followed by the provided
            optimizer type.
          use_locking:
            Whether to use locking when updating variables.
            See Optimizer.__init__ for more info.
        """
        if name is None:
            name = "Distributed{}".format(type(optimizer).__name__)

        self._optimizer = optimizer
        super(MixedPrecisionOptimizer, self).__init__(name=name, use_locking=use_locking)

    def compute_gradients(self, loss, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.compute_gradients(loss*128.0, *args, **kwargs)
  
    def _get_variable_name(self, var_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", var_name)
        if m is not None:
            var_name = m.group(1)
        return var_name

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Apply computed gradients to all trainable variables.
        Create shadow fp32 variables for trainable fp16 variables.

        See Optimizer.apply_gradients() for more info.
        """
        fp32_gradients = []
        copy_afterwards = []
        with tf.name_scope(self._name + "_ShadowWeights"):
            for grad, var in grads_and_vars:
                if grad is not None:
                    if isinstance(grad, tf.IndexedSlices):
                        grad = tf.convert_to_tensor(grad)
                    if grad.dtype.base_dtype != tf.float32:
                        grad = tf.cast(grad, tf.float32)
                        # create var_fp32
                        var_name = self._get_variable_name(var.name)
                        var_fp32 = tf.get_variable(
                            name=var_name + "/shadow",
                            dtype=tf.float32,
                            trainable=False,
                            initializer=tf.cast(var.initialized_value(),tf.float32))
                        copy_afterwards.append((var, var_fp32))
                        fp32_gradients.append((grad/128.0, var_fp32))
                    else:
                        fp32_gradients.append((grad/128.0, var))
                else:
                    fp32_gradients.append((None, var))
        retval = self._optimizer.apply_gradients(fp32_gradients, *args, **kwargs)
        copies = []
        for var, var_fp32 in copy_afterwards:
            copies.append(var.assign(tf.cast(var_fp32, var.dtype.base_dtype)))
        with tf.control_dependencies([retval]):
            copies = tf.group(*copies, name='fp16_copies')
        return tf.group(retval, copies)

    def get_slot(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.variables(*args, **kwargs)
