import tensorflow as tf


class PiecewiseConstantWithWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Schedule that starts with learning rate at `init_value` and monotonically increases
    it up to `values[0]` at step `boundaries[0]`. After that the learning rate changes
    on each boundary to corresponding value.
    """

    def __init__(self, init_value, boundaries, values, scale=1.0, name='PiecewiseConstantWithWarmup'):
        """
        Constructs piecewise constant learning rate with linear warmup.
        Args:
            init_value (float): Learning rate at step 0.
            boundaries (List[int]): Steps at which the learning rate will change.
            values (List[float]): Values to which the learning rate will be changed.
            scale (float): Scales the computed lr by given constant.
            name (str): Name of the operation.
        """
        assert len(values) > 0
        assert len(values) == len(boundaries)

        self._init_value = float(init_value)
        self._values = list(map(float, values))
        self._boundaries = list(map(float, boundaries))
        self._scale = float(scale)
        self._name = name

    def __call__(self, step):
        with tf.name_scope(self._name):
            # linear learning rate before first boundary
            warmup_lr = self._init_value + (self._values[0] - self._init_value) * (step / self._boundaries[0])
            warmup_pred = (tf.less(step, self._boundaries[0]), lambda: warmup_lr)

            # step learning rate after first boundary
            boundaries_pred = [
                (tf.less(step, limit), lambda v=v: v)
                for limit, v in zip(self._boundaries[1:], self._values)
            ]

            learning_rate = tf.case(
                pred_fn_pairs=[warmup_pred] + boundaries_pred,
                default=lambda: self._values[-1]
            )

            return learning_rate * self._scale

    def get_config(self):
        return {
            "init_value": self._init_value,
            "values": self._values,
            "boundaries": self._boundaries,
            "name": self._name
        }
