import tensorflow as tf


class Conv2DBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides, padding='SAME',
                 use_batch_norm=True, use_relu=True, trainable=True,
                 trainable_batch_norm=False, *args, **kwargs):
        super().__init__(trainable=trainable, *args, **kwargs)
        self.conv2d = None
        self.batch_norm = None
        self.relu = None

        self.conv2d = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=not use_batch_norm,
            trainable=trainable
        )

        if use_batch_norm:
            self.batch_norm = tf.keras.layers.BatchNormalization(
                momentum=0.9,
                scale=True,
                epsilon=1e-05,
                trainable=trainable and trainable_batch_norm,
                fused=True,
                center=True
            )

        if use_relu:
            self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=None, **kwargs):
        net = inputs

        net = self.conv2d(net)
        if self.batch_norm:
            net = self.batch_norm(net, training=training)
        if self.relu:
            net = self.relu(net)

        return net
