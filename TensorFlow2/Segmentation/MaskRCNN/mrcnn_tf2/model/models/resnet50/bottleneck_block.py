import tensorflow as tf

from mrcnn_tf2.model.models.resnet50 import Conv2DBlock


class BottleneckBlock(tf.keras.layers.Layer):

    def __init__(self, filters, strides, expansion=1, shortcut='conv2d', trainable=True, *args, **kwargs):
        super().__init__(trainable=trainable, *args, **kwargs)

        if shortcut == 'conv2d':
            self.shortcut = Conv2DBlock(
                filters=filters * expansion,
                kernel_size=1,
                strides=strides,
                use_batch_norm=True,
                use_relu=False,  # Applied at the end after addition with bottleneck
                name='shortcut'
            )
        elif shortcut == 'avg_pool':
            self.shortcut = tf.keras.layers.AveragePooling2D(
                pool_size=1,
                strides=strides,
                name='shortcut'
            )
        else:
            self.shortcut = tf.keras.layers.Layer(name='shortcut')  # identity

        self.conv2d_1 = Conv2DBlock(
            filters=filters,
            kernel_size=1,
            strides=1,
            use_batch_norm=True,
            use_relu=True
        )
        self.conv2d_2 = Conv2DBlock(
            filters=filters,
            kernel_size=3,
            strides=strides,
            use_batch_norm=True,
            use_relu=True
        )
        self.conv2d_3 = Conv2DBlock(
            filters=filters * expansion,
            kernel_size=1,
            strides=1,
            use_batch_norm=True,
            use_relu=False  # Applied at the end after addition with shortcut
        )

        self.add = tf.keras.layers.Add()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=None, **kwargs):
        shortcut = self.shortcut(inputs)

        bottleneck = self.conv2d_1(inputs, training=training)
        bottleneck = self.conv2d_2(bottleneck, training=training)
        bottleneck = self.conv2d_3(bottleneck, training=training)

        net = self.add([bottleneck, shortcut])
        net = self.relu(net)

        return net
