import tensorflow as tf

from mrcnn_tf2.model.models.resnet50 import BottleneckGroup, Conv2DBlock


class ResNet50(tf.keras.Model):

    def __init__(self, name='resnet50', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.conv2d = Conv2DBlock(
            filters=64,
            kernel_size=7,
            strides=2,
            use_batch_norm=True,
            use_relu=True,
            trainable=False
        )
        self.maxpool2d = tf.keras.layers.MaxPool2D(
            pool_size=3,
            strides=2,
            padding='SAME'
        )
        self.group_1 = BottleneckGroup(
            blocks=3,
            filters=64,
            strides=1,
            trainable=False
        )
        self.group_2 = BottleneckGroup(
            blocks=4,
            filters=128,
            strides=2
        )
        self.group_3 = BottleneckGroup(
            blocks=6,
            filters=256,
            strides=2
        )
        self.group_4 = BottleneckGroup(
            blocks=3,
            filters=512,
            strides=2
        )

    def call(self, inputs, training=None, mask=None):

        net = self.conv2d(inputs, training=training)
        net = self.maxpool2d(net)
        c2 = self.group_1(net, training=training)
        c3 = self.group_2(c2, training=training)
        c4 = self.group_3(c3, training=training)
        c5 = self.group_4(c4, training=training)

        return {2: c2, 3: c3, 4: c4, 5: c5}

    def get_config(self):
        pass
