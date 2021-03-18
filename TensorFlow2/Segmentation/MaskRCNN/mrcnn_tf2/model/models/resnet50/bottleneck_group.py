import tensorflow as tf

from mrcnn_tf2.model.models.resnet50 import BottleneckBlock


class BottleneckGroup(tf.keras.layers.Layer):

    def __init__(self, blocks, filters, strides, trainable=True):
        super().__init__(trainable=trainable)

        self.blocks = []

        for block_id in range(blocks):
            self.blocks.append(
                BottleneckBlock(
                    filters=filters,
                    strides=strides if block_id == 0 else 1,
                    expansion=4,
                    shortcut='conv2d' if block_id == 0 else None
                )
            )

    def call(self, inputs, training=None, **kwargs):
        net = inputs

        for block in self.blocks:
            net = block(net, training=training)

        return net
