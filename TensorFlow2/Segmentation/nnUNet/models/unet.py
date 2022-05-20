import tensorflow as tf

from models import layers


class UNet(tf.keras.Model):
    def __init__(
        self,
        input_shape,
        n_class,
        kernels,
        strides,
        normalization_layer,
        negative_slope,
        dimension,
        deep_supervision,
    ):
        super().__init__()
        self.dim = dimension
        self.n_class = n_class
        self.negative_slope = negative_slope
        self.norm = normalization_layer
        self.deep_supervision = deep_supervision
        filters = [min(2 ** (5 + i), 320 if dimension == 3 else 512) for i in range(len(strides))]
        self.filters = filters
        self.kernels = kernels
        self.strides = strides

        down_block = layers.ConvBlock
        self.input_block = self.get_conv_block(
            conv_block=down_block,
            filters=filters[0],
            kernel_size=kernels[0],
            stride=strides[0],
            input_shape=input_shape,
        )
        self.downsamples = self.get_block_list(
            conv_block=down_block, filters=filters[1:], kernels=kernels[1:-1], strides=strides[1:-1]
        )
        self.bottleneck = self.get_conv_block(
            conv_block=down_block, filters=filters[-1], kernel_size=kernels[-1], stride=strides[-1]
        )
        self.upsamples = self.get_block_list(
            conv_block=layers.UpsampleBlock,
            filters=filters[:-1][::-1],
            kernels=kernels[1:][::-1],
            strides=strides[1:][::-1],
        )
        self.output_block = self.get_output_block()
        if self.deep_supervision:
            self.deep_supervision_heads = [self.get_output_block(), self.get_output_block()]
        self.n_layers = len(self.upsamples) - 1

    def call(self, x, training=True):
        skip_connections = []
        out = self.input_block(x)
        skip_connections.append(out)

        for down_block in self.downsamples:
            out = down_block(out)
            skip_connections.append(out)

        out = self.bottleneck(out)

        decoder_outputs = []
        for up_block in self.upsamples:
            out = up_block(out, skip_connections.pop())
            decoder_outputs.append(out)

        out = self.output_block(out)

        if training and self.deep_supervision:
            out = [
                out,
                self.deep_supervision_heads[0](decoder_outputs[-2]),
                self.deep_supervision_heads[1](decoder_outputs[-3]),
            ]
        return out

    def get_output_block(self):
        return layers.OutputBlock(filters=self.n_class, dim=self.dim, negative_slope=self.negative_slope)

    def get_conv_block(self, conv_block, filters, kernel_size, stride, **kwargs):
        return conv_block(
            dim=self.dim,
            stride=stride,
            norm=self.norm,
            kernel_size=kernel_size,
            filters=filters,
            negative_slope=self.negative_slope,
            **kwargs,
        )

    def get_block_list(self, conv_block, filters, kernels, strides):
        layers = []
        for filter, kernel, stride in zip(filters, kernels, strides):
            conv_layer = self.get_conv_block(conv_block, filter, kernel, stride)
            layers.append(conv_layer)
        return layers
