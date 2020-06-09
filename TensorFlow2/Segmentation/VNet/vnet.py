
import numpy as np

import tensorflow as tf
from layers import InputBlock, DownsampleBlock, UpsampleBlock, OutputBlock


class Vnet(tf.keras.Model):
    """
    https://arxiv.org/pdf/1606.04797.pdf
    V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    """
    def __init__(self, n_classes,
                 init_filters = 16,
                 kernel_size=3,
                 upscale_blocks=[3,3],downscale_blocks=[3,3,3],
                 upsampling='transposed_conv',
                 pooling='conv_pool',
                 normalization='batchnorm',
                 conv_activation=None,
                 final_activation='relu'):
        super().__init__(self)
        self._init_filters = init_filters
        self._kernel_size = kernel_size
        self._pooling = pooling
        self._upsampling = upsampling
        self._normalization = normalization
        self._conv_activation = conv_activation
        self._final_activation = final_activation
        self._n_classes = n_classes
        self._downscale_blocks = downscale_blocks
        self._upscale_blocks = upscale_blocks
        
        self.input_block = InputBlock(filters=self._init_filters, kernel_size=self._kernel_size, conv_activation=self._conv_activation, final_activation=self._final_activation, normalization=self._normalization)
        
        self.downsample_blocks = [DownsampleBlock(res_depth=d, res_kernel_size=self._kernel_size, down_pooling=self._pooling, normalization=self._normalization, conv_activation=self._conv_activation, final_activation=self._final_activation) for d in self._downscale_blocks]
        
        self.upsample_blocks = [UpsampleBlock(res_depth=d, upsampling=self._upsampling, res_kernel_size=self._kernel_size, conv_activation=self._conv_activation, final_activation=self._final_activation, normalization=self._normalization) for d in self._upscale_blocks]
        
        self.output_block = OutputBlock(n_classes=self._n_classes, con_kernel_size=self._kernel_size, upsampling=self._upsampling, up_normalization=self._normalization, up_conv_activation=self._conv_activation, up_final_activation=self._final_activation)
        
    def call(self, features, training=True):
        x = self.input_block(features, training=training)
        skip_connections = [x]
        for downsample_block in self.downsample_blocks:
            x = downsample_block(x, training=training)
            skip_connections.append(x)
        
        del skip_connections[-1]
        
        for upsample_block in self.upsample_blocks:
            x = upsample_block(x, residual_inputs=skip_connections.pop(), training=training)
        
        return self.output_block(x, residual_inputs=skip_connections.pop(), training=training)
    
    
if __name__ == '__main__':
    np.random.seed = 42
    a = np.random.randint(10, size = (1,7,7,7,1)).astype(np.float32)
    
    a = a = np.ones((1,7,7,7,1))
    
    
    for i in range(5):
        print (i)
        vnet = Vnet(n_classes=4, downscale_blocks=[3,3,3], normalization=None, conv_activation='relu',final_activation=None)
        res = vnet(a)
        