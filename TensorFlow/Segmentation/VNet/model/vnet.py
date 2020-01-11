# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from model.layers import input_block, downsample_block, upsample_block, output_block


class Builder():
    def __init__(self, kernel_size, n_classes, upscale_blocks, downscale_blocks, upsampling, pooling, normalization,
                 activation, mode):
        self._kernel_size = kernel_size
        self._pooling = pooling
        self._upsampling = upsampling
        self._normalization = normalization
        self._activation = activation
        self._mode = mode
        self._n_classes = n_classes
        self._downscale_blocks = downscale_blocks
        self._upscale_blocks = upscale_blocks

    def __call__(self, features):

        x = input_block(inputs=features,
                        filters=16,
                        kernel_size=self._kernel_size,
                        normalization=self._normalization,
                        activation=self._activation,
                        mode=self._mode)

        skip_connections = [x]

        for depth in self._downscale_blocks:
            x = downsample_block(inputs=x,
                                 depth=depth,
                                 kernel_size=self._kernel_size,
                                 pooling=self._pooling,
                                 normalization=self._normalization,
                                 activation=self._activation,
                                 mode=self._mode)

            skip_connections.append(x)

        del skip_connections[-1]

        for depth in self._upscale_blocks:
            x = upsample_block(inputs=x,
                               residual_inputs=skip_connections.pop(),
                               depth=depth,
                               upsampling=self._upsampling,
                               kernel_size=self._kernel_size,
                               normalization=self._normalization,
                               activation=self._activation,
                               mode=self._mode)

        return output_block(inputs=x,
                            residual_inputs=skip_connections.pop(),
                            kernel_size=self._kernel_size,
                            n_classes=self._n_classes,
                            upsampling=self._upsampling,
                            normalization=self._normalization,
                            activation=self._activation,
                            mode=self._mode)
