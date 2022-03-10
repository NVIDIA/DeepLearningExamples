# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

""" UNet3D model construction """
from model.layers import downsample_block, upsample_block, output_layer, input_block


class Builder: # pylint: disable=R0903
    """ Model builder """
    def __init__(self, n_classes, mode, normalization='none'):
        """ Configure the unet3d builder

        :param n_classes: Number of output channels
        :param mode: Estimator's execution mode
        :param normalization: Name of the normalization layer
        """
        self._n_classes = n_classes
        self._mode = mode
        self._normalization = normalization

    def __call__(self, features):
        """ Build UNet3D

        :param features: Input features
        :return: Output of the graph
        """
        skip_128 = input_block(inputs=features,
                               out_channels=32,
                               normalization=self._normalization,
                               mode=self._mode)

        skip_64 = downsample_block(inputs=skip_128,
                                   out_channels=64,
                                   normalization=self._normalization,
                                   mode=self._mode)

        skip_32 = downsample_block(inputs=skip_64,
                                   out_channels=128,
                                   normalization=self._normalization,
                                   mode=self._mode)

        skip_16 = downsample_block(inputs=skip_32,
                                   out_channels=256,
                                   normalization=self._normalization,
                                   mode=self._mode)

        skip_8 = downsample_block(inputs=skip_16,
                                  out_channels=320,
                                  normalization=self._normalization,
                                  mode=self._mode)

        out = downsample_block(inputs=skip_8,
                               out_channels=320,
                               normalization=self._normalization,
                               mode=self._mode)

        out = upsample_block(out, skip_8,
                             out_channels=320,
                             normalization=self._normalization,
                             mode=self._mode)

        out = upsample_block(out, skip_16,
                             out_channels=256,
                             normalization=self._normalization,
                             mode=self._mode)

        out = upsample_block(out, skip_32,
                             out_channels=128,
                             normalization=self._normalization,
                             mode=self._mode)

        out = upsample_block(out, skip_64,
                             out_channels=64,
                             normalization=self._normalization,
                             mode=self._mode)

        out = upsample_block(out, skip_128,
                             out_channels=32,
                             normalization=self._normalization,
                             mode=self._mode)

        return output_layer(out,
                            out_channels=self._n_classes,
                            activation='softmax')
