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

""" Model construction utils

This module provides a convenient way to create different topologies
based around UNet.

"""
import tensorflow as tf
from model.layers import output_block, upsample_block, bottleneck, downsample_block, input_block


def unet_v1(features,  mode):
    """ U-Net: Convolutional Networks for Biomedical Image Segmentation

    Source:
        https://arxiv.org/pdf/1505.04597

    """

    skip_connections = []

    out, skip = input_block(features, filters=64)

    skip_connections.append(skip)

    for idx, filters in enumerate([128, 256, 512]):
        out, skip = downsample_block(out, filters=filters, idx=idx)
        skip_connections.append(skip)

    out = bottleneck(out, filters=1024, mode=mode)

    for idx, filters in enumerate([512, 256, 128]):
        out = upsample_block(out,
                             residual_input=skip_connections.pop(),
                             filters=filters,
                             idx=idx)
    return output_block(out, residual_input=skip_connections.pop(), filters=64, n_classes=2)
