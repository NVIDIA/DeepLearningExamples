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

from .activations import *
from .cond_conv2d import CondConv2d, get_condconv_initializer
from .config import is_exportable, is_scriptable, is_no_jit, set_exportable, set_scriptable, set_no_jit,\
    set_layer_config
from .conv2d_same import Conv2dSame
from .create_act import create_act_layer, get_act_layer, get_act_fn
from .create_conv2d import create_conv2d
from .drop import DropBlock2d, DropPath, drop_block_2d, drop_path
from .mixed_conv2d import MixedConv2d
from .padding import get_padding
from .pool2d_same import AvgPool2dSame, create_pool2d
from .nms_layer import batched_soft_nms, batched_nms