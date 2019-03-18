# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

__all__ = ['pad']


def pad(inputs, paddings, mode='CONSTANT', name='padding', constant_values=0):

    if mode.upper() not in ['CONSTANT', 'REFLECT', 'SYMMETRIC']:
        raise ValueError("Unknown padding mode: `%s` (accepted: ['CONSTANT', 'REFLECT', 'SYMMETRIC'])" % mode)

    net = tf.pad(inputs, paddings=paddings, mode=mode, name=name, constant_values=constant_values)

    return net
