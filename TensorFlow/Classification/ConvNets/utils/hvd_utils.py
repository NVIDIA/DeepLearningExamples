#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import os
import horovod.tensorflow as hvd


def is_using_hvd():
    return hvd.size() > 1
    rank_env = ['HOROVOD_RANK', 'OMPI_COMM_WORLD_RANK', 'PMI_RANK']
    size_env = ['HOROVOD_SIZE', 'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE']

    for r_var, s_var in zip(rank_env, size_env):
        if r_var in os.environ and s_var in os.environ:
            return int(s_var) > 1
    return False
