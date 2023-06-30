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
#
# author: Tomasz Grel (tgrel@nvidia.com)


import os


def get_variable_path(checkpoint_path, name):
    tokens = name.split('/')
    tokens = [t for t in tokens if 'model_parallel' not in t and 'data_parallel' not in t]
    name = '_'.join(tokens)
    name = name.replace(':', '_')
    filename = name + '.npy'
    return os.path.join(checkpoint_path, filename)
