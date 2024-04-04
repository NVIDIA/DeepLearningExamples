# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import OmegaConf
from data.data_utils import InputTypes, DataTypes, FeatureSpec
import functools
from hydra.utils import get_method

OmegaConf.register_new_resolver("and", lambda x, y: bool(x and y), use_cache=True)
OmegaConf.register_new_resolver("feature.selector",
        lambda x,feat_type,embed_type:
            OmegaConf.create([elem for elem in x if elem.feature_type == feat_type and elem.feature_embed_type == embed_type])
        )
OmegaConf.register_new_resolver("add", lambda x,y: x + y)
OmegaConf.register_new_resolver("if", lambda x,y,z: y if x else z)
OmegaConf.register_new_resolver("feature.cardinalities", lambda x: OmegaConf.create([elem.cardinality for elem in x]))
OmegaConf.register_new_resolver("len", len)
OmegaConf.register_new_resolver("cmp", lambda x, y: x == y)
OmegaConf.register_new_resolver("cont.lower", lambda x, y: y.lower() in x.lower())

def sum_nested(*args):
    if len(args) == 1 and isinstance(args[0], (int, float)):
        return args[0]
    return sum(arg if isinstance(arg, (int, float)) else sum_nested(*arg) for arg in args)

OmegaConf.register_new_resolver("sum", sum_nested)

def partial(func, *args, **kwargs):
    return functools.partial(get_method(func), *args, **kwargs)
