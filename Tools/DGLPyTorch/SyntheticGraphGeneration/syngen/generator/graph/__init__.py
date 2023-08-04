# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

# flake8: noqa
from .base_graph_generator import BaseGenerator, BaseGraphGenerator, BaseBipartiteGraphGenerator
from .rmat import RMATGenerator
from .rmat_bipartite import RMATBipartiteGenerator
from .random import RandomGraph
from .random_bipartite import RandomBipartite


def get_structural_generator_class(type, is_bipartite, is_random):
    if type == 'RMAT':
        rmats = {
            (True, True): RandomBipartite,
            (True, False): RMATBipartiteGenerator,
            (False, True): RandomGraph,
            (False, False): RMATGenerator
        }
        return rmats[(is_bipartite, is_random)]
    else:
        raise ValueError("unsupported generator type")

