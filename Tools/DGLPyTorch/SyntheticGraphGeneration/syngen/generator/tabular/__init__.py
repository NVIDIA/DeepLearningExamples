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
from .base_tabular_generator import BaseTabularGenerator
from .chunked_tabular_generator import ChunkedBaseTabularGenerator
from .ctgan import CTGANGenerator
from .gaussian_generator import GaussianGenerator
from .kde_generator import KDEGenerator
from .random import RandomMVGenerator
from .uniform_generator import UniformGenerator

# Does not include CTGAN
tabular_generators_classes = {
    'kde': KDEGenerator,
    'random': RandomMVGenerator,
    'gaussian': GaussianGenerator,
    'uniform': UniformGenerator,
    'ctgan': CTGANGenerator,
}

tabular_generators_types_to_classes = {
    cls.__class__.__name__: k
    for k, cls in tabular_generators_classes
    .items()
}
