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

from typing import Optional

import cupy as cp
import numpy as np


class BaseSeeder:
    """ Base seeder
    Args:
        seed (int): optional global seed
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value if value is not None else np.random.randint(0, 100)

    def reseed(self):
        """Sets the seed for the project"""
        np.random.seed(self.seed)
        cp.random.seed(self.seed)
