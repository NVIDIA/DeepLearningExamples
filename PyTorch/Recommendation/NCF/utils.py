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
import json
from functools import reduce
import time

def count_parameters(model):
    c = map(lambda p: reduce(lambda x, y: x * y, p.size()), model.parameters())
    return sum(c)


def save_result(result, path):
    write_heading = not os.path.exists(path)
    with open(path, mode='a') as out:
        if write_heading:
            out.write(",".join([str(k) for k, v in result.items()]) + '\n')
        out.write(",".join([str(v) for k, v in result.items()]) + '\n')
