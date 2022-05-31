# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

from enum import Enum


class Mode(Enum):
    TRAIN = 'Train'
    EVAL = 'Eval'


class RunScope(Enum):
    TRAIN_ONLY = 'train_only'
    EVAL_ONLY = 'eval_only'
    TRAIN_EVAL = 'train_eval'
