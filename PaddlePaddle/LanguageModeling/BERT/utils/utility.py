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

import os
import sys
import random
import numpy as np
import paddle


def get_num_trainers():
    """Get number of trainers in distributed training."""
    num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    return num_trainers


def get_trainer_id():
    """Get index of trainer in distributed training."""
    trainer_id = int(os.environ.get('PADDLE_TRAINER_ID', 0))
    return trainer_id


def is_integer(number):
    """Whether a number is integer."""
    if sys.version > '3':
        return isinstance(number, int)
    return isinstance(number, (int, long))


def set_seed(seed):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
