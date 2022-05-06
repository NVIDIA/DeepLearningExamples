# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import horovod.tensorflow as hvd


def csv_str_to_int_list(s):
    """
    Example: '200,80' -> [200, 80]
    """
    return list(map(int, s.split(",")))


def dist_print(*args, force=False, **kwargs):
    if hvd.rank() == 0 or force:
        print(*args, **kwargs)