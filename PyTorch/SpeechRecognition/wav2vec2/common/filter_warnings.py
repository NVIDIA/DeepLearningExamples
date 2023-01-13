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

import warnings


# NGC 22.04-py3 container (PyTorch 1.12.0a0+bd13bc6)
warnings.filterwarnings(
    "ignore",
    message='positional arguments and argument "destination" are deprecated.'
            ' nn.Module.state_dict will not accept them in the future.')

# NGC ~22.05-py3
warnings.filterwarnings(
    "ignore", message="pyprof will be removed by the end of June, 2022")

# 22.08-py3 RC
warnings.filterwarnings(
    "ignore",
    message="is_namedtuple is deprecated, please use the python checks")
