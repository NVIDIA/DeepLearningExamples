# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

"""Mutes known and unrelated PyTorch warnings.

The warnings module keeps a list of filters. Importing it as late as possible
prevents its filters from being overriden.
"""

import warnings


# NGC 22.04-py3 container (PyTorch 1.12.0a0+bd13bc6)
warnings.filterwarnings(
    "ignore",
    message='positional arguments and argument "destination" are deprecated.'
            ' nn.Module.state_dict will not accept them in the future.')

# 22.08-py3 container
warnings.filterwarnings(
    "ignore",
    message="is_namedtuple is deprecated, please use the python checks")
