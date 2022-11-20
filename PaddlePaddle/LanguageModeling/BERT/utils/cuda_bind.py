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
import ctypes

_cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')

_cudart = ctypes.CDLL(os.path.join(_cuda_home, 'lib64/libcudart.so'))


def cuda_profile_start():
    _cudart.cudaProfilerStart()


def cuda_profile_stop():
    _cudart.cudaProfilerStop()


_nvtx = ctypes.CDLL(os.path.join(_cuda_home, 'lib64/libnvToolsExt.so'))


def cuda_nvtx_range_push(name):
    _nvtx.nvtxRangePushW(ctypes.c_wchar_p(name))


def cuda_nvtx_range_pop():
    _nvtx.nvtxRangePop()
