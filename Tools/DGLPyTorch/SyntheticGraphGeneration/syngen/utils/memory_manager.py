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

import pynvml
import psutil


class MemoryManager(object):

    def __init__(self, gpus=None):
        pynvml.nvmlInit()

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(MemoryManager, cls).__new__(cls)
        return cls.instance

    def get_available_gpus(self):
        return pynvml.nvmlDeviceGetCount()

    def get_memory_info_on_gpu(self, gpu_id):
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        return pynvml.nvmlDeviceGetMemoryInfo(h)

    def get_min_available_across_gpus_memory(self, gpus):
        total = None
        used = 0
        for g_id in range(gpus):
            info = self.get_memory_info_on_gpu(g_id)
            if total is None:
                total = info.total
            else:
                assert total == info.total
            used = max(used, info.used)
        return total - used

    def get_available_virtual_memory(self):
        return psutil.virtual_memory().available
