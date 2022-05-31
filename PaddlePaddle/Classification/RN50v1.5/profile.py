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
import logging
from contextlib import contextmanager
from utils.cuda_bind import cuda_profile_start, cuda_profile_stop
from utils.cuda_bind import cuda_nvtx_range_push, cuda_nvtx_range_pop


class Profiler:
    def __init__(self):
        super().__init__()
        self._enable_profile = int(os.environ.get('ENABLE_PROFILE', 0))
        self._start_step = int(os.environ.get('PROFILE_START_STEP', 0))
        self._stop_step = int(os.environ.get('PROFILE_STOP_STEP', 0))

        if self._enable_profile:
            log_msg = f"Profiling start at {self._start_step}-th and stop at {self._stop_step}-th iteration"
            logging.info(log_msg)

    def profile_setup(self, step):
        """
        Setup profiling related status.

        Args:
            step (int): the index of iteration.
        Return:
            stop (bool): a signal to indicate whether profiling should stop or not.
        """

        if self._enable_profile and step == self._start_step:
            cuda_profile_start()
            logging.info("Profiling start at %d-th iteration",
                         self._start_step)

        if self._enable_profile and step == self._stop_step:
            cuda_profile_stop()
            logging.info("Profiling stop at %d-th iteration", self._stop_step)
            return True
        return False

    def profile_tag_push(self, step, msg):
        if self._enable_profile and \
           step >= self._start_step and \
           step < self._stop_step:
            tag_msg = f"Iter-{step}-{msg}"
            cuda_nvtx_range_push(tag_msg)

    def profile_tag_pop(self):
        if self._enable_profile:
            cuda_nvtx_range_pop()

    @contextmanager
    def profile_tag(self, step, msg):
        self.profile_tag_push(step, msg)
        yield
        self.profile_tag_pop()
