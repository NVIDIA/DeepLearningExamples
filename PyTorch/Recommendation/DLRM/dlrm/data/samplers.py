# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
import torch

from torch.utils.data import RandomSampler

from dlrm.utils.distributed import get_local_rank


class RandomDistributedSampler(RandomSampler):

    _SAMPLE_FILE = "/tmp/dlrm_training_sample.npy"

    def __iter__(self):
        """
        To guarantee all ranks have the same same permutation, generating it from rank 0 and sync
        to other rank by writing to disk
        """
        if get_local_rank() == 0:
            np.save(self._SAMPLE_FILE, np.array(super().__iter__()))
        torch.distributed.barrier()

        sample = np.load(self._SAMPLE_FILE)
        return iter(sample)
