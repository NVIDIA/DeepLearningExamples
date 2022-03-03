# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import sys
import os
import h5py
import numpy as np

folder = sys.argv[1]
print(folder)
files = [os.path.join(folder, f) for f in os.listdir(folder) if
                         os.path.isfile(os.path.join(folder, f))]
counts = []
for input_file in files:
	f = h5py.File(input_file, "r")
	keys = ['input_ids']
	inputs = np.asarray(f[keys[0]][:])
	print(inputs.shape)
	counts.append(inputs.shape[0])
	f.close()
print(counts)
print(sum(counts))
print(len(counts))
