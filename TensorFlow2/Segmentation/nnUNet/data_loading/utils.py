# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import glob
import os

import numpy as np
from runtime.utils import get_task_code
from sklearn.model_selection import KFold


def get_split(data, idx):
    return list(np.array(data)[idx])


def load_data(path, files_pattern):
    return sorted(glob.glob(os.path.join(path, files_pattern)))


def get_path(args):
    data_path = str(args.data)
    if data_path != "/data":
        return data_path
    data_path = os.path.join(data_path, get_task_code(args))
    if args.exec_mode == "predict" and not args.benchmark:
        data_path = os.path.join(data_path, "test")
    return data_path


def get_test_fnames(args, data_path, meta=None):
    kfold = KFold(n_splits=args.nfolds, shuffle=True, random_state=12345)
    test_imgs = load_data(data_path, "*_x.npy")

    if args.exec_mode == "predict" and "val" in data_path:
        _, val_idx = list(kfold.split(test_imgs))[args.fold]
        test_imgs = sorted(get_split(test_imgs, val_idx))
        if meta is not None:
            meta = sorted(get_split(meta, val_idx))

    return test_imgs, meta
