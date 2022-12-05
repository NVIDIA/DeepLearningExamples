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

import horovod.tensorflow as hvd
from runtime.utils import get_config_file, is_main_process
from sklearn.model_selection import KFold

from data_loading.dali_loader import fetch_dali_loader
from data_loading.utils import get_path, get_split, get_test_fnames, load_data


class DataModule:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_imgs = []
        self.train_lbls = []
        self.val_imgs = []
        self.val_lbls = []
        self.test_imgs = []
        self.kfold = KFold(n_splits=self.args.nfolds, shuffle=True, random_state=12345)
        self.data_path = get_path(args)
        configs = get_config_file(self.args)
        self.patch_size = configs["patch_size"]
        self.kwargs = {
            "dim": self.args.dim,
            "patch_size": self.patch_size,
            "seed": self.args.seed,
            "gpus": hvd.size(),
            "num_workers": self.args.num_workers,
            "oversampling": self.args.oversampling,
            "benchmark": self.args.benchmark,
            "nvol": self.args.nvol,
            "bench_steps": self.args.bench_steps,
            "meta": load_data(self.data_path, "*_meta.npy"),
        }

    def setup(self, stage=None):
        imgs = load_data(self.data_path, "*_x.npy")
        lbls = load_data(self.data_path, "*_y.npy")

        self.test_imgs, self.kwargs["meta"] = get_test_fnames(self.args, self.data_path, self.kwargs["meta"])
        if self.args.exec_mode != "predict" or self.args.benchmark:
            train_idx, val_idx = list(self.kfold.split(imgs))[self.args.fold]
            self.train_imgs = get_split(imgs, train_idx)
            self.train_lbls = get_split(lbls, train_idx)
            self.val_imgs = get_split(imgs, val_idx)
            self.val_lbls = get_split(lbls, val_idx)
            if is_main_process():
                ntrain, nval = len(self.train_imgs), len(self.val_imgs)
                print(f"Number of examples: Train {ntrain} - Val {nval}")

            # Shard the validation data
            self.val_imgs = self.val_imgs[hvd.rank() :: hvd.size()]
            self.val_lbls = self.val_lbls[hvd.rank() :: hvd.size()]
            self.cached_val_loader = None
        elif is_main_process():
            print(f"Number of test examples: {len(self.test_imgs)}")

    def train_dataset(self):
        return fetch_dali_loader(
            self.train_imgs,
            self.train_lbls,
            self.args.batch_size,
            "train",
            **self.kwargs,
        )

    def train_size(self):
        return len(self.train_imgs)

    def val_dataset(self):
        if self.cached_val_loader is None:
            self.cached_val_loader = fetch_dali_loader(self.val_imgs, self.val_lbls, 1, "eval", **self.kwargs)
        return self.cached_val_loader

    def val_size(self):
        return len(self.val_imgs)

    def test_dataset(self):
        if self.kwargs["benchmark"]:
            return fetch_dali_loader(
                self.train_imgs,
                self.train_lbls,
                self.args.batch_size,
                "test",
                **self.kwargs,
            )
        return fetch_dali_loader(self.test_imgs, None, 1, "test", **self.kwargs)

    def test_size(self):
        return len(self.test_imgs)

    def test_fname(self, idx):
        return self.test_imgs[idx]
