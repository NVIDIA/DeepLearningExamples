# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from utils.utils import get_config_file, get_task_code, print0

from data_loading.dali_loader import fetch_dali_loader


class DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_path = get_data_path(args)
        self.kfold = get_kfold_splitter(args.nfolds)
        self.kwargs = {
            "dim": self.args.dim,
            "seed": self.args.seed,
            "gpus": self.args.gpus,
            "nvol": self.args.nvol,
            "layout": self.args.layout,
            "overlap": self.args.overlap,
            "benchmark": self.args.benchmark,
            "num_workers": self.args.num_workers,
            "oversampling": self.args.oversampling,
            "test_batches": self.args.test_batches,
            "train_batches": self.args.train_batches,
            "invert_resampled_y": self.args.invert_resampled_y,
            "patch_size": get_config_file(self.args)["patch_size"],
        }
        self.train_imgs, self.train_lbls, self.val_imgs, self.val_lbls, self.test_imgs = ([],) * 5

    def setup(self, stage=None):
        meta = load_data(self.data_path, "*_meta.npy")
        orig_lbl = load_data(self.data_path, "*_orig_lbl.npy")
        imgs, lbls = load_data(self.data_path, "*_x.npy"), load_data(self.data_path, "*_y.npy")
        self.test_imgs, test_meta = get_test_fnames(self.args, self.data_path, meta)

        if self.args.exec_mode != "predict" or self.args.benchmark:
            train_idx, val_idx = list(self.kfold.split(imgs))[self.args.fold]
            orig_lbl, meta = get_split(orig_lbl, val_idx), get_split(meta, val_idx)
            self.kwargs.update({"orig_lbl": orig_lbl, "meta": meta})
            self.train_imgs, self.train_lbls = get_split(imgs, train_idx), get_split(lbls, train_idx)
            self.val_imgs, self.val_lbls = get_split(imgs, val_idx), get_split(lbls, val_idx)

            if self.args.gpus > 1:
                rank = int(os.getenv("LOCAL_RANK", "0"))
                self.val_imgs = self.val_imgs[rank :: self.args.gpus]
                self.val_lbls = self.val_lbls[rank :: self.args.gpus]
        else:
            self.kwargs.update({"meta": test_meta})
        print0(f"{len(self.train_imgs)} training, {len(self.val_imgs)} validation, {len(self.test_imgs)} test examples")

    def train_dataloader(self):
        return fetch_dali_loader(self.train_imgs, self.train_lbls, self.args.batch_size, "train", **self.kwargs)

    def val_dataloader(self):
        return fetch_dali_loader(self.val_imgs, self.val_lbls, 1, "eval", **self.kwargs)

    def test_dataloader(self):
        if self.kwargs["benchmark"]:
            return fetch_dali_loader(self.train_imgs, self.train_lbls, self.args.val_batch_size, "test", **self.kwargs)
        return fetch_dali_loader(self.test_imgs, None, 1, "test", **self.kwargs)


def get_split(data, idx):
    return list(np.array(data)[idx])


def load_data(path, files_pattern, non_empty=True):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    if non_empty:
        assert len(data) > 0, f"No data found in {path} with pattern {files_pattern}"
    return data


def get_kfold_splitter(nfolds):
    return KFold(n_splits=nfolds, shuffle=True, random_state=12345)


def get_test_fnames(args, data_path, meta=None):
    kfold = get_kfold_splitter(args.nfolds)
    test_imgs = load_data(data_path, "*_x.npy", non_empty=False)
    if args.exec_mode == "predict" and "val" in data_path:
        _, val_idx = list(kfold.split(test_imgs))[args.fold]
        test_imgs = sorted(get_split(test_imgs, val_idx))
        if meta is not None:
            meta = sorted(get_split(meta, val_idx))
    return test_imgs, meta


def get_data_path(args):
    if args.data != "/data":
        return args.data
    data_path = os.path.join(args.data, get_task_code(args))
    if args.exec_mode == "predict" and not args.benchmark:
        data_path = os.path.join(data_path, "test")
    return data_path
