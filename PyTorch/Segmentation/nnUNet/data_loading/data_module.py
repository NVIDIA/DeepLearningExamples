import glob
import os
from subprocess import call

import numpy as np
from joblib import Parallel, delayed
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils.utils import get_config_file, get_task_code, is_main_process, make_empty_dir

from data_loading.dali_loader import fetch_dali_loader


class DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tfrecords_train = []
        self.tfrecords_val = []
        self.tfrecords_test = []
        self.train_idx = []
        self.val_idx = []
        self.test_idx = []
        self.kfold = KFold(n_splits=self.args.nfolds, shuffle=True, random_state=12345)
        self.data_path = os.path.join(self.args.data, get_task_code(self.args))
        if self.args.exec_mode == "predict" and not args.benchmark:
            self.data_path = os.path.join(self.data_path, "test")
        configs = get_config_file(self.args)
        self.kwargs = {
            "dim": self.args.dim,
            "patch_size": configs["patch_size"],
            "seed": self.args.seed,
            "gpus": self.args.gpus,
            "num_workers": self.args.num_workers,
            "oversampling": self.args.oversampling,
            "create_idx": self.args.create_idx,
            "benchmark": self.args.benchmark,
        }

    def prepare_data(self):
        if self.args.create_idx:
            tfrecords_train, tfrecords_val, tfrecords_test = self.load_tfrecords()
            make_empty_dir("train_idx")
            make_empty_dir("val_idx")
            make_empty_dir("test_idx")
            self.create_idx("train_idx", tfrecords_train)
            self.create_idx("val_idx", tfrecords_val)
            self.create_idx("test_idx", tfrecords_test)

    def setup(self, stage=None):
        self.tfrecords_train, self.tfrecords_val, self.tfrecords_test = self.load_tfrecords()
        self.train_idx, self.val_idx, self.test_idx = self.load_idx_files()
        if is_main_process():
            ntrain, nval, ntest = len(self.tfrecords_train), len(self.tfrecords_val), len(self.tfrecords_test)
            print(f"Number of examples: Train {ntrain} - Val {nval} - Test {ntest}")

    def train_dataloader(self):
        return fetch_dali_loader(self.tfrecords_train, self.train_idx, self.args.batch_size, "training", **self.kwargs)

    def val_dataloader(self):
        return fetch_dali_loader(self.tfrecords_val, self.val_idx, 1, "eval", **self.kwargs)

    def test_dataloader(self):
        if self.kwargs["benchmark"]:
            return fetch_dali_loader(
                self.tfrecords_train, self.train_idx, self.args.val_batch_size, "eval", **self.kwargs
            )
        return fetch_dali_loader(self.tfrecords_test, self.test_idx, 1, "test", **self.kwargs)

    def load_tfrecords(self):
        if self.args.dim == 2:
            train_tfrecords = self.load_data(self.data_path, "*.train_tfrecord")
            val_tfrecords = self.load_data(self.data_path, "*.val_tfrecord")
        else:
            train_tfrecords = self.load_data(self.data_path, "*.tfrecord")
            val_tfrecords = self.load_data(self.data_path, "*.tfrecord")

        train_idx, val_idx = list(self.kfold.split(train_tfrecords))[self.args.fold]
        train_tfrecords = self.get_split(train_tfrecords, train_idx)
        val_tfrecords = self.get_split(val_tfrecords, val_idx)

        return train_tfrecords, val_tfrecords, self.load_data(os.path.join(self.data_path, "test"), "*.tfrecord")

    def load_idx_files(self):
        if self.args.create_idx:
            test_idx = sorted(glob.glob(os.path.join("test_idx", "*.idx")))
        else:
            test_idx = self.get_idx_list("test/idx_files", self.tfrecords_test)

        if self.args.create_idx:
            train_idx = sorted(glob.glob(os.path.join("train_idx", "*.idx")))
            val_idx = sorted(glob.glob(os.path.join("val_idx", "*.idx")))
        elif self.args.dim == 3:
            train_idx = self.get_idx_list("idx_files", self.tfrecords_train)
            val_idx = self.get_idx_list("idx_files", self.tfrecords_val)
        else:
            train_idx = self.get_idx_list("train_idx_files", self.tfrecords_train)
            val_idx = self.get_idx_list("val_idx_files", self.tfrecords_val)
        return train_idx, val_idx, test_idx

    def create_idx(self, idx_dir, tfrecords):
        idx_files = [os.path.join(idx_dir, os.path.basename(tfrec).split(".")[0] + ".idx") for tfrec in tfrecords]
        Parallel(n_jobs=-1)(
            delayed(self.tfrecord2idx)(tfrec, tfidx)
            for tfrec, tfidx in tqdm(zip(tfrecords, idx_files), total=len(tfrecords))
        )

    def get_idx_list(self, dir_name, tfrecords):
        root_dir = os.path.join(self.data_path, dir_name)
        return sorted([os.path.join(root_dir, os.path.basename(tfr).split(".")[0] + ".idx") for tfr in tfrecords])

    @staticmethod
    def get_split(data, idx):
        return list(np.array(data)[idx])

    @staticmethod
    def load_data(path, files_pattern):
        return sorted(glob.glob(os.path.join(path, files_pattern)))

    @staticmethod
    def tfrecord2idx(tfrecord, tfidx):
        call(["tfrecord2idx", tfrecord, tfidx])
