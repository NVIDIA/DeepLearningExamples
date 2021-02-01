# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


import math
import os
from glob import glob
from subprocess import call

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from tqdm import tqdm
from utils.utils import get_task_code, make_empty_dir


class Converter:
    def __init__(self, args):
        self.args = args
        self.mode = self.args.exec_mode
        task_code = get_task_code(self.args)
        self.data = os.path.join(self.args.data, task_code)
        self.results = os.path.join(self.args.results, task_code)
        if self.mode == "test":
            self.data = os.path.join(self.data, "test")
            self.results = os.path.join(self.results, "test")
        self.vpf = self.args.vpf

        self.imgs = self.load_files("*x.npy")
        self.lbls = self.load_files("*y.npy")

    def run(self):
        print("Saving tfrecords...")
        suffix = "tfrecord" if self.args.dim == 3 else "val_tfrecord"
        self.save_tfrecords(self.imgs, self.lbls, dim=3, suffix=suffix)
        if self.args.dim == 2:
            self.save_tfrecords(self.imgs, self.lbls, dim=2, suffix="train_tfrecord")
            train_tfrecords, train_idx_dir = self.get_tfrecords_data("*.train_tfrecord", "train_idx_files")
            val_tfrecords, val_idx_dir = self.get_tfrecords_data("*.val_tfrecord", "val_idx_files")
            print("Saving idx files...")
            self.create_idx_files(train_tfrecords, train_idx_dir)
            self.create_idx_files(val_tfrecords, val_idx_dir)
        else:
            tfrecords, idx_dir = self.get_tfrecords_data("*.tfrecord", "idx_files")
            print("Saving idx files...")
            self.create_idx_files(tfrecords, idx_dir)

    def save_tfrecords(self, imgs, lbls, dim, suffix):
        if len(lbls) == 0:
            lbls = imgs[:]
        chunks = np.array_split(list(zip(imgs, lbls)), math.ceil(len(imgs) / self.args.vpf))
        Parallel(n_jobs=self.args.n_jobs)(
            delayed(self.convert2tfrec)(chunk, dim, suffix) for chunk in tqdm(chunks, total=len(chunks))
        )

    def convert2tfrec(self, files, dim, suffix):
        examples = []
        for img_path, lbl_path in files:
            img, lbl = np.load(img_path), np.load(lbl_path)
            if dim == 2:
                for depth in range(img.shape[1]):
                    examples.append(self.create_example(img[:, depth], lbl[:, depth], os.path.basename(img_path)))
            else:
                examples.append(self.create_example(img, lbl, os.path.basename(img_path)))

        fname = os.path.basename(files[0][0]).replace("_x.npy", "")
        tfrecord_name = os.path.join(self.results, f"{fname}.{suffix}")
        with tf.io.TFRecordWriter(tfrecord_name) as writer:
            for example in examples:
                writer.write(example.SerializeToString())

    def create_idx_files(self, tfrecords, save_dir):
        make_empty_dir(save_dir)
        tfrecords_idx = []
        for tfrec in tfrecords:
            fname = os.path.basename(tfrec).split(".")[0]
            tfrecords_idx.append(os.path.join(save_dir, f"{fname}.idx"))

        Parallel(n_jobs=self.args.n_jobs)(
            delayed(self.create_idx)(tr, ti) for tr, ti in tqdm(zip(tfrecords, tfrecords_idx), total=len(tfrecords))
        )

    def create_example(self, img, lbl, fname):
        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        feature = {
            "X": _float_feature(img.flatten()),
            "X_shape": _int64_feature(img.shape),
            "fname": _bytes_feature(str.encode(fname)),
        }

        if self.mode == "training":
            feature.update({"Y": _bytes_feature(lbl.flatten().tobytes()), "Y_shape": _int64_feature(lbl.shape)})

        return tf.train.Example(features=tf.train.Features(feature=feature))

    @staticmethod
    def create_idx(tfrecord, tfidx):
        call(["tfrecord2idx", tfrecord, tfidx])

    def load_files(self, suffix):
        return sorted(glob(os.path.join(self.data, suffix)))

    def get_tfrecords_data(self, tfrec_pattern, idx_dir):
        tfrecords = self.load_files(os.path.join(self.results, tfrec_pattern))
        tfrecords_dir = os.path.join(self.results, idx_dir)
        return tfrecords, tfrecords_dir
