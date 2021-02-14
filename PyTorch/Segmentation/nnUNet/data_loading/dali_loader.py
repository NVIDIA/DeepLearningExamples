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

import itertools
import os

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.ops as ops
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class TFRecordTrain(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super(TFRecordTrain, self).__init__(batch_size, num_threads, device_id)
        self.dim = kwargs["dim"]
        self.seed = kwargs["seed"]
        self.oversampling = kwargs["oversampling"]
        self.input = ops.TFRecordReader(
            path=kwargs["tfrecords"],
            index_path=kwargs["tfrecords_idx"],
            features={
                "X_shape": tfrec.FixedLenFeature([self.dim + 1], tfrec.int64, 0),
                "Y_shape": tfrec.FixedLenFeature([self.dim + 1], tfrec.int64, 0),
                "X": tfrec.VarLenFeature([], tfrec.float32, 0.0),
                "Y": tfrec.FixedLenFeature([], tfrec.string, ""),
                "fname": tfrec.FixedLenFeature([], tfrec.string, ""),
            },
            num_shards=kwargs["gpus"],
            shard_id=device_id,
            random_shuffle=True,
            pad_last_batch=True,
            read_ahead=True,
            seed=self.seed,
        )
        self.patch_size = kwargs["patch_size"]
        self.crop_shape = types.Constant(np.array(self.patch_size), dtype=types.INT64)
        self.crop_shape_float = types.Constant(np.array(self.patch_size), dtype=types.FLOAT)
        self.layout = "CDHW" if self.dim == 3 else "CHW"
        self.axis_name = "DHW" if self.dim == 3 else "HW"

    def load_data(self, features):
        img = fn.reshape(features["X"], shape=features["X_shape"], layout=self.layout)
        lbl = fn.reshape(features["Y"], shape=features["Y_shape"], layout=self.layout)
        lbl = fn.reinterpret(lbl, dtype=types.DALIDataType.UINT8)
        return img, lbl

    def random_augmentation(self, probability, augmented, original):
        condition = fn.cast(fn.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
        neg_condition = condition ^ True
        return condition * augmented + neg_condition * original

    @staticmethod
    def slice_fn(img, start_idx, length):
        return fn.slice(img, start_idx, length, axes=[0])

    def crop_fn(self, img, lbl):
        center = fn.segmentation.random_mask_pixel(lbl, foreground=fn.coin_flip(probability=self.oversampling))
        crop_anchor = self.slice_fn(center, 1, self.dim) - self.crop_shape // 2
        adjusted_anchor = math.max(0, crop_anchor)
        max_anchor = self.slice_fn(fn.shapes(lbl), 1, self.dim) - self.crop_shape
        crop_anchor = math.min(adjusted_anchor, max_anchor)
        img = fn.slice(img.gpu(), crop_anchor, self.crop_shape, axis_names=self.axis_name, out_of_bounds_policy="pad")
        lbl = fn.slice(lbl.gpu(), crop_anchor, self.crop_shape, axis_names=self.axis_name, out_of_bounds_policy="pad")
        return img, lbl

    def zoom_fn(self, img, lbl):
        resized_shape = self.crop_shape * self.random_augmentation(0.15, fn.uniform(range=(0.7, 1.0)), 1.0)
        img, lbl = fn.crop(img, crop=resized_shape), fn.crop(lbl, crop=resized_shape)
        img = fn.resize(img, interp_type=types.DALIInterpType.INTERP_CUBIC, size=self.crop_shape_float)
        lbl = fn.resize(lbl, interp_type=types.DALIInterpType.INTERP_NN, size=self.crop_shape_float)
        return img, lbl

    def noise_fn(self, img):
        img_noised = img + fn.normal_distribution(img, stddev=fn.uniform(range=(0.0, 0.33)))
        return self.random_augmentation(0.15, img_noised, img)

    def blur_fn(self, img):
        img_blured = fn.gaussian_blur(img, sigma=fn.uniform(range=(0.5, 1.5)))
        return self.random_augmentation(0.15, img_blured, img)

    def brightness_fn(self, img):
        brightness_scale = self.random_augmentation(0.15, fn.uniform(range=(0.7, 1.3)), 1.0)
        return img * brightness_scale

    def contrast_fn(self, img):
        min_, max_ = fn.reductions.min(img), fn.reductions.max(img)
        scale = self.random_augmentation(0.15, fn.uniform(range=(0.65, 1.5)), 1.0)
        img = math.clamp(img * scale, min_, max_)
        return img

    def flips_fn(self, img, lbl):
        kwargs = {"horizontal": fn.coin_flip(probability=0.33), "vertical": fn.coin_flip(probability=0.33)}
        if self.dim == 3:
            kwargs.update({"depthwise": fn.coin_flip(probability=0.33)})
        return fn.flip(img, **kwargs), fn.flip(lbl, **kwargs)

    def define_graph(self):
        features = self.input(name="Reader")
        img, lbl = self.load_data(features)
        img, lbl = self.crop_fn(img, lbl)
        img, lbl = self.zoom_fn(img, lbl)
        img = self.noise_fn(img)
        img = self.blur_fn(img)
        img = self.brightness_fn(img)
        img = self.contrast_fn(img)
        img, lbl = self.flips_fn(img, lbl)
        return img, lbl


class TFRecordEval(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super(TFRecordEval, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.TFRecordReader(
            path=kwargs["tfrecords"],
            index_path=kwargs["tfrecords_idx"],
            features={
                "X_shape": tfrec.FixedLenFeature([4], tfrec.int64, 0),
                "Y_shape": tfrec.FixedLenFeature([4], tfrec.int64, 0),
                "X": tfrec.VarLenFeature([], tfrec.float32, 0.0),
                "Y": tfrec.FixedLenFeature([], tfrec.string, ""),
                "fname": tfrec.FixedLenFeature([], tfrec.string, ""),
            },
            shard_id=device_id,
            num_shards=kwargs["gpus"],
            read_ahead=True,
            random_shuffle=False,
            pad_last_batch=True,
        )

    def load_data(self, features):
        img = fn.reshape(features["X"].gpu(), shape=features["X_shape"], layout="CDHW")
        lbl = fn.reshape(features["Y"].gpu(), shape=features["Y_shape"], layout="CDHW")
        lbl = fn.reinterpret(lbl, dtype=types.DALIDataType.UINT8)
        return img, lbl

    def define_graph(self):
        features = self.input(name="Reader")
        img, lbl = self.load_data(features)
        return img, lbl, features["fname"]


class TFRecordTest(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super(TFRecordTest, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.TFRecordReader(
            path=kwargs["tfrecords"],
            index_path=kwargs["tfrecords_idx"],
            features={
                "X_shape": tfrec.FixedLenFeature([4], tfrec.int64, 0),
                "X": tfrec.VarLenFeature([], tfrec.float32, 0.0),
                "fname": tfrec.FixedLenFeature([], tfrec.string, ""),
            },
            shard_id=device_id,
            num_shards=kwargs["gpus"],
            read_ahead=True,
            random_shuffle=False,
            pad_last_batch=True,
        )

    def define_graph(self):
        features = self.input(name="Reader")
        img = fn.reshape(features["X"].gpu(), shape=features["X_shape"], layout="CDHW")
        return img, features["fname"]


class TFRecordBenchmark(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super(TFRecordBenchmark, self).__init__(batch_size, num_threads, device_id)
        self.dim = kwargs["dim"]
        self.input = ops.TFRecordReader(
            path=kwargs["tfrecords"],
            index_path=kwargs["tfrecords_idx"],
            features={
                "X_shape": tfrec.FixedLenFeature([self.dim + 1], tfrec.int64, 0),
                "Y_shape": tfrec.FixedLenFeature([self.dim + 1], tfrec.int64, 0),
                "X": tfrec.VarLenFeature([], tfrec.float32, 0.0),
                "Y": tfrec.FixedLenFeature([], tfrec.string, ""),
                "fname": tfrec.FixedLenFeature([], tfrec.string, ""),
            },
            shard_id=device_id,
            num_shards=kwargs["gpus"],
            read_ahead=True,
        )
        self.patch_size = kwargs["patch_size"]
        self.layout = "CDHW" if self.dim == 3 else "CHW"

    def load_data(self, features):
        img = fn.reshape(features["X"].gpu(), shape=features["X_shape"], layout=self.layout)
        lbl = fn.reshape(features["Y"].gpu(), shape=features["Y_shape"], layout=self.layout)
        lbl = fn.reinterpret(lbl, dtype=types.DALIDataType.UINT8)
        return img, lbl

    def crop_fn(self, img, lbl):
        img = fn.crop(img, crop=self.patch_size)
        lbl = fn.crop(lbl, crop=self.patch_size)
        return img, lbl

    def define_graph(self):
        features = self.input(name="Reader")
        img, lbl = self.load_data(features)
        img, lbl = self.crop_fn(img, lbl)
        return img, lbl


class LightningWrapper(DALIGenericIterator):
    def __init__(self, pipe, **kwargs):
        super().__init__(pipe, **kwargs)

    def __next__(self):
        out = super().__next__()
        out = out[0]
        return out


def fetch_dali_loader(tfrecords, idx_files, batch_size, mode, **kwargs):
    assert len(tfrecords) > 0, "Got empty tfrecord list"
    assert len(idx_files) == len(tfrecords), f"Got {len(idx_files)} index files but {len(tfrecords)} tfrecords"

    if kwargs["benchmark"]:
        tfrecords = list(itertools.chain(*(20 * [tfrecords])))
        idx_files = list(itertools.chain(*(20 * [idx_files])))

    pipe_kwargs = {
        "tfrecords": tfrecords,
        "tfrecords_idx": idx_files,
        "gpus": kwargs["gpus"],
        "seed": kwargs["seed"],
        "patch_size": kwargs["patch_size"],
        "dim": kwargs["dim"],
        "oversampling": kwargs["oversampling"],
    }

    if kwargs["benchmark"] and mode == "eval":
        pipeline = TFRecordBenchmark
        output_map = ["image", "label"]
        dynamic_shape = False
    elif mode == "training":
        pipeline = TFRecordTrain
        output_map = ["image", "label"]
        dynamic_shape = False
    elif mode == "eval":
        pipeline = TFRecordEval
        output_map = ["image", "label", "fname"]
        dynamic_shape = True
    else:
        pipeline = TFRecordTest
        output_map = ["image", "fname"]
        dynamic_shape = True

    device_id = int(os.getenv("LOCAL_RANK", "0"))
    pipe = pipeline(batch_size, kwargs["num_workers"], device_id, **pipe_kwargs)
    return LightningWrapper(
        pipe,
        auto_reset=True,
        reader_name="Reader",
        output_map=output_map,
        dynamic_shape=dynamic_shape,
    )
