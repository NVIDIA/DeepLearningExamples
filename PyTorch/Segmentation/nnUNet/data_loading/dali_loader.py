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
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator


def get_numpy_reader(files, shard_id, num_shards, seed, shuffle):
    return ops.readers.Numpy(
        seed=seed,
        files=files,
        device="cpu",
        read_ahead=True,
        shard_id=shard_id,
        pad_last_batch=True,
        num_shards=num_shards,
        dont_use_mmap=True,
        shuffle_after_epoch=shuffle,
    )


class TrainPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super(TrainPipeline, self).__init__(batch_size, num_threads, device_id)
        self.dim = kwargs["dim"]
        self.internal_seed = kwargs["seed"]
        self.oversampling = kwargs["oversampling"]
        self.input_x = get_numpy_reader(
            num_shards=kwargs["gpus"],
            files=kwargs["imgs"],
            seed=kwargs["seed"],
            shard_id=device_id,
            shuffle=True,
        )
        self.input_y = get_numpy_reader(
            num_shards=kwargs["gpus"],
            files=kwargs["lbls"],
            seed=kwargs["seed"],
            shard_id=device_id,
            shuffle=True,
        )
        self.patch_size = kwargs["patch_size"]
        if self.dim == 2:
            self.patch_size = [kwargs["batch_size_2d"]] + self.patch_size

        self.crop_shape = types.Constant(np.array(self.patch_size), dtype=types.INT64)
        self.crop_shape_float = types.Constant(np.array(self.patch_size), dtype=types.FLOAT)

    def load_data(self):
        img, lbl = self.input_x(name="ReaderX"), self.input_y(name="ReaderY")
        img, lbl = fn.reshape(img, layout="CDHW"), fn.reshape(lbl, layout="CDHW")
        return img, lbl

    def random_augmentation(self, probability, augmented, original):
        condition = fn.cast(fn.random.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
        neg_condition = condition ^ True
        return condition * augmented + neg_condition * original

    @staticmethod
    def slice_fn(img):
        return fn.slice(img, 1, 3, axes=[0])

    def biased_crop_fn(self, img, label):
        roi_start, roi_end = fn.segmentation.random_object_bbox(
            label,
            format="start_end",
            foreground_prob=self.oversampling,
            background=0,
            seed=self.internal_seed,
            device="cpu",
            cache_objects=True,
        )

        anchor = fn.roi_random_crop(label, roi_start=roi_start, roi_end=roi_end, crop_shape=[1, *self.patch_size])
        anchor = fn.slice(anchor, 1, 3, axes=[0])  # drop channels from anchor
        img, label = fn.slice(
            [img, label], anchor, self.crop_shape, axis_names="DHW", out_of_bounds_policy="pad", device="cpu"
        )

        return img.gpu(), label.gpu()

    def zoom_fn(self, img, lbl):
        scale = self.random_augmentation(0.15, fn.random.uniform(range=(0.7, 1.0)), 1.0)
        d, h, w = [scale * x for x in self.patch_size]
        if self.dim == 2:
            d = self.patch_size[0]
        img, lbl = fn.crop(img, crop_h=h, crop_w=w, crop_d=d), fn.crop(lbl, crop_h=h, crop_w=w, crop_d=d)
        img = fn.resize(img, interp_type=types.DALIInterpType.INTERP_CUBIC, size=self.crop_shape_float)
        lbl = fn.resize(lbl, interp_type=types.DALIInterpType.INTERP_NN, size=self.crop_shape_float)
        return img, lbl

    def noise_fn(self, img):
        img_noised = img + fn.random.normal(img, stddev=fn.random.uniform(range=(0.0, 0.33)))
        return self.random_augmentation(0.15, img_noised, img)

    def blur_fn(self, img):
        img_blurred = fn.gaussian_blur(img, sigma=fn.random.uniform(range=(0.5, 1.5)))
        return self.random_augmentation(0.15, img_blurred, img)

    def brightness_fn(self, img):
        brightness_scale = self.random_augmentation(0.15, fn.random.uniform(range=(0.7, 1.3)), 1.0)
        return img * brightness_scale

    def contrast_fn(self, img):
        min_, max_ = fn.reductions.min(img), fn.reductions.max(img)
        scale = self.random_augmentation(0.15, fn.random.uniform(range=(0.65, 1.5)), 1.0)
        img = math.clamp(img * scale, min_, max_)
        return img

    def flips_fn(self, img, lbl):
        kwargs = {
            "horizontal": fn.random.coin_flip(probability=0.33),
            "vertical": fn.random.coin_flip(probability=0.33),
        }
        if self.dim == 3:
            kwargs.update({"depthwise": fn.random.coin_flip(probability=0.33)})
        return fn.flip(img, **kwargs), fn.flip(lbl, **kwargs)

    def transpose_fn(self, img, lbl):
        img, lbl = fn.transpose(img, perm=(1, 0, 2, 3)), fn.transpose(lbl, perm=(1, 0, 2, 3))
        return img, lbl

    def define_graph(self):
        img, lbl = self.load_data()
        img, lbl = self.biased_crop_fn(img, lbl)
        img, lbl = self.zoom_fn(img, lbl)
        img, lbl = self.flips_fn(img, lbl)
        img = self.noise_fn(img)
        img = self.blur_fn(img)
        img = self.brightness_fn(img)
        img = self.contrast_fn(img)
        if self.dim == 2:
            img, lbl = self.transpose_fn(img, lbl)
        return img, lbl


class EvalPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super(EvalPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input_x = get_numpy_reader(
            files=kwargs["imgs"],
            shard_id=0,
            num_shards=1,
            seed=kwargs["seed"],
            shuffle=False,
        )
        self.input_y = get_numpy_reader(
            files=kwargs["lbls"],
            shard_id=0,
            num_shards=1,
            seed=kwargs["seed"],
            shuffle=False,
        )

    def define_graph(self):
        img, lbl = self.input_x(name="ReaderX").gpu(), self.input_y(name="ReaderY").gpu()
        img, lbl = fn.reshape(img, layout="CDHW"), fn.reshape(lbl, layout="CDHW")
        return img, lbl


class BermudaPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super(BermudaPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input_x = get_numpy_reader(
            files=kwargs["imgs"],
            shard_id=device_id,
            num_shards=kwargs["gpus"],
            seed=kwargs["seed"],
            shuffle=False,
        )
        self.input_y = get_numpy_reader(
            files=kwargs["lbls"],
            shard_id=device_id,
            num_shards=kwargs["gpus"],
            seed=kwargs["seed"],
            shuffle=False,
        )
        self.patch_size = kwargs["patch_size"]

    def crop_fn(self, img, lbl):
        img = fn.crop(img, crop=self.patch_size, out_of_bounds_policy="pad")
        lbl = fn.crop(lbl, crop=self.patch_size, out_of_bounds_policy="pad")
        return img, lbl

    def define_graph(self):
        img, lbl = self.input_x(name="ReaderX"), self.input_y(name="ReaderY")
        img, lbl = fn.reshape(img, layout="CDHW"), fn.reshape(lbl, layout="CDHW")
        img, lbl = self.crop_fn(img, lbl)
        return img, lbl


class TestPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super(TestPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input_x = get_numpy_reader(
            files=kwargs["imgs"],
            shard_id=device_id,
            num_shards=kwargs["gpus"],
            seed=kwargs["seed"],
            shuffle=False,
        )
        self.input_meta = get_numpy_reader(
            files=kwargs["meta"],
            shard_id=device_id,
            num_shards=kwargs["gpus"],
            seed=kwargs["seed"],
            shuffle=False,
        )

    def define_graph(self):
        img, meta = self.input_x(name="ReaderX").gpu(), self.input_meta(name="ReaderY").gpu()
        img = fn.reshape(img, layout="CDHW")
        return img, meta


class BenchmarkPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super(BenchmarkPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input_x = get_numpy_reader(
            files=kwargs["imgs"],
            shard_id=device_id,
            seed=kwargs["seed"],
            num_shards=kwargs["gpus"],
            shuffle=False,
        )
        self.input_y = get_numpy_reader(
            files=kwargs["lbls"],
            shard_id=device_id,
            num_shards=kwargs["gpus"],
            seed=kwargs["seed"],
            shuffle=False,
        )
        self.dim = kwargs["dim"]
        self.patch_size = kwargs["patch_size"]
        if self.dim == 2:
            self.patch_size = [kwargs["batch_size_2d"]] + self.patch_size

    def load_data(self):
        img, lbl = self.input_x(name="ReaderX").gpu(), self.input_y(name="ReaderY").gpu()
        img, lbl = fn.reshape(img, layout="CDHW"), fn.reshape(lbl, layout="CDHW")
        return img, lbl

    def transpose_fn(self, img, lbl):
        img, lbl = fn.transpose(img, perm=(1, 0, 2, 3)), fn.transpose(lbl, perm=(1, 0, 2, 3))
        return img, lbl

    def crop_fn(self, img, lbl):
        img = fn.crop(img, crop=self.patch_size, out_of_bounds_policy="pad")
        lbl = fn.crop(lbl, crop=self.patch_size, out_of_bounds_policy="pad")
        return img, lbl

    def define_graph(self):
        img, lbl = self.load_data()
        img, lbl = self.crop_fn(img, lbl)
        if self.dim == 2:
            img, lbl = self.transpose_fn(img, lbl)
        return img, lbl


class LightningWrapper(DALIGenericIterator):
    def __init__(self, pipe, **kwargs):
        super().__init__(pipe, **kwargs)

    def __next__(self):
        out = super().__next__()
        out = out[0]
        return out


def fetch_dali_loader(imgs, lbls, batch_size, mode, **kwargs):
    assert len(imgs) > 0, "Got empty list of images"
    if lbls is not None:
        assert len(imgs) == len(lbls), f"Got {len(imgs)} images but {len(lbls)} lables"

    if kwargs["benchmark"]:  # Just to make sure the number of examples is large enough for benchmark run.
        nbs = kwargs["test_batches"] if mode == "test" else kwargs["train_batches"]
        if kwargs["dim"] == 3:
            nbs *= batch_size
        imgs = list(itertools.chain(*(100 * [imgs])))[: nbs * kwargs["gpus"]]
        lbls = list(itertools.chain(*(100 * [lbls])))[: nbs * kwargs["gpus"]]

    if mode == "eval":  # To avoid padding for the multigpu evaluation.
        rank = int(os.getenv("LOCAL_RANK", "0"))
        imgs, lbls = np.array_split(imgs, kwargs["gpus"]), np.array_split(lbls, kwargs["gpus"])
        imgs, lbls = [list(x) for x in imgs], [list(x) for x in lbls]
        imgs, lbls = imgs[rank], lbls[rank]

    pipe_kwargs = {
        "imgs": imgs,
        "lbls": lbls,
        "dim": kwargs["dim"],
        "gpus": kwargs["gpus"],
        "seed": kwargs["seed"],
        "meta": kwargs["meta"],
        "patch_size": kwargs["patch_size"],
        "oversampling": kwargs["oversampling"],
    }

    if kwargs["benchmark"]:
        pipeline = BenchmarkPipeline
        output_map = ["image", "label"]
        dynamic_shape = False
        if kwargs["dim"] == 2:
            pipe_kwargs.update({"batch_size_2d": batch_size})
            batch_size = 1
    elif mode == "train":
        pipeline = TrainPipeline
        output_map = ["image", "label"]
        dynamic_shape = False
        if kwargs["dim"] == 2:
            pipe_kwargs.update({"batch_size_2d": batch_size // kwargs["nvol"]})
            batch_size = kwargs["nvol"]
    elif mode == "eval":
        pipeline = EvalPipeline
        output_map = ["image", "label"]
        dynamic_shape = True
    elif mode == "bermuda":
        pipeline = BermudaPipeline
        output_map = ["image", "label"]
        dynamic_shape = False
    else:
        pipeline = TestPipeline
        output_map = ["image", "meta"]
        dynamic_shape = True

    device_id = int(os.getenv("LOCAL_RANK", "0"))
    pipe = pipeline(batch_size, kwargs["num_workers"], device_id, **pipe_kwargs)
    return LightningWrapper(
        pipe,
        auto_reset=True,
        reader_name="ReaderX",
        output_map=output_map,
        dynamic_shape=dynamic_shape,
    )
