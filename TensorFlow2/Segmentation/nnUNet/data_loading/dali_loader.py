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

import itertools

import horovod.tensorflow as hvd
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.ops as ops
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.types as types
import tensorflow as tf
from nvidia.dali.pipeline import Pipeline


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


def random_augmentation(probability, augmented, original):
    condition = fn.cast(fn.random.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
    neg_condition = condition ^ True
    return condition * augmented + neg_condition * original


class GenericPipeline(Pipeline):
    def __init__(
        self,
        batch_size,
        num_threads,
        shard_id,
        seed,
        num_gpus,
        dim,
        shuffle_input=True,
        input_x_files=None,
        input_y_files=None,
        use_cpu=False,
    ):
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=hvd.rank(),
            seed=seed,
        )

        if input_x_files is not None:
            self.input_x = get_numpy_reader(
                files=input_x_files,
                shard_id=shard_id,
                seed=seed,
                num_shards=num_gpus,
                shuffle=shuffle_input,
            )
        if input_y_files is not None:
            self.input_y = get_numpy_reader(
                files=input_y_files,
                shard_id=shard_id,
                seed=seed,
                num_shards=num_gpus,
                shuffle=shuffle_input,
            )

        self.dim = dim
        self.internal_seed = seed
        self.use_cpu = use_cpu

    def mark_pipeline_start(self, x, y):
        if not self.use_cpu:
            x, y = x.gpu(), y.gpu()
        return x, y


class TrainPipeline(GenericPipeline):
    def __init__(self, imgs, lbls, oversampling, patch_size, read_roi=False, batch_size_2d=None, **kwargs):
        super().__init__(input_x_files=imgs, input_y_files=lbls, shuffle_input=True, **kwargs)
        self.oversampling = oversampling
        self.read_roi = read_roi
        self.patch_size = patch_size
        if self.dim == 2 and batch_size_2d is not None:
            self.patch_size = [batch_size_2d] + self.patch_size
        self.crop_shape = types.Constant(np.array(self.patch_size), dtype=types.INT64)
        self.crop_shape_float = types.Constant(np.array(self.patch_size), dtype=types.FLOAT)

    def load_data(self):
        img, lbl = self.input_x(name="ReaderX"), self.input_y(name="ReaderY")
        img, lbl = fn.reshape(img, layout="DHWC"), fn.reshape(lbl, layout="DHWC")
        return img, lbl

    @staticmethod
    def slice_fn(img):
        return fn.slice(img, 1, 3, axes=[0])

    def biased_crop_fn(self, img, lbl):
        roi_start, roi_end = fn.segmentation.random_object_bbox(
            lbl,
            format="start_end",
            foreground_prob=self.oversampling,
            k_largest=2,
            device="cpu",
            cache_objects=True,
        )

        anchor = fn.roi_random_crop(
            lbl,
            roi_start=roi_start,
            roi_end=roi_end,
            crop_shape=[*self.patch_size, 1],
        )
        anchor = fn.slice(anchor, 0, 3, axes=[0])  # drop channel from anchor
        img, lbl = fn.slice(
            [img, lbl],
            anchor,
            self.crop_shape,
            axis_names="DHW",
            out_of_bounds_policy="pad",
            device="cpu",
        )

        return img.gpu(), lbl.gpu()

    def load_roi(self):
        lbl = self.input_y(name="ReaderY")
        lbl = fn.reshape(lbl, layout="DHWC")
        roi_start, roi_end = fn.segmentation.random_object_bbox(
            lbl,
            format="start_end",
            foreground_prob=self.oversampling,
            k_largest=2,
            device="cpu",
            cache_objects=True,
        )
        anchor = fn.roi_random_crop(lbl, roi_start=roi_start, roi_end=roi_end, crop_shape=[1, *self.patch_size])
        anchor = fn.slice(anchor, 1, 3, axes=[0])  # drop channel from anchor
        lbl = fn.slice(
            lbl,
            anchor,
            self.crop_shape,
            axis_names="DHW",
            out_of_bounds_policy="pad",
            device="cpu",
        )

        img = self.input_x(
            name="ReaderX",
            roi_start=fn.cast(anchor, dtype=types.INT32),
            roi_axes=[1, 2, 3],
            roi_shape=self.patch_size,
            out_of_bounds_policy="pad",
        )
        img = fn.reshape(img, layout="DHWC")

        return img, lbl

    def zoom_fn(self, img, lbl):
        scale = random_augmentation(0.15, fn.random.uniform(range=(0.7, 1.0)), 1.0)
        d, h, w = [scale * x for x in self.patch_size]
        if self.dim == 2:
            d = self.patch_size[0]
        img, lbl = fn.crop(img, crop_h=h, crop_w=w, crop_d=d), fn.crop(lbl, crop_h=h, crop_w=w, crop_d=d)
        img = fn.resize(
            img,
            interp_type=types.DALIInterpType.INTERP_CUBIC,
            size=self.crop_shape_float,
        )
        lbl = fn.resize(lbl, interp_type=types.DALIInterpType.INTERP_NN, size=self.crop_shape_float)
        return img, lbl

    def noise_fn(self, img):
        img_noised = img + fn.random.normal(img, stddev=fn.random.uniform(range=(0.0, 0.33)))
        return random_augmentation(0.15, img_noised, img)

    def blur_fn(self, img):
        img_blurred = fn.gaussian_blur(img, sigma=fn.random.uniform(range=(0.5, 1.5)))
        return random_augmentation(0.15, img_blurred, img)

    def brightness_fn(self, img):
        brightness_scale = random_augmentation(0.15, fn.random.uniform(range=(0.7, 1.3)), 1.0)
        return img * brightness_scale

    def contrast_fn(self, img):
        min_, max_ = fn.reductions.min(img), fn.reductions.max(img)
        scale = random_augmentation(0.15, fn.random.uniform(range=(0.65, 1.5)), 1.0)
        img = math.clamp(img * scale, min_, max_)
        return img

    def flips_fn(self, img, lbl):
        kwargs = {
            "horizontal": fn.random.coin_flip(probability=0.5),
            "vertical": fn.random.coin_flip(probability=0.5),
        }
        if self.dim == 3:
            kwargs.update({"depthwise": fn.random.coin_flip(probability=0.5)})
        return fn.flip(img, **kwargs), fn.flip(lbl, **kwargs)

    def define_graph(self):
        if self.read_roi:
            img, lbl = self.load_roi()
        else:
            img, lbl = self.load_data()
            img, lbl = self.biased_crop_fn(img, lbl)
        img, lbl = img.gpu(), lbl.gpu()
        img, lbl = self.zoom_fn(img, lbl)
        img, lbl = self.flips_fn(img, lbl)
        img = self.brightness_fn(img)
        img = self.contrast_fn(img)
        return img, lbl


class EvalPipeline(GenericPipeline):
    def __init__(self, imgs, lbls, patch_size, **kwargs):
        super().__init__(input_x_files=imgs, input_y_files=lbls, shuffle_input=False, **kwargs)
        self.patch_size = patch_size

    def define_graph(self):
        img, lbl = self.input_x(name="ReaderX").gpu(), self.input_y(name="ReaderY").gpu()
        img, lbl = fn.reshape(img, layout="DHWC"), fn.reshape(lbl, layout="DHWC")
        return img, lbl


class TestPipeline(GenericPipeline):
    def __init__(self, imgs, meta, **kwargs):
        super().__init__(input_x_files=imgs, input_y_files=meta, shuffle_input=False, **kwargs)

    def define_graph(self):
        img, meta = self.input_x(name="ReaderX").gpu(), self.input_y(name="ReaderY").gpu()
        img = fn.reshape(img, layout="DHWC")
        return img, meta


class BenchmarkPipeline(GenericPipeline):
    def __init__(self, imgs, lbls, patch_size, batch_size_2d=None, sw_benchmark=False, **kwargs):
        super().__init__(input_x_files=imgs, input_y_files=lbls, shuffle_input=False, **kwargs)
        self.patch_size = patch_size
        if self.dim == 2 and batch_size_2d is not None:
            self.patch_size = [batch_size_2d] + self.patch_size
        self.crop = not sw_benchmark

    def crop_fn(self, img, lbl):
        img = fn.crop(img, crop=self.patch_size, out_of_bounds_policy="pad")
        lbl = fn.crop(lbl, crop=self.patch_size, out_of_bounds_policy="pad")
        return img, lbl

    def define_graph(self):
        img, lbl = self.input_x(name="ReaderX").gpu(), self.input_y(name="ReaderY").gpu()
        img, lbl = fn.reshape(img, layout="DHWC"), fn.reshape(lbl, layout="DHWC")
        if self.crop:
            img, lbl = self.crop_fn(img, lbl)
        return img, lbl


def fetch_dali_loader(imgs, lbls, batch_size, mode, **kwargs):
    assert len(imgs) > 0, "No images found"
    if lbls is not None:
        assert len(imgs) == len(lbls), f"Got {len(imgs)} images but {len(lbls)} lables"

    gpus = hvd.size()
    device_id = hvd.rank()
    if kwargs["benchmark"]:
        # Just to make sure the number of examples is large enough for benchmark run.
        nbs = kwargs["bench_steps"]
        if kwargs["dim"] == 3:
            nbs *= batch_size
        imgs = list(itertools.chain(*(100 * [imgs])))[: nbs * gpus]
        lbls = list(itertools.chain(*(100 * [lbls])))[: nbs * gpus]

    pipe_kwargs = {
        "dim": kwargs["dim"],
        "num_gpus": gpus,
        "seed": kwargs["seed"],
        "batch_size": batch_size,
        "num_threads": kwargs["num_workers"],
        "shard_id": device_id,
        "use_cpu": kwargs["use_cpu"],
    }
    if kwargs["dim"] == 2:
        if kwargs["benchmark"]:
            pipe_kwargs.update({"batch_size_2d": batch_size})
            batch_size = 1
        elif mode == "train":
            pipe_kwargs.update({"batch_size_2d": batch_size // kwargs["nvol"]})
            batch_size = kwargs["nvol"]
    if mode == "eval":  # Validation data is manually sharded beforehand.
        pipe_kwargs["shard_id"] = 0
        pipe_kwargs["num_gpus"] = 1

    output_dtypes = (tf.float32, tf.uint8)
    if kwargs["benchmark"]:
        pipeline = BenchmarkPipeline(
            imgs, lbls, kwargs["patch_size"], sw_benchmark=kwargs["sw_benchmark"], **pipe_kwargs
        )
    elif mode == "train":
        pipeline = TrainPipeline(
            imgs, lbls, kwargs["oversampling"], kwargs["patch_size"], kwargs["read_roi"], **pipe_kwargs
        )
    elif mode == "eval":
        pipeline = EvalPipeline(imgs, lbls, kwargs["patch_size"], **pipe_kwargs)
    else:
        pipeline = TestPipeline(imgs, kwargs["meta"], **pipe_kwargs)
        output_dtypes = (tf.float32, tf.int64)

    tf_pipe = dali_tf.DALIDataset(pipeline, batch_size=batch_size, device_id=device_id, output_dtypes=output_dtypes)
    return tf_pipe
