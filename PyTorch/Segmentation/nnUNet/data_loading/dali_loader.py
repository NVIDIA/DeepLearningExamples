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

import itertools
import os

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator


def random_augmentation(probability, augmented, original):
    condition = fn.cast(fn.random.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
    neg_condition = condition ^ True
    return condition * augmented + neg_condition * original


class GenericPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super().__init__(batch_size, num_threads, device_id)
        self.kwargs = kwargs
        self.dim = kwargs["dim"]
        self.device = device_id
        self.layout = kwargs["layout"]
        self.patch_size = kwargs["patch_size"]
        self.load_to_gpu = kwargs["load_to_gpu"]
        self.input_x = self.get_reader(kwargs["imgs"])
        self.input_y = self.get_reader(kwargs["lbls"]) if kwargs["lbls"] is not None else None
        self.cdhw2dhwc = ops.Transpose(device="gpu", perm=[1, 2, 3, 0])

    def get_reader(self, data):
        return ops.readers.Numpy(
            files=data,
            device="cpu",
            read_ahead=True,
            dont_use_mmap=True,
            pad_last_batch=True,
            shard_id=self.device,
            seed=self.kwargs["seed"],
            num_shards=self.kwargs["gpus"],
            shuffle_after_epoch=self.kwargs["shuffle"],
        )

    def load_data(self):
        img = self.input_x(name="ReaderX")
        if self.load_to_gpu:
            img = img.gpu()
        img = fn.reshape(img, layout="CDHW")
        if self.input_y is not None:
            lbl = self.input_y(name="ReaderY")
            if self.load_to_gpu:
                lbl = lbl.gpu()
            lbl = fn.reshape(lbl, layout="CDHW")
            return img, lbl
        return img

    def make_dhwc_layout(self, img, lbl):
        img, lbl = self.cdhw2dhwc(img), self.cdhw2dhwc(lbl)
        return img, lbl

    def crop(self, data):
        return fn.crop(data, crop=self.patch_size, out_of_bounds_policy="pad")

    def crop_fn(self, img, lbl):
        img, lbl = self.crop(img), self.crop(lbl)
        return img, lbl

    def transpose_fn(self, img, lbl):
        img, lbl = fn.transpose(img, perm=(1, 0, 2, 3)), fn.transpose(lbl, perm=(1, 0, 2, 3))
        return img, lbl


class TrainPipeline(GenericPipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super().__init__(batch_size, num_threads, device_id, **kwargs)
        self.oversampling = kwargs["oversampling"]
        self.crop_shape = types.Constant(np.array(self.patch_size), dtype=types.INT64)
        self.crop_shape_float = types.Constant(np.array(self.patch_size), dtype=types.FLOAT)

    @staticmethod
    def slice_fn(img):
        return fn.slice(img, 1, 3, axes=[0])

    def resize(self, data, interp_type):
        return fn.resize(data, interp_type=interp_type, size=self.crop_shape_float)

    def biased_crop_fn(self, img, label):
        roi_start, roi_end = fn.segmentation.random_object_bbox(
            label,
            device="cpu",
            background=0,
            format="start_end",
            cache_objects=True,
            foreground_prob=self.oversampling,
        )
        anchor = fn.roi_random_crop(label, roi_start=roi_start, roi_end=roi_end, crop_shape=[1, *self.patch_size])
        anchor = fn.slice(anchor, 1, 3, axes=[0])  # drop channels from anchor
        img, label = fn.slice(
            [img, label], anchor, self.crop_shape, axis_names="DHW", out_of_bounds_policy="pad", device="cpu"
        )
        return img.gpu(), label.gpu()

    def zoom_fn(self, img, lbl):
        scale = random_augmentation(0.15, fn.random.uniform(range=(0.7, 1.0)), 1.0)
        d, h, w = [scale * x for x in self.patch_size]
        if self.dim == 2:
            d = self.patch_size[0]
        img, lbl = fn.crop(img, crop_h=h, crop_w=w, crop_d=d), fn.crop(lbl, crop_h=h, crop_w=w, crop_d=d)
        img, lbl = self.resize(img, types.DALIInterpType.INTERP_CUBIC), self.resize(lbl, types.DALIInterpType.INTERP_NN)
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
        scale = random_augmentation(0.15, fn.random.uniform(range=(0.65, 1.5)), 1.0)
        return math.clamp(img * scale, fn.reductions.min(img), fn.reductions.max(img))

    def flips_fn(self, img, lbl):
        kwargs = {
            "horizontal": fn.random.coin_flip(probability=0.5),
            "vertical": fn.random.coin_flip(probability=0.5),
        }
        if self.dim == 3:
            kwargs.update({"depthwise": fn.random.coin_flip(probability=0.5)})
        return fn.flip(img, **kwargs), fn.flip(lbl, **kwargs)

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
        if self.layout == "NDHWC" and self.dim == 3:
            img, lbl = self.make_dhwc_layout(img, lbl)
        return img, lbl


class EvalPipeline(GenericPipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super().__init__(batch_size, num_threads, device_id, **kwargs)
        self.invert_resampled_y = kwargs["invert_resampled_y"]
        if self.invert_resampled_y:
            self.input_meta = self.get_reader(kwargs["meta"])
            self.input_orig_y = self.get_reader(kwargs["orig_lbl"])

    def define_graph(self):
        img, lbl = self.load_data()
        if self.invert_resampled_y:
            meta = self.input_meta(name="ReaderM")
            orig_lbl = self.input_orig_y(name="ReaderO")
            return img, lbl, meta, orig_lbl
        if self.layout == "NDHWC" and self.dim == 3:
            img, lbl = self.make_dhwc_layout(img, lbl)
        return img, lbl


class TritonPipeline(GenericPipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super().__init__(batch_size, num_threads, device_id, **kwargs)

    def define_graph(self):
        img, lbl = self.load_data()
        img, lbl = self.crop_fn(img, lbl)
        return img, lbl


class TestPipeline(GenericPipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super().__init__(batch_size, num_threads, device_id, **kwargs)
        self.input_meta = self.get_reader(kwargs["meta"])

    def define_graph(self):
        img = self.load_data()
        meta = self.input_meta(name="ReaderM")
        return img, meta


class BenchmarkPipeline(GenericPipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super().__init__(batch_size, num_threads, device_id, **kwargs)

    def define_graph(self):
        img, lbl = self.load_data()
        img, lbl = self.crop_fn(img, lbl)
        if self.dim == 2:
            img, lbl = self.transpose_fn(img, lbl)
        if self.layout == "NDHWC" and self.dim == 3:
            img, lbl = self.make_dhwc_layout(img, lbl)
        return img, lbl


PIPELINES = {
    "train": TrainPipeline,
    "eval": EvalPipeline,
    "test": TestPipeline,
    "benchmark": BenchmarkPipeline,
    "triton": TritonPipeline,
}


class LightningWrapper(DALIGenericIterator):
    def __init__(self, pipe, **kwargs):
        super().__init__(pipe, **kwargs)

    def __next__(self):
        out = super().__next__()[0]
        return out


def fetch_dali_loader(imgs, lbls, batch_size, mode, **kwargs):
    assert len(imgs) > 0, "Empty list of images!"
    if lbls is not None:
        assert len(imgs) == len(lbls), f"Number of images ({len(imgs)}) not matching number of labels ({len(lbls)})"

    if kwargs["benchmark"]:  # Just to make sure the number of examples is large enough for benchmark run.
        batches = kwargs["test_batches"] if mode == "test" else kwargs["train_batches"]
        examples = batches * batch_size * kwargs["gpus"]
        imgs = list(itertools.chain(*(100 * [imgs])))[:examples]
        lbls = list(itertools.chain(*(100 * [lbls])))[:examples]
        mode = "benchmark"

    pipeline = PIPELINES[mode]
    shuffle = True if mode == "train" else False
    dynamic_shape = True if mode in ["eval", "test"] else False
    load_to_gpu = True if mode in ["eval", "test", "benchmark"] else False
    pipe_kwargs = {"imgs": imgs, "lbls": lbls, "load_to_gpu": load_to_gpu, "shuffle": shuffle, **kwargs}
    output_map = ["image", "meta"] if mode == "test" else ["image", "label"]

    if kwargs["dim"] == 2 and mode in ["train", "benchmark"]:
        batch_size_2d = batch_size // kwargs["nvol"] if mode == "train" else batch_size
        batch_size = kwargs["nvol"] if mode == "train" else 1
        pipe_kwargs.update({"patch_size": [batch_size_2d] + kwargs["patch_size"]})

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if mode == "eval":  # We sharded the data for evaluation manually.
        rank = 0
        pipe_kwargs["gpus"] = 1

    pipe = pipeline(batch_size, kwargs["num_workers"], rank, **pipe_kwargs)
    return LightningWrapper(
        pipe,
        auto_reset=True,
        reader_name="ReaderX",
        output_map=output_map,
        dynamic_shape=dynamic_shape,
    )
