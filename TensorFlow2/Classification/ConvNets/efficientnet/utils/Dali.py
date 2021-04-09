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

import sys

import tensorflow as tf
import horovod.tensorflow.keras as hvd

from nvidia import dali
import nvidia.dali.plugin.tf as dali_tf
import numpy as np

class DaliPipeline(dali.pipeline.Pipeline):

    def __init__(
        self,
        tfrec_filenames,
        tfrec_idx_filenames,
        height,
        width,
        batch_size,
        num_threads,
        device_id,
        shard_id,
        num_gpus,
        num_classes,
        deterministic=False,
        dali_cpu=True,
        training=True
    ):

        kwargs = dict()
        if deterministic:
            kwargs['seed'] = 7 * (1 + hvd.rank())
        super(DaliPipeline, self).__init__(batch_size, num_threads, device_id, **kwargs)

        self.training = training
        self.input = dali.ops.TFRecordReader(
            path=tfrec_filenames,
            index_path=tfrec_idx_filenames,
            random_shuffle=True,
            shard_id=shard_id,
            num_shards=num_gpus,
            initial_fill=10000,
            features={
                'image/encoded': dali.tfrecord.FixedLenFeature((), dali.tfrecord.string, ""),
                'image/class/label': dali.tfrecord.FixedLenFeature([1], dali.tfrecord.int64, -1),
                'image/class/text': dali.tfrecord.FixedLenFeature([], dali.tfrecord.string, ''),
                'image/object/bbox/xmin': dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0),
                'image/object/bbox/ymin': dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0),
                'image/object/bbox/xmax': dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0),
                'image/object/bbox/ymax': dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0)
            }
        )

        if self.training:
            self.decode = dali.ops.ImageDecoderRandomCrop(
                device="cpu" if dali_cpu else "mixed",
                output_type=dali.types.RGB,
                random_aspect_ratio=[0.75, 1.33],
                random_area=[0.05, 1.0],
                num_attempts=100
            )
            self.resize = dali.ops.Resize(device="cpu" if dali_cpu else "gpu", resize_x=width, resize_y=height)
        else:
            self.decode = dali.ops.ImageDecoder(
                device="cpu",
                output_type=dali.types.RGB
            )
            # Make sure that every image > 224 for CropMirrorNormalize
            self.resize = dali.ops.Resize(device="cpu" if dali_cpu else "gpu", resize_x=width, resize_y=height)

        self.normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=dali.types.FLOAT,
            image_type=dali.types.RGB,
            output_layout=dali.types.NHWC,
            mirror=1 if self.training else 0
        )
        self.one_hot = dali.ops.OneHot(num_classes=num_classes)
        self.shapes = dali.ops.Shapes(type=dali.types.INT32)
        self.crop = dali.ops.Crop(device="gpu")
        self.cast_float = dali.ops.Cast(dtype=dali.types.FLOAT)
        self.extract_h = dali.ops.Slice(normalized_anchor=False, normalized_shape=False, axes=[0])
        self.extract_w = dali.ops.Slice(normalized_anchor=False, normalized_shape=False, axes=[0])

    def define_graph(self):
        # Read images and labels
        inputs = self.input(name="Reader")
        images = inputs["image/encoded"]
        labels = inputs["image/class/label"]
        labels -= 1
        labels = self.one_hot(labels).gpu()

        # Decode and augmentation
        images = self.decode(images)
        if not self.training:
            shapes = self.shapes(images)
            h = self.extract_h(shapes, dali.types.Constant(np.array([0], dtype=np.float32)), dali.types.Constant(np.array([1], dtype=np.float32)))
            w = self.extract_w(shapes, dali.types.Constant(np.array([1], dtype=np.float32)), dali.types.Constant(np.array([1], dtype=np.float32)))
            CROP_PADDING = 32
            CROP_H = h * h / (h + CROP_PADDING)
            CROP_W = w * w / (w + CROP_PADDING)
            CROP_H = self.cast_float(CROP_H)
            CROP_W = self.cast_float(CROP_W)
            images = images.gpu()
            images = self.crop(images, crop_h = CROP_H, crop_w = CROP_W)
        images = self.resize(images)
        images = self.normalize(images)


        return (images, labels)
