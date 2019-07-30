#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
import horovod.tensorflow as hvd

from utils import image_processing
from utils import hvd_utils

from nvidia import dali
import nvidia.dali.plugin.tf as dali_tf

__all__ = ["get_synth_input_fn", "normalized_inputs"]


class HybridPipe(dali.pipeline.Pipeline):
    def __init__(self,
                 tfrec_filenames,
                 tfrec_idx_filenames,
                 height, width,
                 batch_size,
                 num_threads,
                 device_id,
                 shard_id,
                 num_gpus,
                 deterministic=False,
                 dali_cpu=True,
                 training=True):

        kwargs = dict()
        if deterministic:
            kwargs['seed'] = 7 * (1 + hvd.rank())
        super(HybridPipe, self).__init__(batch_size, num_threads, device_id, **kwargs)

        self.input = dali.ops.TFRecordReader(
            path=tfrec_filenames,
            index_path=tfrec_idx_filenames,
            random_shuffle=True,
            shard_id=shard_id,
            num_shards=num_gpus,
            initial_fill=10000,
            features={
                'image/encoded':dali.tfrecord.FixedLenFeature((), dali.tfrecord.string, ""),
                'image/class/label':dali.tfrecord.FixedLenFeature([1], dali.tfrecord.int64,  -1),
                'image/class/text':dali.tfrecord.FixedLenFeature([ ], dali.tfrecord.string, ''),
                'image/object/bbox/xmin':dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0),
                'image/object/bbox/ymin':dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0),
                'image/object/bbox/xmax':dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0),
                'image/object/bbox/ymax':dali.tfrecord.VarLenFeature(dali.tfrecord.float32, 0.0)})
        if dali_cpu:
            self.decode = dali.ops.HostDecoder(device="cpu", output_type=dali.types.RGB)
            resize_device = "cpu"
        else:
            self.decode = dali.ops.nvJPEGDecoder(
                device="mixed",
                output_type=dali.types.RGB)
            resize_device = "gpu"

        if training:
            self.resize = dali.ops.RandomResizedCrop(
                device=resize_device,
                size=[height, width],
                interp_type=dali.types.INTERP_LINEAR,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100)
        else:
            # Make sure that every image > 224 for CropMirrorNormalize
            self.resize = dali.ops.Resize (device=resize_device, resize_shorter=256)

        self.normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=dali.types.FLOAT,
            crop=(height, width),
            image_type=dali.types.RGB,
            mean=[121., 115., 100.],
            std=[70., 68., 71.],
            output_layout=dali.types.NHWC)
        self.uniform = dali.ops.Uniform(range=(0.0, 1.0))
        self.cast_float = dali.ops.Cast(device="gpu", dtype=dali.types.FLOAT)
        self.mirror = dali.ops.CoinFlip()
        self.iter = 0

    def define_graph(self):
        # Read images and labels
        inputs = self.input(name="Reader")
        images = inputs["image/encoded"]
        labels = inputs["image/class/label"].gpu()

        # Decode and augmentation
        images = self.decode(images)
        images = self.resize(images)
        images = self.normalize(images.gpu(), mirror=self.mirror())

        return (images, labels)

class DALIPreprocessor(object):
    def __init__(self,
                 filenames,
                 idx_filenames,
                 height, width,
                 batch_size,
                 num_threads,
                 dtype=tf.uint8,
                 dali_cpu=True,
                 deterministic=False,
                 training=False):
        device_id = hvd.local_rank()
        shard_id = hvd.rank()
        num_gpus = hvd.size()
        pipe = HybridPipe(
            tfrec_filenames=filenames,
            tfrec_idx_filenames=idx_filenames,
            height=height,
            width=width,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            shard_id=shard_id,
            num_gpus=num_gpus,
            deterministic=deterministic,
            dali_cpu=dali_cpu,
            training=training)
        
        daliop = dali_tf.DALIIterator()

        with tf.device("/gpu:0"):
            self.images, self.labels = daliop(
                pipeline=pipe,
                shapes=[(batch_size, height, width, 3), (batch_size, 1)],
                dtypes=[tf.float32, tf.int64],
                device_id=device_id)

    def get_device_minibatches(self):
        with tf.device("/gpu:0"):
            self.labels -= 1 # Change to 0-based (don't use background class)
            self.labels = tf.squeeze(self.labels)
        return self.images, self.labels