# Copyright 2017-2018 The Apache Software Foundation
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -----------------------------------------------------------------------
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import mxnet as mx
import mxnet.ndarray as nd
import random
import argparse
from mxnet.io import DataBatch, DataIter
import numpy as np
import horovod.mxnet as hvd

import dali

def add_data_args(parser):
    def float_list(x):
        return list(map(float, x.split(',')))
    def int_list(x):
        return list(map(int, x.split(',')))

    data = parser.add_argument_group('Data')
    data.add_argument('--data-train', type=str, help='the training data')
    data.add_argument('--data-train-idx', type=str, default='', help='the index of training data')
    data.add_argument('--data-val', type=str, help='the validation data')
    data.add_argument('--data-val-idx', type=str, default='', help='the index of validation data')
    data.add_argument('--data-pred', type=str, help='the image on which run inference (only for pred mode)')

    data.add_argument('--data-backend', choices=('dali-gpu', 'dali-cpu', 'mxnet', 'synthetic'), default='dali-gpu',
                      help='set data loading & augmentation backend')
    data.add_argument('--image-shape', type=int_list, default=[3, 224, 224],
                      help='the image shape feed into the network')
    data.add_argument('--rgb-mean', type=float_list, default=[123.68, 116.779, 103.939],
                      help='a tuple of size 3 for the mean rgb')
    data.add_argument('--rgb-std', type=float_list, default=[58.393, 57.12, 57.375],
                      help='a tuple of size 3 for the std rgb')

    data.add_argument('--input-layout', type=str, default='NCHW', choices=('NCHW', 'NHWC'),
                      help='the layout of the input data')
    data.add_argument('--conv-layout', type=str, default='NCHW', choices=('NCHW', 'NHWC'),
                      help='the layout of the data assumed by the conv operation')
    data.add_argument('--batchnorm-layout', type=str, default='NCHW', choices=('NCHW', 'NHWC'),
                      help='the layout of the data assumed by the batchnorm operation')
    data.add_argument('--pooling-layout', type=str, default='NCHW', choices=('NCHW', 'NHWC'),
                      help='the layout of the data assumed by the pooling operation')

    data.add_argument('--num-examples', type=int, default=1281167,
                      help="the number of training examples (doesn't work with mxnet data backend)")
    data.add_argument('--data-val-resize', type=int, default=256,
                      help='base length of shorter edge for validation dataset')

    return data

def add_data_aug_args(parser):
    aug = parser.add_argument_group(
            'MXNet data backend', 'entire group applies only to mxnet data backend')
    aug.add_argument('--data-mxnet-threads', type=int, default=40,
                     help='number of threads for data decoding for mxnet data backend')
    aug.add_argument('--random-crop', type=int, default=0,
                     help='if or not randomly crop the image')
    aug.add_argument('--random-mirror', type=int, default=1,
                     help='if or not randomly flip horizontally')
    aug.add_argument('--max-random-h', type=int, default=0,
                     help='max change of hue, whose range is [0, 180]')
    aug.add_argument('--max-random-s', type=int, default=0,
                     help='max change of saturation, whose range is [0, 255]')
    aug.add_argument('--max-random-l', type=int, default=0,
                     help='max change of intensity, whose range is [0, 255]')
    aug.add_argument('--min-random-aspect-ratio', type=float, default=0.75,
                     help='min value of aspect ratio, whose value is either None or a positive value.')
    aug.add_argument('--max-random-aspect-ratio', type=float, default=1.33,
                     help='max value of aspect ratio. If min_random_aspect_ratio is None, '
                          'the aspect ratio range is [1-max_random_aspect_ratio, '
                          '1+max_random_aspect_ratio], otherwise it is '
                          '[min_random_aspect_ratio, max_random_aspect_ratio].')
    aug.add_argument('--max-random-rotate-angle', type=int, default=0,
                     help='max angle to rotate, whose range is [0, 360]')
    aug.add_argument('--max-random-shear-ratio', type=float, default=0,
                     help='max ratio to shear, whose range is [0, 1]')
    aug.add_argument('--max-random-scale', type=float, default=1,
                     help='max ratio to scale')
    aug.add_argument('--min-random-scale', type=float, default=1,
                     help='min ratio to scale, should >= img_size/input_shape. '
                          'otherwise use --pad-size')
    aug.add_argument('--max-random-area', type=float, default=1,
                     help='max area to crop in random resized crop, whose range is [0, 1]')
    aug.add_argument('--min-random-area', type=float, default=0.05,
                     help='min area to crop in random resized crop, whose range is [0, 1]')
    aug.add_argument('--min-crop-size', type=int, default=-1,
                     help='Crop both width and height into a random size in '
                          '[min_crop_size, max_crop_size]')
    aug.add_argument('--max-crop-size', type=int, default=-1,
                     help='Crop both width and height into a random size in '
                          '[min_crop_size, max_crop_size]')
    aug.add_argument('--brightness', type=float, default=0,
                     help='brightness jittering, whose range is [0, 1]')
    aug.add_argument('--contrast', type=float, default=0,
                     help='contrast jittering, whose range is [0, 1]')
    aug.add_argument('--saturation', type=float, default=0,
                     help='saturation jittering, whose range is [0, 1]')
    aug.add_argument('--pca-noise', type=float, default=0,
                     help='pca noise, whose range is [0, 1]')
    aug.add_argument('--random-resized-crop', type=int, default=1,
                     help='whether to use random resized crop')
    return aug

def get_data_loader(args):
    if args.data_backend == 'dali-gpu':
        return (lambda *args, **kwargs: dali.get_rec_iter(*args, **kwargs, dali_cpu=False))
    if args.data_backend == 'dali-cpu':
        return (lambda *args, **kwargs: dali.get_rec_iter(*args, **kwargs, dali_cpu=True))
    if args.data_backend == 'synthetic':
        return get_synthetic_rec_iter
    if args.data_backend == 'mxnet':
        return get_rec_iter
    raise ValueError('Wrong data backend')

class DataGPUSplit:
    def __init__(self, dataloader, ctx, dtype):
        self.dataloader = dataloader
        self.ctx = ctx
        self.dtype = dtype
        self.batch_size = dataloader.batch_size // len(ctx)
        self._num_gpus = len(ctx)

    def __iter__(self):
        return DataGPUSplit(iter(self.dataloader), self.ctx, self.dtype)

    def __next__(self):
        data = next(self.dataloader)
        ret = []
        for i in range(len(self.ctx)):
            start = i * len(data.data[0]) // len(self.ctx)
            end = (i + 1) * len(data.data[0]) // len(self.ctx)
            pad = max(0, min(data.pad - (len(self.ctx) - i - 1) * self.batch_size, self.batch_size))
            ret.append(mx.io.DataBatch(
                [data.data[0][start:end].as_in_context(self.ctx[i]).astype(self.dtype)],
                [data.label[0][start:end].as_in_context(self.ctx[i])],
                pad=pad))
        return ret

    def next(self):
        return next(self)

    def reset(self):
        self.dataloader.reset()

def get_rec_iter(args, kv=None):
    gpus = args.gpus
    if 'horovod' in args.kv_store:
        rank = hvd.rank()
        nworker = hvd.size()
        gpus = [gpus[0]]
        batch_size = args.batch_size // hvd.size()
    else:
        rank = kv.rank if kv else 0
        nworker = kv.num_workers if kv else 1
        batch_size = args.batch_size

    if args.input_layout == 'NHWC':
        raise ValueError('ImageRecordIter cannot handle layout {}'.format(args.input_layout))


    train = DataGPUSplit(mx.io.ImageRecordIter(
            path_imgrec         = args.data_train,
            path_imgidx         = args.data_train_idx,
            label_width         = 1,
            mean_r              = args.rgb_mean[0],
            mean_g              = args.rgb_mean[1],
            mean_b              = args.rgb_mean[2],
            std_r               = args.rgb_std[0],
            std_g               = args.rgb_std[1],
            std_b               = args.rgb_std[2],
            data_name           = 'data',
            label_name          = 'softmax_label',
            data_shape          = args.image_shape,
            batch_size          = batch_size,
            rand_crop           = args.random_crop,
            max_random_scale    = args.max_random_scale,
            random_resized_crop = args.random_resized_crop,
            min_random_scale    = args.min_random_scale,
            max_aspect_ratio    = args.max_random_aspect_ratio,
            min_aspect_ratio    = args.min_random_aspect_ratio,
            max_random_area     = args.max_random_area,
            min_random_area     = args.min_random_area,
            min_crop_size       = args.min_crop_size,
            max_crop_size       = args.max_crop_size,
            brightness          = args.brightness,
            contrast            = args.contrast,
            saturation          = args.saturation,
            pca_noise           = args.pca_noise,
            random_h            = args.max_random_h,
            random_s            = args.max_random_s,
            random_l            = args.max_random_l,
            max_rotate_angle    = args.max_random_rotate_angle,
            max_shear_ratio     = args.max_random_shear_ratio,
            rand_mirror         = args.random_mirror,
            preprocess_threads  = args.data_mxnet_threads,
            shuffle             = True,
            num_parts           = nworker,
            part_index          = rank,
            seed                = args.seed or '0',
        ), [mx.gpu(gpu) for gpu in gpus], args.dtype)
    if args.data_val is None:
        return (train, None)
    val = DataGPUSplit(mx.io.ImageRecordIter(
            path_imgrec         = args.data_val,
            path_imgidx         = args.data_val_idx,
            label_width         = 1,
            mean_r              = args.rgb_mean[0],
            mean_g              = args.rgb_mean[1],
            mean_b              = args.rgb_mean[2],
            std_r               = args.rgb_std[0],
            std_g               = args.rgb_std[1],
            std_b               = args.rgb_std[2],
            data_name           = 'data',
            label_name          = 'softmax_label',
            batch_size          = batch_size,
            round_batch         = False,
            data_shape          = args.image_shape,
            preprocess_threads  = args.data_mxnet_threads,
            rand_crop           = False,
            rand_mirror         = False,
            num_parts           = nworker,
            part_index          = rank,
            resize              = args.data_val_resize,
        ), [mx.gpu(gpu) for gpu in gpus], args.dtype)
    return (train, val)


class SyntheticDataIter(DataIter):
    def __init__(self, num_classes, data_shape, max_iter, ctx, dtype):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        self.dtype = dtype
        label = np.random.randint(0, num_classes, [self.batch_size,])
        data = np.random.uniform(-1, 1, data_shape)
        self.data = []
        self.label = []
        self._num_gpus = len(ctx)
        for dev in ctx:
            self.data.append(mx.nd.array(data, dtype=self.dtype, ctx=dev))
            self.label.append(mx.nd.array(label, dtype=self.dtype, ctx=dev))

    def __iter__(self):
        return self

    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return [DataBatch(data=(data,), label=(label,), pad=0) for data, label in zip(self.data, self.label)]
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def reset(self):
        self.cur_iter = 0

def get_synthetic_rec_iter(args, kv=None):
    gpus = args.gpus
    if 'horovod' in args.kv_store:
        gpus = [gpus[0]]
        batch_size = args.batch_size // hvd.size()
    else:
        batch_size = args.batch_size

    if args.input_layout == 'NCHW':
        data_shape = (batch_size, *args.image_shape)
    elif args.input_layout == 'NHWC':
        data_shape = (batch_size, *args.image_shape[1:], args.image_shape[0])
    else:
        raise ValueError('Wrong input layout')

    train = SyntheticDataIter(args.num_classes, data_shape,
                              args.num_examples // args.batch_size,
                              [mx.gpu(gpu) for gpu in gpus], args.dtype)
    if args.data_val is None:
        return (train, None)

    val = SyntheticDataIter(args.num_classes, data_shape,
                            args.num_examples // args.batch_size,
                            [mx.gpu(gpu) for gpu in gpus], args.dtype)
    return (train, val)

def load_image(args, path, ctx=mx.cpu()):
    image = mx.image.imread(path).astype('float32')
    image = mx.image.imresize(image, *args.image_shape[1:])
    image = (image - nd.array(args.rgb_mean)) / nd.array(args.rgb_std)
    image = image.as_in_context(ctx)
    if args.input_layout == 'NCHW':
        image = image.transpose((2, 0, 1))
    image = image.astype(args.dtype)
    if args.image_shape[0] == 4:
        dim = 0 if args.input_layout == 'NCHW' else 2
        image = nd.concat(image, nd.zeros((1, *image.shape[1:]), dtype=image.dtype, ctx=image.context), dim=dim)
    return image
