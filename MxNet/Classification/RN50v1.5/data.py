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
import random
import argparse
from mxnet.io import DataBatch, DataIter
import numpy as np

def add_data_args(parser):
    data = parser.add_argument_group('Data', 'the input images')
    data.add_argument('--data-train', type=str, help='the training data')
    data.add_argument('--data-train-idx', type=str, default='', help='the index of training data')
    data.add_argument('--data-val', type=str, help='the validation data')
    data.add_argument('--data-val-idx', type=str, default='', help='the index of validation data')
    data.add_argument('--rgb-mean', type=str, default='123.68,116.779,103.939',
                      help='a tuple of size 3 for the mean rgb')
    data.add_argument('--rgb-std', type=str, default='1,1,1',
                      help='a tuple of size 3 for the std rgb')
    data.add_argument('--pad-size', type=int, default=0,
                      help='padding the input image')
    data.add_argument('--fill-value', type=int, default=127,
                      help='Set the padding pixels value to fill_value')
    data.add_argument('--image-shape', type=str,
                      help='the image shape feed into the network, e.g. (3,224,224)')
    data.add_argument('--num-classes', type=int, help='the number of classes')
    data.add_argument('--num-examples', type=int, help='the number of training examples')
    data.add_argument('--data-nthreads', type=int, default=4,
                      help='number of threads for data decoding')
    data.add_argument('--benchmark-iters', type=int, default=None,
                      help='run only benchmark-iters iterations from each epoch')
    data.add_argument('--input-layout', type=str, default='NCHW',
                      help='the layout of the input data (e.g. NCHW)')
    data.add_argument('--conv-layout', type=str, default='NCHW',
                      help='the layout of the data assumed by the conv operation (e.g. NCHW)')
    data.add_argument('--conv-algo', type=int, default=-1,
                      help='set the convolution algos (fwd, dgrad, wgrad)')
    data.add_argument('--batchnorm-layout', type=str, default='NCHW',
                      help='the layout of the data assumed by the batchnorm operation (e.g. NCHW)')
    data.add_argument('--batchnorm-eps', type=float, default=2e-5,
                      help='the amount added to the batchnorm variance to prevent output explosion.')
    data.add_argument('--batchnorm-mom', type=float, default=0.9,
                      help='the leaky-integrator factor controling the batchnorm mean and variance.')
    data.add_argument('--pooling-layout', type=str, default='NCHW',
                      help='the layout of the data assumed by the pooling operation (e.g. NCHW)')
    data.add_argument('--verbose', type=int, default=0,
                      help='turn on reporting of chosen algos for convolution, etc.')
    data.add_argument('--seed', type=int, default=None,
                      help='set the seed for python, nd and mxnet rngs')
    data.add_argument('--custom-bn-off', type=int, default=0,
                      help='disable use of custom batchnorm kernel')
    data.add_argument('--fuse-bn-relu', type=int, default=0,
                      help='have batchnorm kernel perform activation relu')
    data.add_argument('--fuse-bn-add-relu', type=int, default=0,
                      help='have batchnorm kernel perform add followed by activation relu')
    data.add_argument('--force-tensor-core', type=int, default=0,
                      help='require conv algos to be tensor core')
    return data

# Action to translate --set-resnet-aug flag to its component settings.
class SetResnetAugAction(argparse.Action):
    def __init__(self, nargs=0, **kwargs):
        if nargs != 0:
            raise ValueError('nargs for SetResnetAug must be 0.')
        super(SetResnetAugAction, self).__init__(nargs=nargs, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        # standard data augmentation setting for resnet training
        setattr(namespace, 'random_crop', 1)
        setattr(namespace, 'random_resized_crop', 1)
        setattr(namespace, 'random_mirror', 1)
        setattr(namespace, 'min_random_area', 0.08)
        setattr(namespace, 'max_random_aspect_ratio', 4./3.)
        setattr(namespace, 'min_random_aspect_ratio', 3./4.)
        setattr(namespace, 'brightness', 0.4)
        setattr(namespace, 'contrast', 0.4)
        setattr(namespace, 'saturation', 0.4)
        setattr(namespace, 'pca_noise', 0.1)
        # record that this --set-resnet-aug 'macro arg' has been invoked
        setattr(namespace, self.dest, 1)

# Similar to the above, but suitable for calling within a training script to set the defaults.
def set_resnet_aug(aug):
    # standard data augmentation setting for resnet training
    aug.set_defaults(random_crop=0, random_resized_crop=1)
    aug.set_defaults(random_mirror=1)
    aug.set_defaults(min_random_area=0.08)
    aug.set_defaults(max_random_aspect_ratio=4./3., min_random_aspect_ratio=3./4.)
    aug.set_defaults(brightness=0.4, contrast=0.4, saturation=0.4, pca_noise=0.1)

# Action to translate --set-data-aug-level <N> arg to its component settings.
class SetDataAugLevelAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(SetDataAugLevelAction, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        level = values
        # record that this --set-data-aug-level <N> 'macro arg' has been invoked
        setattr(namespace, self.dest, level)
        if level >= 1:
            setattr(namespace, 'random_crop', 1)
            setattr(namespace, 'random_mirror', 1)
        if level >= 2:
            setattr(namespace, 'max_random_h', 36)
            setattr(namespace, 'max_random_s', 50)
            setattr(namespace, 'max_random_l', 50)
        if level >= 3:
            setattr(namespace, 'max_random_rotate_angle', 10)
            setattr(namespace, 'max_random_shear_ratio', 0.1)
            setattr(namespace, 'max_random_aspect_ratio', 0.25)

# Similar to the above, but suitable for calling within a training script to set the defaults.
def set_data_aug_level(aug, level):
    if level >= 1:
        aug.set_defaults(random_crop=1, random_mirror=1)
    if level >= 2:
        aug.set_defaults(max_random_h=36, max_random_s=50, max_random_l=50)
    if level >= 3:
        aug.set_defaults(max_random_rotate_angle=10, max_random_shear_ratio=0.1, max_random_aspect_ratio=0.25)

def add_data_aug_args(parser):
    aug = parser.add_argument_group(
        'Image augmentations', 'implemented in src/io/image_aug_default.cc')
    aug.add_argument('--random-crop', type=int, default=0,
                     help='if or not randomly crop the image')
    aug.add_argument('--random-mirror', type=int, default=0,
                     help='if or not randomly flip horizontally')
    aug.add_argument('--max-random-h', type=int, default=0,
                     help='max change of hue, whose range is [0, 180]')
    aug.add_argument('--max-random-s', type=int, default=0,
                     help='max change of saturation, whose range is [0, 255]')
    aug.add_argument('--max-random-l', type=int, default=0,
                     help='max change of intensity, whose range is [0, 255]')
    aug.add_argument('--min-random-aspect-ratio', type=float, default=None,
                     help='min value of aspect ratio, whose value is either None or a positive value.')
    aug.add_argument('--max-random-aspect-ratio', type=float, default=0,
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
    aug.add_argument('--min-random-area', type=float, default=1,
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
    aug.add_argument('--random-resized-crop', type=int, default=0,
                     help='whether to use random resized crop')
    aug.add_argument('--set-resnet-aug', action=SetResnetAugAction,
                     help='whether to employ standard resnet augmentations (see data.py)')
    aug.add_argument('--set-data-aug-level', type=int, default=None, action=SetDataAugLevelAction,
                     help='set multiple data augmentations based on a `level` (see data.py)')
    return aug

def get_rec_iter(args, kv=None):
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    if args.input_layout == 'NHWC':
        image_shape = image_shape[1:] + (image_shape[0],)
    if kv:
        (rank, nworker) = (kv.rank, kv.num_workers)
    else:
        (rank, nworker) = (0, 1)
    rgb_mean = [float(i) for i in args.rgb_mean.split(',')]
    rgb_std = [float(i) for i in args.rgb_std.split(',')]
    if args.input_layout == 'NHWC':
        raise ValueError('ImageRecordIter cannot handle layout {}'.format(args.input_layout))

    train = mx.io.ImageRecordIter(
        path_imgrec         = args.data_train,
        path_imgidx         = args.data_train_idx,
        label_width         = 1,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        std_r               = rgb_std[0],
        std_g               = rgb_std[1],
        std_b               = rgb_std[2],
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = image_shape,
        batch_size          = args.batch_size,
        rand_crop           = args.random_crop,
        max_random_scale    = args.max_random_scale,
        pad                 = args.pad_size,
        fill_value          = args.fill_value,
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
        preprocess_threads  = args.data_nthreads,
        shuffle             = True,
        num_parts           = nworker,
        part_index          = rank)
    if args.data_val is None:
        return (train, None)
    val = mx.io.ImageRecordIter(
        path_imgrec         = args.data_val,
        path_imgidx         = args.data_val_idx,
        label_width         = 1,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        std_r               = rgb_std[0],
        std_g               = rgb_std[1],
        std_b               = rgb_std[2],
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = args.batch_size,
        round_batch         = False,
        data_shape          = image_shape,
        preprocess_threads  = args.data_nthreads,
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = nworker,
        part_index          = rank)
    return (train, val)
