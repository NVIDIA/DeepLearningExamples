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

'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
(Original author Wei Wu) by Antti-Pekka Hynninen

"Flexible Layout" (fl) version created by Dick Carter.

Implementing the original resnet ILSVRC 2015 winning network from:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"
'''
import mxnet as mx
import numpy as np
import random

# Transform a symbol from one layout to another, or do nothing if they have the same layout
def transform_layout(data, from_layout, to_layout):
    supported_layouts = ['NCHW', 'NHWC']
    if from_layout not in supported_layouts:
        raise ValueError('Not prepared to handle layout: {}'.format(from_layout))
    if to_layout not in supported_layouts:
        raise ValueError('Not prepared to handle layout: {}'.format(to_layout))

    # Insert transpose if from_layout and to_layout don't match
    if from_layout == 'NCHW' and to_layout == 'NHWC':
        return mx.sym.transpose(data, axes=(0, 2, 3, 1))
    elif from_layout == 'NHWC' and to_layout == 'NCHW':
        return mx.sym.transpose(data, axes=(0, 3, 1, 2))
    else:
        return data

# A BatchNorm wrapper that responds to the input layout
def batchnorm(data, io_layout, batchnorm_layout, **kwargs):
    # Transpose as needed to batchnorm_layout
    transposed_as_needed = transform_layout(data, io_layout, batchnorm_layout)
    bn_axis = 3 if batchnorm_layout == 'NHWC' else 1
    batchnormed = mx.sym.BatchNorm(data=transposed_as_needed, axis=bn_axis, **kwargs)
    # Transpose back to i/o layout as needed
    return transform_layout(batchnormed, batchnorm_layout, io_layout)

# A BatchNormAddRelu wrapper that responds to the input layout
def batchnorm_add_relu(data, addend, io_layout, batchnorm_layout, **kwargs):
    # Transpose as needed to batchnorm_layout
    transposed_data_as_needed = transform_layout(data, io_layout, batchnorm_layout)
    transposed_addend_as_needed = transform_layout(addend, io_layout, batchnorm_layout)
    bn_axis = 3 if batchnorm_layout == 'NHWC' else 1
    batchnormed = mx.sym.BatchNormAddRelu(data=transposed_data_as_needed,
                                      addend=transposed_addend_as_needed,
                                      axis=bn_axis, **kwargs)
    # Transpose back to i/o layout as needed
    return transform_layout(batchnormed, batchnorm_layout, io_layout)

# A Pooling wrapper that responds to the input layout
def pooling(data, io_layout, pooling_layout, **kwargs):
    # Pooling kernel, as specified by pooling_layout, may be in conflict with i/o layout.
    transposed_as_needed = transform_layout(data, io_layout, pooling_layout)
    pooled = mx.sym.Pooling(data=transposed_as_needed, layout=pooling_layout, **kwargs)
    # Transpose back to i/o layout as needed
    return transform_layout(pooled, pooling_layout, io_layout)

# Assumption is that data comes in and out in the 'conv_layout' format.
# If this format is different from the 'batchnorm_layout' format, then the batchnorm() routine
# will introduce transposes on both sides of the mx.sym.BatchNorm symbol
def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True,
                  workspace=256, memonger=False, conv_layout='NCHW', batchnorm_layout='NCHW',
                  verbose=False, cudnn_bn_off=False, bn_eps=2e-5, bn_mom=0.9, conv_algo=-1,
                  fuse_bn_relu=False, fuse_bn_add_relu=False, cudnn_tensor_core_only=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """

    act = 'relu' if fuse_bn_relu else None
    if bottle_neck:
        conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1', layout=conv_layout,
                                   cudnn_algo_verbose=verbose,
                                   cudnn_algo_fwd=conv_algo, cudnn_algo_bwd_data=conv_algo, cudnn_algo_bwd_filter=conv_algo,
                                   cudnn_tensor_core_only=cudnn_tensor_core_only)
        bn1 = batchnorm(data=conv1, io_layout=conv_layout, batchnorm_layout=batchnorm_layout,
                        fix_gamma=False, eps=bn_eps, momentum=bn_mom, name=name + '_bn1', cudnn_off=cudnn_bn_off, act_type=act)
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1') if not fuse_bn_relu else bn1
        conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2', layout=conv_layout,
                                   cudnn_algo_verbose=verbose,
                                   cudnn_algo_fwd=conv_algo, cudnn_algo_bwd_data=conv_algo, cudnn_algo_bwd_filter=conv_algo,
                                   cudnn_tensor_core_only=cudnn_tensor_core_only)
        bn2 = batchnorm(data=conv2, io_layout=conv_layout, batchnorm_layout=batchnorm_layout,
                        fix_gamma=False, eps=bn_eps, momentum=bn_mom, name=name + '_bn2', cudnn_off=cudnn_bn_off, act_type=act)
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2') if not fuse_bn_relu else bn2
        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3', layout=conv_layout,
                                   cudnn_algo_verbose=verbose,
                                   cudnn_algo_fwd=conv_algo, cudnn_algo_bwd_data=conv_algo, cudnn_algo_bwd_filter=conv_algo,
                                   cudnn_tensor_core_only=cudnn_tensor_core_only)

        if dim_match:
            shortcut = data
        else:
            conv1sc = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc', layout=conv_layout,
                                         cudnn_algo_verbose=verbose,
                                         cudnn_algo_fwd=conv_algo, cudnn_algo_bwd_data=conv_algo, cudnn_algo_bwd_filter=conv_algo,
                                         cudnn_tensor_core_only=cudnn_tensor_core_only)
            shortcut = batchnorm(data=conv1sc, io_layout=conv_layout, batchnorm_layout=batchnorm_layout,
                                 fix_gamma=False, eps=bn_eps, momentum=bn_mom, name=name + '_sc', cudnn_off=cudnn_bn_off)
        if memonger:
            shortcut._set_attr(mirror_stage='True')

        if fuse_bn_add_relu:
            return batchnorm_add_relu(data=conv3, addend=shortcut, io_layout=conv_layout, batchnorm_layout=batchnorm_layout,
                            fix_gamma=False, eps=bn_eps, momentum=bn_mom, name=name + '_bn3', cudnn_off=cudnn_bn_off)
        else:
            bn3 = batchnorm(data=conv3, io_layout=conv_layout, batchnorm_layout=batchnorm_layout,
                            fix_gamma=False, eps=bn_eps, momentum=bn_mom, name=name + '_bn3', cudnn_off=cudnn_bn_off)
            return mx.sym.Activation(data=bn3 + shortcut, act_type='relu', name=name + '_relu3')

    else:
        conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1', layout=conv_layout,
                                   cudnn_algo_verbose=verbose,
                                   cudnn_algo_fwd=conv_algo, cudnn_algo_bwd_data=conv_algo, cudnn_algo_bwd_filter=conv_algo,
                                   cudnn_tensor_core_only=cudnn_tensor_core_only)
        bn1 = batchnorm(data=conv1, io_layout=conv_layout, batchnorm_layout=batchnorm_layout,
                        fix_gamma=False, momentum=bn_mom, eps=bn_eps, name=name + '_bn1', cudnn_off=cudnn_bn_off, act_type=act)
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1') if not fuse_bn_relu else bn1
        conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2', layout=conv_layout,
                                   cudnn_algo_verbose=verbose,
                                   cudnn_algo_fwd=conv_algo, cudnn_algo_bwd_data=conv_algo, cudnn_algo_bwd_filter=conv_algo,
                                   cudnn_tensor_core_only=cudnn_tensor_core_only)

        if dim_match:
            shortcut = data
        else:
            conv1sc = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc', layout=conv_layout,
                                         cudnn_algo_verbose=verbose,
                                         cudnn_algo_fwd=conv_algo, cudnn_algo_bwd_data=conv_algo, cudnn_algo_bwd_filter=conv_algo,
                                         cudnn_tensor_core_only=cudnn_tensor_core_only)
            shortcut = batchnorm(data=conv1sc, io_layout=conv_layout, batchnorm_layout=batchnorm_layout,
                                 fix_gamma=False, momentum=bn_mom, eps=bn_eps, name=name + '_sc', cudnn_off=cudnn_bn_off)
        if memonger:
            shortcut._set_attr(mirror_stage='True')

        if fuse_bn_add_relu:
            return batchnorm_add_relu(data=conv2, addend=shortcut, io_layout=conv_layout, batchnorm_layout=batchnorm_layout,
                            fix_gamma=False, momentum=bn_mom, eps=bn_eps, name=name + '_bn2', cudnn_off=cudnn_bn_off)
        else:
            bn2 = batchnorm(data=conv2, io_layout=conv_layout, batchnorm_layout=batchnorm_layout,
                            fix_gamma=False, momentum=bn_mom, eps=bn_eps, name=name + '_bn2', cudnn_off=cudnn_bn_off)
            return mx.sym.Activation(data=bn2 + shortcut, act_type='relu', name=name + '_relu2')

def resnet(units, num_stages, filter_list, num_classes, image_shape, bottle_neck=True, workspace=256, dtype='float32', memonger=False,
           input_layout='NCHW', conv_layout='NCHW',  batchnorm_layout='NCHW', pooling_layout='NCHW', verbose=False,
           cudnn_bn_off=False, bn_eps=2e-5, bn_mom=0.9, conv_algo=-1,
           fuse_bn_relu=False, fuse_bn_add_relu=False, force_tensor_core=False, use_dali=True):
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    dtype : str
        Precision (float32 or float16)
    memonger : boolean
        Activates "memory monger" to reduce the model's memory footprint
    input_layout : str
        interpretation (e.g. NCHW vs NHWC) of data provided by the i/o pipeline (may introduce transposes
        if in conflict with 'layout' above)
    conv_layout : str
        interpretation (e.g. NCHW vs NHWC) of data for convolution operation.
    batchnorm_layout : str
        directs which kernel performs the batchnorm (may introduce transposes if in conflict with 'conv_layout' above)
    pooling_layout : str
        directs which kernel performs the pooling (may introduce transposes if in conflict with 'conv_layout' above)
    """

    act = 'relu' if fuse_bn_relu else None
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    if not use_dali:
        # double buffering of data
        if dtype == 'float32':
            data = mx.sym.identity(data=data, name='id')
        else:
            if dtype == 'float16':
                data = mx.sym.Cast(data=data, dtype=np.float16)
    (nchannel, height, width) = image_shape

    # Insert transpose as needed to get the input layout to match the desired processing layout
    data = transform_layout(data, input_layout, conv_layout)

    if height <= 32:            # such as cifar10
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace, layout=conv_layout,
                                  cudnn_algo_verbose=verbose,
                                  cudnn_algo_fwd=conv_algo, cudnn_algo_bwd_data=conv_algo, cudnn_algo_bwd_filter=conv_algo,
                                  cudnn_tensor_core_only=force_tensor_core)
        # Is this BatchNorm supposed to be here?
        body = batchnorm(data=body, io_layout=conv_layout, batchnorm_layout=batchnorm_layout,
                         fix_gamma=False, eps=bn_eps, momentum=bn_mom, name='bn0', cudnn_off=cudnn_bn_off)
    else:                       # often expected to be 224 such as imagenet
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace, layout=conv_layout,
                                  cudnn_algo_verbose=verbose,
                                  cudnn_algo_fwd=conv_algo, cudnn_algo_bwd_data=conv_algo, cudnn_algo_bwd_filter=conv_algo,
                                  cudnn_tensor_core_only=force_tensor_core)
        body = batchnorm(data=body, io_layout=conv_layout, batchnorm_layout=batchnorm_layout,
                         fix_gamma=False, eps=bn_eps, momentum=bn_mom, name='bn0', cudnn_off=cudnn_bn_off, act_type=act)
        if not fuse_bn_relu:
            body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = pooling(data=body, io_layout=conv_layout, pooling_layout=pooling_layout,
                       kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1),
                             bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger, conv_layout=conv_layout, batchnorm_layout=batchnorm_layout,
                             verbose=verbose, cudnn_bn_off=cudnn_bn_off, bn_eps=bn_eps, bn_mom=bn_mom,
                             conv_algo=conv_algo, fuse_bn_relu=fuse_bn_relu, fuse_bn_add_relu=fuse_bn_add_relu,
                             cudnn_tensor_core_only=force_tensor_core)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace,
                                 memonger=memonger, conv_layout=conv_layout, batchnorm_layout=batchnorm_layout,
                                 verbose=verbose, cudnn_bn_off=cudnn_bn_off, bn_eps = bn_eps, bn_mom=bn_mom,
                                 conv_algo=conv_algo, fuse_bn_relu=fuse_bn_relu, fuse_bn_add_relu=fuse_bn_add_relu,
                                 cudnn_tensor_core_only=force_tensor_core)
    # bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    # relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = pooling(data=body, io_layout=conv_layout, pooling_layout=pooling_layout,
                    global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='fc1', cublas_algo_verbose=verbose)
    if dtype == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
    return mx.sym.SoftmaxOutput(data=fc1, name='softmax')

def get_symbol(num_classes, num_layers, image_shape, conv_workspace=256, dtype='float32',
               input_layout='NCHW', conv_layout='NCHW', batchnorm_layout='NCHW', pooling_layout='NCHW',
               verbose=False, seed=None, cudnn_bn_off=False, batchnorm_eps=2e-5, batchnorm_mom=0.9,
               conv_algo=-1, fuse_bn_relu=False, fuse_bn_add_relu=False, force_tensor_core=False, use_dali=True, **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
    (Original author Wei Wu) by Antti-Pekka Hynninen
    Implementing the original resnet ILSVRC 2015 winning network from:
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"
    """
    if seed is not None:
        print('Setting seeds to %s' % (seed,))
        random.seed(seed)
        np.random.seed(seed)
        mx.random.seed(seed)

    image_shape = [int(l) for l in image_shape.split(',')]
    (nchannel, height, width) = image_shape
    if height <= 28:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    return resnet(units             = units,
                  num_stages        = num_stages,
                  filter_list       = filter_list,
                  num_classes       = num_classes,
                  image_shape       = image_shape,
                  bottle_neck       = bottle_neck,
                  workspace         = conv_workspace,
                  dtype             = dtype,
                  input_layout      = input_layout,
                  conv_layout       = conv_layout,
                  batchnorm_layout  = batchnorm_layout,
                  pooling_layout    = pooling_layout,
                  verbose           = verbose,
                  cudnn_bn_off      = cudnn_bn_off,
                  bn_eps            = batchnorm_eps,
                  bn_mom            = batchnorm_mom,
                  conv_algo         = conv_algo,
                  fuse_bn_relu      = fuse_bn_relu,
                  fuse_bn_add_relu  = fuse_bn_add_relu,
                  force_tensor_core = force_tensor_core,
                  use_dali          = use_dali)
