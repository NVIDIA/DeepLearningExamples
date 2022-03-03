# Lint as: python3
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
# ==============================================================================
"""Contains definitions for EfficientNet v1 model.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from typing import Any, Dict, Optional, List, Text, Tuple
import copy

import tensorflow as tf
import tensorflow.keras.backend as K
import horovod.tensorflow as hvd

from utils.optimizer_factory import GradientAccumulator
from model.layers import simple_swish, hard_swish, identity, gelu, get_activation
from model.blocks import conv2d_block, mb_conv_block
from model.common_modules import round_filters, round_repeats, load_weights
from dataloader import preprocessing
from dataloader.dataset_factory import mixing_lite

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1 / 3.0,
        'mode': 'fan_in',
        'distribution': 'uniform'
    }
}



@tf.keras.utils.register_keras_serializable(package='Vision')
class Model(tf.keras.Model):
  """Wrapper class for an EfficientNet v1 Keras model.

  Contains helper methods to build, manage, and save metadata about the model.
  """

  def __init__(self,
               config: Dict[Text, Any] = None):
    """Create an EfficientNet v1 model.

    Args:
      config: (optional) the main model parameters to create the model
      overrides: (optional) a dict containing keys that can override
                 config
    """
    super().__init__()
    self.config = config
    if self.config.grad_accum_steps > 1:
      self.grad_accumulator = GradientAccumulator()
      self.gradients_gnorm = tf.Variable(0, trainable=False, dtype=tf.float32)
      self.local_step = tf.Variable(initial_value=0, dtype=tf.int64, trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
      
    input_channels = config.mparams.input_channels

    # Consistent with channels last format. will be permuted in _build, if channels first requested.
    input_shape = (None, None, input_channels)  # Should handle any image size
    image_input = tf.keras.layers.Input(shape=input_shape)  
    is_training ="predict" not in config.mode
    if is_training:
      mixup_input = tf.keras.layers.Input(shape=(1, 1, 1))
      cutmix_input = tf.keras.layers.Input(shape=(None, None, 1))
      is_tr_split = tf.keras.layers.Input(shape=(1)) # indicates whether we use tr or eval data loader
      inputs = [image_input,mixup_input,cutmix_input,is_tr_split]
    else:
      inputs = [image_input]
    output = self._build(inputs)

    # Cast to float32 in case we have a different model dtype
    output = tf.cast(output, tf.float32)
    
    # defining a Model object within another Model object is not the best design idea,
    # but I wanted to make use of existing functional API code from Subhankar
    self.model = tf.keras.Model(inputs=inputs,outputs=output)
  
  def call(self,data):
    is_predict ="predict" in self.config.mode
    if not is_predict:
      x=data['image']
      mixup_weights = data['mixup_weight']
      cutmix_masks = data['cutmix_mask']
      is_tr_split = data['is_tr_split']
      return self.model([x,mixup_weights,cutmix_masks,is_tr_split])
    else:
      return self.model([data])
    
  def _build(self,
            input: List[tf.keras.layers.Input]):
    """Creates an EfficientNet v1 graph given the model parameters.

    This function is wrapped by the `EfficientNet_v1` class to make a tf.keras.Model.

    Args:
      image_input: the input batch of images

    Returns:
      the output of efficientnet v1
    """
    config = self.config
    depth_coefficient = config.mparams.depth_coefficient
    blocks = config.mparams.blocks
    stem_base_filters = config.mparams.stem_base_filters
    top_base_filters = config.mparams.top_base_filters
    activation = get_activation(config.mparams.activation)
    dropout_rate = config.mparams.dropout_rate
    drop_connect_rate = config.mparams.drop_connect_rate
    num_classes = config.mparams.num_classes
    input_channels = config.mparams.input_channels
    rescale_input = config.mparams.rescale_input
    data_format = tf.keras.backend.image_data_format()
    dtype = config.mparams.dtype
    weight_decay = config.weight_decay
    weight_init = config.mparams.weight_init
    train_batch_size = config.train_batch_size
    do_mixup = config.mixup_alpha > 0
    do_cutmix = config.cutmix_alpha > 0
    
    def cond_mixing(args):
      images,mixup_weights,cutmix_masks,is_tr_split = args
      return tf.cond(tf.keras.backend.equal(is_tr_split[0],0), 
                     lambda: images, # eval phase
                     lambda: mixing_lite(images,mixup_weights,cutmix_masks, train_batch_size, do_mixup, do_cutmix)) # tr phase
      
    images = input[0]
    x = images
    if len(input) > 1:
      # we get here only during train or train_and_eval modes
      if self.config.defer_img_mixing:
        # we get here only if we chose not to perform image mixing in the data loader
        # image mixing on device further accelrates training
        mixup_weights = input[1]
        cutmix_masks = input[2]
        is_tr_split = input[3]
        x = tf.keras.layers.Lambda(cond_mixing)([images,mixup_weights,cutmix_masks,is_tr_split])
      
      

    # data loader outputs data in the channels last format
    if data_format == 'channels_first':
      # Happens on GPU/TPU if available.
      x = tf.keras.layers.Permute((3, 1, 2))(x)
      
    if rescale_input:
      # x-mean/std
      x = preprocessing.normalize_images(x,
                                        mean_rgb=config.mparams.mean_rgb,
                                        stddev_rgb=config.mparams.std_rgb,
                                        num_channels=input_channels,
                                        dtype=dtype,
                                        data_format=data_format)

    # Build stem
    x = conv2d_block(x,
                    round_filters(stem_base_filters, config),
                    config,
                    kernel_size=[3, 3],
                    strides=[2, 2],
                    activation=activation,
                    name='stem')

    # Build blocks
    num_blocks_total = sum(
        round_repeats(block['num_repeat'], depth_coefficient) for block in blocks)
    block_num = 0

    for stack_idx, block in enumerate(blocks):
      assert block['num_repeat'] > 0
      # Update block input and output filters based on depth multiplier
      block.update({
          'input_filters':round_filters(block['input_filters'], config),
          'output_filters':round_filters(block['output_filters'], config),
          'num_repeat':round_repeats(block['num_repeat'], depth_coefficient)})

      # The first block needs to take care of stride and filter size increase
      drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
      config.mparams.update({'drop_connect_rate': drop_rate})  # TODO(Sugh) replace
      block_prefix = 'stack_{}/block_0/'.format(stack_idx)
      x = mb_conv_block(x, block, config, block_prefix)
      block_num += 1
      if block['num_repeat'] > 1:
        block.update({
            'input_filters':block['output_filters'],
            'strides':(1, 1)
        })

        for block_idx in range(block['num_repeat'] - 1):
          drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
          config.mparams.update({'drop_connect_rate': drop_rate})
          block_prefix = 'stack_{}/block_{}/'.format(stack_idx, block_idx + 1)
          x = mb_conv_block(x, block, config, prefix=block_prefix)
          block_num += 1

    # Build top
    x = conv2d_block(x,
                    round_filters(top_base_filters, config),
                    config,
                    activation=activation,
                    name='top')

    # Build classifier
    DENSE_KERNEL_INITIALIZER['config']['mode'] = weight_init
    x = tf.keras.layers.GlobalAveragePooling2D(name='top_pool')(x)
    if dropout_rate and dropout_rate > 0:
      x = tf.keras.layers.Dropout(dropout_rate, name='top_dropout')(x)
    x = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer=DENSE_KERNEL_INITIALIZER,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(weight_decay),
        name='logits')(x)
    x = tf.keras.layers.Activation('softmax', name='probs', dtype=tf.float32)(x)

    return x

