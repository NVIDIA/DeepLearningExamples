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
"""Contains definitions for EfficientNet model.

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

from model.layers import simple_swish, hard_swish, identity, gelu, get_activation
from model.blocks import conv2d_block, mb_conv_block
from model.common_modules import round_filters, round_repeats, load_weights
from utils import preprocessing


def build_dict(name, args=None):
    if name == "ModelConfig":
        return_dict = copy.deepcopy(ModelConfig)
    elif name == "BlockConfig":
        return_dict = copy.deepcopy(BlockConfig)
    else:
        raise ValueError("Name of requested dictionary not found!")
    if args is None:
      return return_dict
    if isinstance(args, dict):
        return_dict.update(args)
    elif isinstance(args, tuple):
        return_dict.update( {a: p for a, p in zip(list(return_dict.keys()), args)} )
    else:
        raise ValueError("Expected tuple or dict!")
    return return_dict

# Config for a single MB Conv Block.
BlockConfig = {
  'input_filters': 0,
  'output_filters': 0,
  'kernel_size': 3,
  'num_repeat': 1,
  'expand_ratio': 1,
  'strides': (1, 1),
  'se_ratio': None,
  'id_skip': True,
  'fused_conv': False,
  'conv_type': 'depthwise'
  }

# Default Config for Efficientnet-B0.
ModelConfig = {
  'width_coefficient': 1.0,
  'depth_coefficient': 1.0,
  'resolution': 224,
  'dropout_rate': 0.2,
  'blocks': (
      # (input_filters, output_filters, kernel_size, num_repeat,
      #  expand_ratio, strides, se_ratio)
      # pylint: disable=bad-whitespace
      build_dict(name="BlockConfig", args=(32,  16,  3, 1, 1, (1, 1), 0.25)),
      build_dict(name="BlockConfig", args=(16,  24,  3, 2, 6, (2, 2), 0.25)),
      build_dict(name="BlockConfig", args=(24,  40,  5, 2, 6, (2, 2), 0.25)),
      build_dict(name="BlockConfig", args=(40,  80,  3, 3, 6, (2, 2), 0.25)),
      build_dict(name="BlockConfig", args=(80,  112, 5, 3, 6, (1, 1), 0.25)),
      build_dict(name="BlockConfig", args=(112, 192, 5, 4, 6, (2, 2), 0.25)),
      build_dict(name="BlockConfig", args=(192, 320, 3, 1, 6, (1, 1), 0.25)),
      # pylint: enable=bad-whitespace
  ),
  'stem_base_filters': 32,
  'top_base_filters': 1280,
  'activation': 'simple_swish',
  'batch_norm': 'default',
  'bn_momentum': 0.99,
  'bn_epsilon': 1e-3,
  # While the original implementation used a weight decay of 1e-5,
  # tf.nn.l2_loss divides it by 2, so we halve this to compensate in Keras
  'weight_decay': 5e-6,
  'drop_connect_rate': 0.2,
  'depth_divisor': 8,
  'min_depth': None,
  'use_se': True,
  'input_channels': 3,
  'num_classes': 1000,
  'model_name': 'efficientnet',
  'rescale_input': True,
  'data_format': 'channels_last',
  'dtype': 'float32',
  'weight_init': 'fan_in',
}

MODEL_CONFIGS = {
    # (width, depth, resolution, dropout)
    'efficientnet-b0': build_dict(name="ModelConfig", args=(1.0, 1.0, 224, 0.2)),
    'efficientnet-b1': build_dict(name="ModelConfig", args=(1.0, 1.1, 240, 0.2)),
    'efficientnet-b2': build_dict(name="ModelConfig", args=(1.1, 1.2, 260, 0.3)),
    'efficientnet-b3': build_dict(name="ModelConfig", args=(1.2, 1.4, 300, 0.3)),
    'efficientnet-b4': build_dict(name="ModelConfig", args=(1.4, 1.8, 380, 0.4)),
    'efficientnet-b5': build_dict(name="ModelConfig", args=(1.6, 2.2, 456, 0.4)),
    'efficientnet-b6': build_dict(name="ModelConfig", args=(1.8, 2.6, 528, 0.5)),
    'efficientnet-b7': build_dict(name="ModelConfig", args=(2.0, 3.1, 600, 0.5)),
    'efficientnet-b8': build_dict(name="ModelConfig", args=(2.2, 3.6, 672, 0.5)),
    'efficientnet-l2': build_dict(name="ModelConfig", args=(4.3, 5.3, 800, 0.5)),
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1 / 3.0,
        'mode': 'fan_in',
        'distribution': 'uniform'
    }
}


def efficientnet(input: List[tf.keras.layers.Input],
                 config: dict):
  """Creates an EfficientNet graph given the model parameters.

  This function is wrapped by the `EfficientNet` class to make a tf.keras.Model.

  Args:
    image_input: the input batch of images
    config: the model config

  Returns:
    the output of efficientnet
  """
  depth_coefficient = config['depth_coefficient']
  blocks = config['blocks']
  stem_base_filters = config['stem_base_filters']
  top_base_filters = config['top_base_filters']
  activation = get_activation(config['activation'])
  dropout_rate = config['dropout_rate']
  drop_connect_rate = config['drop_connect_rate']
  num_classes = config['num_classes']
  input_channels = config['input_channels']
  rescale_input = config['rescale_input']
  data_format = tf.keras.backend.image_data_format()
  dtype = config['dtype']
  weight_decay = config['weight_decay']
  weight_init = config['weight_init']

  # Move the mixup of images to device
  images = input[0]
  if len(input) > 1:
    mix_weight = input[1]
    x = (images * mix_weight + images[::-1] * (1. - mix_weight))
  else:
    x = images

  if data_format == 'channels_first':
    # Happens on GPU/TPU if available.
    x = tf.keras.layers.Permute((3, 1, 2))(x)
  if rescale_input:
    x = preprocessing.normalize_images(x,
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
    config.update({'drop_connect_rate': drop_rate})  # TODO(Sugh) replace
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
        config.update({'drop_connect_rate': drop_rate})
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


@tf.keras.utils.register_keras_serializable(package='Vision')
class EfficientNet(tf.keras.Model):
  """Wrapper class for an EfficientNet Keras model.

  Contains helper methods to build, manage, and save metadata about the model.
  """

  def __init__(self,
               config: Dict[Text, Any] = None,
               overrides: Dict[Text, Any] = None):
    """Create an EfficientNet model.

    Args:
      config: (optional) the main model parameters to create the model
      overrides: (optional) a dict containing keys that can override
                 config
    """
    overrides = overrides or {}
    is_training = overrides.pop('is_training', False)
    config = config or build_dict(name="ModelConfig")
    self.config = config
    self.config.update(overrides)


    input_channels = self.config['input_channels']
    model_name = self.config['model_name']
    input_shape = (None, None, input_channels)  # Should handle any size image
    image_input = tf.keras.layers.Input(shape=input_shape)
    if is_training:
      beta_input = tf.keras.layers.Input(shape=(1, 1, 1))
      inputs = (image_input, beta_input)
      output = efficientnet(inputs, self.config)
    else:
      inputs = [image_input]
      output = efficientnet(inputs, self.config)

    # Cast to float32 in case we have a different model dtype
    output = tf.cast(output, tf.float32)

    super(EfficientNet, self).__init__(
        inputs=inputs, outputs=output, name=model_name)

  @classmethod
  def from_name(cls,
                model_name: Text,
                model_weights_path: Text = None,
                weights_format: Text = 'saved_model',
                overrides: Dict[Text, Any] = None):
    """Construct an EfficientNet model from a predefined model name.

    E.g., `EfficientNet.from_name('efficientnet-b0')`.

    Args:
      model_name: the predefined model name
      model_weights_path: the path to the weights (h5 file or saved model dir)
      weights_format: the model weights format. One of 'saved_model', 'h5',
       or 'checkpoint'.
      overrides: (optional) a dict containing keys that can override config

    Returns:
      A constructed EfficientNet instance.
    """
    model_configs = dict(MODEL_CONFIGS)
    overrides = dict(overrides) if overrides else {}

    # One can define their own custom models if necessary
    model_configs.update(overrides.pop('model_config', {}))

    if model_name not in model_configs:
      raise ValueError('Unknown model name {}'.format(model_name))

    config = model_configs[model_name]

    model = cls(config=config, overrides=overrides)

    if model_weights_path:
      load_weights(model, model_weights_path, weights_format=weights_format)

    return model
