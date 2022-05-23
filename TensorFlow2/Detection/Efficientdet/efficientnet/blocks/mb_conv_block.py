import tensorflow as tf
from typing import Any, Dict, Optional, Text, Tuple

from efficientnet.layers import get_activation
from efficientnet.blocks import conv2d_block

__all__ = ['mb_conv_block']

def mb_conv_block(inputs: tf.Tensor,
                  block: dict,
                  config: dict,
                  prefix: Text = None):
  """Mobile Inverted Residual Bottleneck.

  Args:
    inputs: the Keras input to the block
    block: BlockConfig, arguments to create a Block
    config: ModelConfig, a set of model parameters
    prefix: prefix for naming all layers

  Returns:
    the output of the block
  """
  use_se = config['use_se']
  activation = get_activation(config['activation'])
  drop_connect_rate = config['drop_connect_rate']
  data_format = tf.keras.backend.image_data_format()
  use_depthwise = block['conv_type'] != 'no_depthwise'
  prefix = prefix or ''

  filters = block['input_filters'] * block['expand_ratio']

  x = inputs

  if block['fused_conv']:
    # If we use fused mbconv, skip expansion and use regular conv.
    x = conv2d_block(x,
                     filters,
                     config,
                     kernel_size=block['kernel_size'],
                     strides=block['strides'],
                     activation=activation,
                     name=prefix + 'fused')
  else:
    if block['expand_ratio'] != 1:
      # Expansion phase
      kernel_size = (1, 1) if use_depthwise else (3, 3)
      x = conv2d_block(x,
                       filters,
                       config,
                       kernel_size=kernel_size,
                       activation=activation,
                       name=prefix + 'expand')

    # Depthwise Convolution
    if use_depthwise:
      x = conv2d_block(x,
                       conv_filters=None,
                       config=config,
                       kernel_size=block['kernel_size'],
                       strides=block['strides'],
                       activation=activation,
                       depthwise=True,
                       name=prefix + 'depthwise')

  # Squeeze and Excitation phase
  if use_se:
    assert block['se_ratio'] is not None
    assert 0 < block['se_ratio'] <= 1
    num_reduced_filters = max(1, int(
        block['input_filters'] * block['se_ratio']
    ))

    if data_format == 'channels_first':
      se_shape = (filters, 1, 1)
    else:
      se_shape = (1, 1, filters)

    se = tf.keras.layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)
    se = tf.keras.layers.Reshape(se_shape, name=prefix + 'se_reshape')(se)

    se = conv2d_block(se,
                      num_reduced_filters,
                      config,
                      use_bias=True,
                      use_batch_norm=False,
                      activation=activation,
                      name=prefix + 'se_reduce')
    se = conv2d_block(se,
                      filters,
                      config,
                      use_bias=True,
                      use_batch_norm=False,
                      activation='sigmoid',
                      name=prefix + 'se_expand')
    x = tf.keras.layers.multiply([x, se], name=prefix + 'se_excite')

  # Output phase
  x = conv2d_block(x,
                   block['output_filters'],
                   config,
                   activation=None,
                   name=prefix + 'project')

  # Add identity so that quantization-aware training can insert quantization
  # ops correctly.
  x = tf.keras.layers.Activation(get_activation('identity'),
                                 name=prefix + 'id')(x)

  if (block['id_skip']
      and all(s == 1 for s in block['strides'])
      and block['input_filters'] == block['output_filters']):
    if drop_connect_rate and drop_connect_rate > 0:
      # Apply dropconnect
      # The only difference between dropout and dropconnect in TF is scaling by
      # drop_connect_rate during training. See:
      # https://github.com/keras-team/keras/pull/9898#issuecomment-380577612
      x = tf.keras.layers.Dropout(drop_connect_rate,
                                  noise_shape=(None, 1, 1, 1),
                                  name=prefix + 'drop')(x)

    x = tf.keras.layers.add([x, inputs], name=prefix + 'add')

  return x