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
"""Common modeling utilities."""
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf
from typing import Text, Optional

__all__ = ['count_params', 'load_weights', 'round_filters', 'round_repeats']


def count_params(model, trainable_only=True):
  """Returns the count of all model parameters, or just trainable ones."""
  if not trainable_only:
    return model.count_params()
  else:
    return int(np.sum([tf.keras.backend.count_params(p)
                       for p in model.trainable_weights]))


def load_weights(model: tf.keras.Model,
                 model_weights_path: Text,
                 weights_format: Text = 'saved_model'):
  """Load model weights from the given file path.

  Args:
    model: the model to load weights into
    model_weights_path: the path of the model weights
    weights_format: the model weights format. One of 'saved_model', 'h5',
       or 'checkpoint'.
  """
  if weights_format == 'saved_model':
    loaded_model = tf.keras.models.load_model(model_weights_path)
    model.set_weights(loaded_model.get_weights())
  else:
    model.load_weights(model_weights_path)

def round_filters(filters: int,
                  config: dict) -> int:
  """Round number of filters based on width coefficient."""
  width_coefficient = config['width_coefficient']
  min_depth = config['min_depth']
  divisor = config['depth_divisor']
  orig_filters = filters

  if not width_coefficient:
    return filters

  filters *= width_coefficient
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  return int(new_filters)


def round_repeats(repeats: int, depth_coefficient: float) -> int:
  """Round number of repeats based on depth coefficient."""
  return int(math.ceil(depth_coefficient * repeats))
