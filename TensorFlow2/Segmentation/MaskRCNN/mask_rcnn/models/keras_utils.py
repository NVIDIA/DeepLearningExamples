#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import tensorflow as tf


__all__ = ["KerasMockLayer"]


class KerasMockLayer(tf.Module):
    """
    This class reproduces the Keras Layer important APIs without enforcing a variable scope.
    """
    def __init__(self, trainable=True, *args, **kwargs):
        super(KerasMockLayer, self).__init__(*args, **kwargs)
        self._local_layers = dict()
        self._trainable = trainable

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        for layer in getattr(self, '_layers', []):
            layer.trainable = value

    @property
    def variables(self):
        """Returns the list of all layer variables/weights.
        Alias of `self.weights`.
        Returns:
          A list of variables.
        """
        return self.weights

    @property
    def trainable_variables(self):
        return self.trainable_weights

    @property
    def non_trainable_variables(self):
        return self.non_trainable_weights

    @property
    def weights(self):
        """Returns the list of all layer variables/weights.
        Returns:
          A list of variables.
        """
        return self.trainable_weights + self.non_trainable_weights

    @property
    def name(self):
        return self._name

    @property
    def trainable_weights(self):
        layers = list()

        for layer in self._local_layers.values():
            if not isinstance(layer, dict):
                layers.append(layer)
            else:
                for sublayer in layer.values():
                    layers.append(sublayer)

        return list(itertools.chain.from_iterable([layer.trainable_variables for layer in layers]))

    @property
    def non_trainable_weights(self):
        layers = list()

        for layer in self._local_layers.values():
            if not isinstance(layer, dict):
                layers.append(layer)
            else:
                for sublayer in layer.values():
                    layers.append(sublayer)

        return list(itertools.chain.from_iterable([layer.non_trainable_weights for layer in layers]))
