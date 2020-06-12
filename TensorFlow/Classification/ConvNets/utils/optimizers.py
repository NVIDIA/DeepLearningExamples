#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import print_function

import tensorflow as tf

__all__ = ['FixedLossScalerOptimizer']


class FixedLossScalerOptimizer(tf.train.Optimizer):
    """An optimizer that scales loss and un-scales gradients for FP16 training."""

    def __init__(self, optimizer, scale=None, name="LossScalingOptimizer", use_locking=False):

        super(FixedLossScalerOptimizer, self).__init__(name=name, use_locking=use_locking)

        self._optimizer = optimizer
        self._scale = float(scale) if scale is not None else 1.0

    def compute_gradients(self, loss, var_list=None, *args, **kwargs):

        if self._scale != 1.0:
            loss = tf.scalar_mul(self._scale, loss)

        gradvar = self._optimizer.compute_gradients(loss, var_list, *args, **kwargs)
        gradvar = [(tf.scalar_mul(1. / self._scale, g), v) for g, v in gradvar]

        return gradvar

    def apply_gradients(self, *args, **kwargs):
        return self._optimizer.apply_gradients(*args, **kwargs)
