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

import tensorflow as tf

__all__ = ['model_variable_scope']


def model_variable_scope(name, reuse=False, dtype=tf.float32, debug_mode=False, *args, **kwargs):
    """Returns a variable scope that the model should be created under.
    If self.dtype is a castable type, model variable will be created in fp32
    then cast to self.dtype before being used.
    Returns:
      A variable scope for the model.
    """

    def _custom_dtype_getter(getter, name, shape=None, dtype=None, trainable=True, regularizer=None, *args, **kwargs):
        """Creates variables in fp32, then casts to fp16 if necessary.
        This function is a custom getter. A custom getter is a function with the
        same signature as tf.get_variable, except it has an additional getter
        parameter. Custom getters can be passed as the `custom_getter` parameter of
        tf.variable_scope. Then, tf.get_variable will call the custom getter,
        instead of directly getting a variable itself. This can be used to change
        the types of variables that are retrieved with tf.get_variable.
        The `getter` parameter is the underlying variable getter, that would have
        been called if no custom getter was used. Custom getters typically get a
        variable with `getter`, then modify it in some way.
        This custom getter will create an fp32 variable. If a low precision
        (e.g. float16) variable was requested it will then cast the variable to the
        requested dtype. The reason we do not directly create variables in low
        precision dtypes is that applying small gradients to such variables may
        cause the variable not to change.
        Args:
          getter: The underlying variable getter, that has the same signature as
            tf.get_variable and returns a variable.
          name: The name of the variable to get.
          shape: The shape of the variable to get.
          *args: Additional arguments to pass unmodified to getter.
          **kwargs: Additional keyword arguments to pass unmodified to getter.
        Returns:
          A variable which is cast to fp16 if necessary.
        """

        storage_dtype = tf.float32 if dtype in [tf.float32, tf.float16] else dtype

        variable = getter(
            name,
            shape,
            dtype=storage_dtype,
            trainable=trainable,
            regularizer=(
                regularizer if
                (trainable and not any(l_name.lower() in name.lower()
                                       for l_name in ['batchnorm', 'batch_norm'])) else None
            ),
            *args,
            **kwargs
        )

        if dtype != tf.float32:
            cast_name = name + '/fp16_cast'

            try:
                cast_variable = tf.get_default_graph().get_tensor_by_name(cast_name + ':0')

            except KeyError:
                cast_variable = tf.cast(variable, dtype, name=cast_name)

            cast_variable._ref = variable._ref
            variable = cast_variable

        return variable

    return tf.variable_scope(name, reuse=reuse, dtype=dtype, custom_getter=_custom_dtype_getter, *args, **kwargs)
