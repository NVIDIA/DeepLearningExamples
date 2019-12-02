# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

import tensorflow as tf
from tensorflow.keras.layers import Dense


class DenseFromSparse(Dense):
    def call(self, inputs):
        if type(inputs) != tf.sparse.SparseTensor:
            raise ValueError("input should be of type " + str(tf.sparse.SparseTensor))
        rank = len(inputs.get_shape().as_list())
        if rank != 2:
            raise NotImplementedError("input should be rank 2")
        else:
            outputs = tf.sparse.sparse_dense_matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs
