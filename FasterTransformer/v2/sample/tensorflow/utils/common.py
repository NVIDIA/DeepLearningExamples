# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from datetime import datetime
import tensorflow as tf
import numpy as np

class TransformerArgument:
  def __init__( self,
                batch_size,
                beam_width,
                head_num,
                size_per_head,
                num_layer,
                max_seq_len,
                dtype):
    
    self.batch_size = batch_size
    self.beam_width = beam_width
    self.head_num = head_num
    self.size_per_head = size_per_head
    self.num_layer = num_layer
    self.max_seq_len = max_seq_len
    self.dtype = dtype
    self.hidden_dim = self.head_num * self.size_per_head

class DecodingArgument:
  def __init__( self,
                batch_size,
                beam_width,
                head_num,
                size_per_head,
                num_layer,
                max_seq_len,
                vocab_size,
                start_id,
                end_id,
                encoder_hidden_dim,
                dtype):
    
    self.decoder_args = TransformerArgument(batch_size,
                                            beam_width,
                                            head_num,
                                            size_per_head,
                                            num_layer,
                                            max_seq_len,
                                            dtype)
    self.vocab_size = vocab_size
    self.start_id = start_id
    self.end_id = end_id
    self.encoder_hidden_dim = encoder_hidden_dim

def create_initializer(initializer_range=0.02, data_type=tf.float32):
  return tf.truncated_normal_initializer(stddev=initializer_range, dtype=data_type)

def _get_shape_invariants(tensor):
      """Returns the shape of the tensor but sets middle dims to None."""
      if isinstance(tensor, tf.TensorArray):
        shape = None
      else:
        shape = tensor.shape.as_list()
        for i in range(1, len(shape) - 1):
          shape[i] = None
      return tf.TensorShape(shape)

def time_test(sess, tensor, iterations=100, warmup=True):
    # return in ms

    # warmup
    if warmup == True:
      for i in range(iterations):
          sess.run(tensor)
        
    t1 = datetime.now()
    for i in range(iterations):
      sess.run(tensor)
    t2 = datetime.now()
    time_sum = (t2 - t1).total_seconds()
    return time_sum * 1000 / iterations

def cross_check(name, tf_val, op_val, atol_threshold):
  abs_diff = np.fabs(tf_val - op_val)
  print("[INFO] {} Cross check {}".format(name, np.allclose(tf_val, op_val, atol=atol_threshold)))
  print("[INFO] Max diff {}".format(abs_diff.max()))
  print("[INFO] min diff {}".format(abs_diff.min()))
  
def int_result_cross_check(name, tf_result, op_result, shape):
  print(" ")
  is_same = (tf_result.flatten() == op_result.flatten()).all()
  print("       {} cross-check: {}".format(name, is_same))
  if is_same == False:
    tf_reshaped_result = np.reshape(tf_result, shape)
    op_reshaped_result = np.reshape(op_result, shape)
    
    for i in range(tf_reshaped_result.shape[0]):
      is_true = (tf_reshaped_result[i] == op_reshaped_result[i]).all()
      print("       Cross-Check on step-{} {}".format(i, is_true))
      if is_true == False:
        print("TF result: {}".format(tf_reshaped_result[i]))
        print("OP result: {}".format(op_reshaped_result[i]))