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
import ctypes
from utils.beam_search import BeamSearch
from utils.beam_search import DiverseSiblingSearch

class TransformerArgument:
  def __init__( self,
                beam_width,
                head_num,
                size_per_head,
                num_layer,
                dtype=tf.float32,
                kernel_init_range=0.02,
                bias_init_range=0.02,
                fuse_qkv=True,
                remove_padding=False):
    '''
    The arguments of Transformer layer (for both encoder and decoder).
    
    Args:
        beam_width: The beam_width size for beam search. This argument is always one for encoder.
        head_num: The head number of self attention in transformer layer.
        size_per_head: The size of hidden dimension for each head of self attention in transformer layer. 
        num_layer: The number of transformer layer. For example, BERT-base uses 12 layers.
        dtype: The data type of weights initializer and inputs. 
        kernel_init_range: The initializer range of kernel for all convolution layer and fully-connected layer. 
        kernel_init_range: The initializer range of bias for all convolution layer and fully-connected layer. 
        fuse_qkv: bool. Wether fuse the q, k, v gemm or not.
        remove_padding: bool. Remove the padding of sentences of encoder. 
    '''
    
    self.beam_width = beam_width
    self.head_num = head_num
    self.size_per_head = size_per_head
    self.num_layer = num_layer
    self.dtype = dtype
    self.hidden_dim = self.head_num * self.size_per_head
    self.kernel_init_range = kernel_init_range
    self.bias_init_range = bias_init_range
    if self.dtype == tf.float32:
      self.check_threshold = 2e-5
    elif self.dtype == tf.float16:
      self.check_threshold = 2e-2
    self.fuse_qkv = fuse_qkv
    self.remove_padding = remove_padding

class DecodingArgument(object):
  def __init__( self,
                vocab_size,
                start_id,
                end_id,
                max_seq_len,
                decoder_args):
    '''
    The arguments of Decoding.
    Decoding is the function which contains the whole translation process.
    For example, the embedding lookup, position encoding, decoder, and
      beam search or sampling to choose the token.
    
    Args:
        vocab_size: The size of vocabulary of Decoding. 
        start_id: The id of start token in vocabulary.
        end_id: The id of end token in vocabulary.
        max_seq_len: The maximum length of sentence in translation. 
        decoder_args: The arguments of decoder layer.
    '''
    
    self.vocab_size = vocab_size
    self.start_id = start_id
    self.end_id = end_id
    self.max_seq_len = max_seq_len
    self.decoder_args = decoder_args
      
class DecodingBeamsearchArgument(DecodingArgument):
  def __init__( self,
                vocab_size,
                start_id,
                end_id,
                max_seq_len,
                decoder_args,
                beam_search_diversity_rate=-0.0):
    '''
    The arguments of Decoding with beam search.
    Most arguments are similar to DecodingArgument except the beam_search_diversity_rate.
    
    Args:
        vocab_size: The size of vocabulary of Decoding. 
        start_id: The id of start token in vocabulary.
        end_id: The id of end token in vocabulary.
        max_seq_len: The maximum length of sentence in translation. 
        decoder_args: The arguments of decoder layer.
        beam_search_diversity_rate: The diversity rate of beam search. When it is 0, 
          it is equivalent to naive beam search. 
    '''
    
    super(DecodingBeamsearchArgument, self).__init__(vocab_size,
                                                    start_id,
                                                    end_id,
                                                    max_seq_len,
                                                    decoder_args)
    
    self.beam_search_diversity_rate = beam_search_diversity_rate
    if abs(self.beam_search_diversity_rate) == 0.0:
      self.search_method = BeamSearch()
    else:
      self.search_method = DiverseSiblingSearch(beam_search_diversity_rate)

class DecodingSamplingArgument(DecodingArgument):
  def __init__( self,
                vocab_size,
                start_id,
                end_id,
                max_seq_len,
                decoder_args,
                top_k=0, 
                top_p=0.0):
    '''
    The arguments of Decoding with sampling.
    Most arguments are similar to DecodingArgument except the top_k and top_p.
    
    Args:
        vocab_size: The size of vocabulary of Decoding. 
        start_id: The id of start token in vocabulary.
        end_id: The id of end token in vocabulary.
        max_seq_len: The maximum length of sentence in translation. 
        decoder_args: The arguments of decoder layer.
        top_k: A int value. The value of k for top k sampling.
        top_p: A float value. The value of p for top p sampling. 
        
    Note that top_k and top_p both are 0 in the same time is invalid. 
    Note that top_k and top_p both are non-zero in the same time is invalid. 
    If top_k is non-zero, the Decoding function will use the top k sampling. 
    If top_k is non-zero, the Decoding function will use the top p sampling.
    '''
    
    super(DecodingSamplingArgument, self).__init__(vocab_size,
                                                  start_id,
                                                  end_id,
                                                  max_seq_len,
                                                  decoder_args)

    self.top_k = top_k
    self.top_p = top_p
    if self.top_k == 0 and self.top_p == 0.0:
      print("[ERROR] top_k and top_p cannot both be 0.")
      exit(-1)
    elif self.top_k != 0 and self.top_p != 0.0:
      print("[ERROR] top_k and top_p cannot both be non-zero.")
      exit(-1)

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
        
class cudaProfiler:
  
  def __init__(self):
    self.profiler = ctypes.CDLL("libcudart.so")
    
  def start(self):
    ret = self.profiler.cudaProfilerStart()
    if ret != 0:
      raise Exception("cudaProfilerStart() return %d " %ret)
    
  def stop(self):
    ret = self.profiler.cudaProfilerStop()
    if ret != 0:
      raise Exception("cudaProfilerStop() return %d " %ret)
    