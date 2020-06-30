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

import tensorflow as tf

class Sampling():
    
    def __init__(self, sample_method):
        if sample_method == "top_k":
            self.sample_method = self.top_k_logits
        elif sample_method == "top_p":
            self.sample_method = self.top_p_logits
        else:
            print("[ERROR] the sample method should be one of top_k and top_p")
            exit(-1)
        
        pass
    
    def sample(self, logits, threshold, num_samples=1):
        '''
        inputs:
              logits: [batch_size, vocab_size], the values of log logits
              threshold: int when using top_k, and a probability (0~1) when using top_p
        
        outputs:
              samples: [batch_size]
        '''
        
        logits = self.sample_method(logits, threshold)
        samples = tf.multinomial(logits, num_samples=num_samples, output_dtype=tf.int32)
        samples = tf.reshape(samples, [-1])
        return samples
    
    def top_k_logits(self, logits, k):
        if k == 0:
            return logits
        else:
            values, _ = tf.nn.top_k(logits, k=k) # [batch size, k]
            min_values = values[:, -1, tf.newaxis] #[batch size, 1]
            return tf.where(
                logits < min_values,
                tf.ones_like(logits, dtype=logits.dtype) * logits.dtype.min,
                logits
            )
            
    def top_p_logits(self, logits, p):
        sorted_logits = tf.sort(logits, direction='DESCENDING')
        sorted_probs = tf.nn.softmax(sorted_logits)
        probs_sums = tf.cumsum(sorted_probs, axis=1, exclusive=True)
        logits_masked = tf.where(
            probs_sums < p,
            sorted_logits,
            tf.ones_like(sorted_logits) * 1000
        ) # [batchsize, vocab]
        min_logits = tf.reduce_min(logits_masked, axis=1, keepdims=True) # [batch size, 1]
        return tf.where(
            logits < min_logits,
            tf.ones_like(logits, dtype=logits.dtype) * logits.dtype.min,
            logits
        )
        
        
    