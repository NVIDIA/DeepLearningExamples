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

def search_word(beam_width,
                vocab_size,
                step,
                logits,
                cum_log_probs,
                finished,
                cache,
                extra_vars,
                op_self_cache=None,
                search_method=None):

    # [batch_size * beam_width, vocab_size]
    batchxbeam = tf.shape(logits)[0] 
    log_probs = tf.nn.log_softmax(logits)
            
    parent_ids = extra_vars[0]
    sequence_lengths = extra_vars[1]
    total_probs = log_probs + tf.expand_dims(cum_log_probs, 1)
    # [batch_size * beam_width, vocab_size] + [batch_size * beam_width], has to broadcast
    total_probs = tf.reshape(total_probs, [-1, beam_width * vocab_size])

    if search_method == None:
        search_method = BeamSearch()
    sample_ids = search_method.process(total_probs, beam_width, vocab_size)
    
    word_ids = sample_ids % vocab_size  # [batch_size * beam_width]
    beam_ids = sample_ids // vocab_size  # [batch_size * beam_width]
    # [batch_size * beam_width]
    beam_indices = (tf.range(batchxbeam) // beam_width) * beam_width + beam_ids

    sequence_lengths = tf.where(
        finished, x=sequence_lengths, y=sequence_lengths + 1)

    # [batch_size * beam_width]
    batch_pos = tf.range(batchxbeam) // beam_width
    cum_log_probs = tf.gather_nd(total_probs, tf.stack(
        [batch_pos, sample_ids], axis=-1))  # [batch_size * beam_width]
    finished = tf.gather(finished, beam_indices)
    sequence_lengths = tf.gather(sequence_lengths, beam_indices)

    cache = tf.contrib.framework.nest.map_structure(
        lambda s: tf.gather(s, beam_indices), cache)
    if op_self_cache != None:
        op_self_cache = tf.contrib.framework.nest.map_structure(
            lambda s: tf.gather(s, beam_indices, axis=3), op_self_cache)

    parent_ids = parent_ids.write(step, beam_ids)
    extra_vars = [parent_ids, sequence_lengths]

    return word_ids, cum_log_probs, finished, cache, tuple(extra_vars), op_self_cache

class Search():
    
    def __init__(self):
        pass
    
    def process(self, total_probs, beam_width, vocab_size):
        pass

class BeamSearch(Search):
    
    def __init__(self):
        pass

    def process(self, total_probs, beam_width, vocab_size):
        '''
        inputs:
            total_probs: float tensor, [batch_size * beam_width, vocab_size]
            beam_width: int scalar
        
        outputs: 
            sample_ids: int tensor, [batch_size * beam_width]
        '''
        
        # [batch_size, beam_width * vocab_size], can skip in cuda
        total_probs = tf.reshape(total_probs, [-1, beam_width * vocab_size])

        _, sample_ids = tf.nn.top_k(total_probs, beam_width)
        # [batch_size * beam_width], can skip in cuda
        sample_ids = tf.reshape(sample_ids, [-1])
        
        return sample_ids

class DiverseSiblingSearch(Search):
    
    def __init__(self, diversity_rate):
        '''
        inputs:
            diversity: int scalar, >= 0
            if diversity_rate == 0, then it is equivalent to beam_search
        '''
        self.diversity_rate = diversity_rate
    
    def process(self, total_probs, beam_width, vocab_size):
        '''
        inputs:
            total_probs: float tensor, [batch_size * beam_width, vocab_size]
        
        outputs:
            sample_ids: int tensor, [batch_size * beam_width]
            beam_ids: int tensor, [batch_size * beam_width]
        
        1. calculate hypothese for each beam
        2. Intra-sibling ordering
        3. rewrite scores
        4. choose top K hypothese
        '''
        
        total_probs = tf.reshape(total_probs, [-1, beam_width, vocab_size]) # [batch size, beam width, vocab size]

        sibling_score = tf.cast(tf.range(1, beam_width+1), total_probs.dtype) * self.diversity_rate # [beam_width]
        
        scores, ids = tf.nn.top_k(total_probs, beam_width) # [batch size, beam width, beam width]
        scores = tf.add(scores, sibling_score) # [batch size, beam width, beam width]
        
        scores = tf.reshape(scores, [-1, beam_width * beam_width])
        ids = ids + tf.expand_dims(tf.expand_dims(tf.range(0, beam_width) * vocab_size, 0), -1)
        ids = tf.reshape(ids, [-1, beam_width * beam_width])
        
        _, final_ids = tf.nn.top_k(scores, beam_width) # [batch size, beam width]
        
        batch_size = tf.shape(final_ids)[0]
        final_ids = tf.reshape(final_ids, [-1, 1])
        batch_index = tf.range(0, batch_size) 
        batch_index = tf.reshape(batch_index, [-1, 1])
        batch_index = tf.tile(batch_index, [1, beam_width])
        batch_index = tf.reshape(batch_index, [-1, 1])

        index = tf.concat([batch_index, final_ids ], axis=1)
        sample_ids = tf.gather_nd(ids, index)
         
        return sample_ids 
        

