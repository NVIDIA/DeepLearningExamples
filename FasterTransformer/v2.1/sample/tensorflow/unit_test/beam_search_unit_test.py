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
import numpy as np
import os,sys
sys.path.append(os.getcwd())
from utils.beam_search import beam_search, diverse_sibling_search

sample_number = 100000

np.random.seed(1)
tf.set_random_seed(1)

def beam_search_unit_test():
    batch_size = 1
    beam_width = 2
    vocab_size = 4
    
    # case 1                            
    probs_1 = np.asarray([ [4.0, 3.0, 2.0, 1.0],
                           [4.1, 3.1, 2.1, 1.1] ])
    diversity_rate_1 = 0.0
    np_result_1 = np.asarray([4, 0]).astype(np.int32)
    final_id_1 = diverse_sibling_search(probs_1, beam_width, vocab_size, diversity_rate=diversity_rate_1)
    
    # case 2        
    probs_2 = np.asarray([ [4.0, 3.0, 2.0, 1.0],
                           [4.1, 3.1, 2.1, 1.1] ])
    diversity_rate_2 = -1.0
    np_result_2 = np.asarray([4, 0]).astype(np.int32)
    final_id_2 = diverse_sibling_search(probs_2, beam_width, vocab_size, diversity_rate=diversity_rate_2)
    
    # case 3        
    probs_3 = np.asarray([ [4.0, 3.0, 2.0, 1.0],
                           [2.1, 1.1, 0.1, 0.01] ])
    diversity_rate_3 = -1.0
    np_result_3 = np.asarray([0, 4]).astype(np.int32)
    final_id_3 = diverse_sibling_search(probs_3, beam_width, vocab_size, diversity_rate=diversity_rate_3)
    
    # case 4        
    probs_4 = np.asarray([ [4.0, 3.0, 2.0, 1.0],
                           [2.1, 1.1, 0.1, 0.01] ])
    diversity_rate_4 = 0.0
    np_result_4 = np.asarray([0, 1]).astype(np.int32)
    final_id_4 = diverse_sibling_search(probs_4, beam_width, vocab_size, diversity_rate=diversity_rate_4)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True    
    print("[INFO] start the diversing beam search unit test.")
    with tf.Session(config=config) as sess:
        tf_result_1 = sess.run(final_id_1)
        tf_result_1 = np.asarray(tf_result_1).astype(np.int32)
        for i, j in zip(tf_result_1, np_result_1):
            assert(i == j)
        print("[INFO] case_1 pass.")
        
        tf_result_2 = sess.run(final_id_2)
        tf_result_2 = np.asarray(tf_result_2).astype(np.int32)
        for i, j in zip(tf_result_2, np_result_2):
            assert(i == j)
        print("[INFO] case_2 pass.")
        
        tf_result_3 = sess.run(final_id_3)
        tf_result_3 = np.asarray(tf_result_3).astype(np.int32)
        for i, j in zip(tf_result_3, np_result_3):
            assert(i == j)
        print("[INFO] case_3 pass.")
        
        tf_result_4 = sess.run(final_id_4)
        tf_result_4 = np.asarray(tf_result_4).astype(np.int32)
        for i, j in zip(tf_result_4, np_result_4):
            assert(i == j)
        print("[INFO] case_4 pass.")
    
if __name__ == "__main__":
    beam_search_unit_test()
    

