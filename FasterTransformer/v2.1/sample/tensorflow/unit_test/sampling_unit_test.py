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
from utils.sampling import Sampling

sample_number = 100000

np.random.seed(1)
tf.set_random_seed(1)

def top_k_sampling_unit_test():
    top_k_sampling = Sampling("top_k")
    
    probs_1 = np.asarray([ [4.0, 3.0, 2.0, 1.0] ])
    np_result_1 = [0]
    k_1 = 1
    tf_result_1 = top_k_sampling.sample(tf.convert_to_tensor(probs_1), k_1)
    
    probs_2 = np.asarray(np.log([ [0.6, 0.4, 0.3, 0.1] ]))
    np_result_2 = [0.6, 0.4, 0.0, 0.0]
    k_2 = 2
    tf_result_2 = top_k_sampling.sample(tf.convert_to_tensor(probs_2), k_2, sample_number)
    
    np_probs_3 = [0.3, 0.4, 0.25, 0.01, 0.05]
    probs_3 = np.asarray(np.log([np_probs_3]))
    np_result_3 = [0.3, 0.4, 0.25, 0.0, 0.05]
    k_3 = 4
    tf_result_3 = top_k_sampling.sample(tf.convert_to_tensor(probs_3), k_3, sample_number)
    
    np_probs_4 = [0.3, 0.4, 0.25, 0.01, 0.05]
    probs_4 = np.asarray(np.log([np_probs_4]))
    np_result_4 = [0.3/0.7, 0.4/0.7, 0.0, 0.00, 0.00]
    k_4 = 2
    tf_result_4 = top_k_sampling.sample(tf.convert_to_tensor(probs_4), k_4, sample_number)
    
    np_probs_5 = np.random.randn(1, 10000) * 1
    np_probs_5 = np.abs(np_probs_5)
    np_probs_5[0][0] *= 5
    np_result_5 = np_probs_5
    k_5 = 10
    np_sorted_result_5 = np.sort(np_probs_5)
    threshold = np_sorted_result_5[:,-k_5]
    mask = np_probs_5 >= threshold
    np_result_5 = np_probs_5 * mask
    np_result_5[0] = np_result_5[0] / np.sum(np_result_5[0])
    tf_result_5 = top_k_sampling.sample(tf.convert_to_tensor(np.log(np_probs_5)), k_5, sample_number)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True    
    print("[INFO] start the top k sampling unit test.")
    with tf.Session(config=config) as sess:
        tf_result_1 = sess.run(tf_result_1)
        tf_result_1 = np.asarray(tf_result_1).astype(np.int32)
        print("[INFO] case_1.")
        for i, j in zip(tf_result_1, np_result_1):
            assert(i == j)
        
        print("[INFO] case_2.")
        tf_result_2 = sess.run(tf_result_2)
        p0 = 0
        p1 = 0
        for i in tf_result_2:
            if i == 0:
                p0 += 1
            elif i == 1:
                p1 += 1
        print(p0/sample_number, p1/sample_number)
        print(np_result_2)
        
        print("[INFO] case_3.")
        tf_result_3 = sess.run(tf_result_3)
        p = np.zeros_like(np_result_3)
        for i in tf_result_3:
            p[i] += 1
        print(p * 1.0 / sample_number)
        print(np_result_3)
        
        print("[INFO] case_4.")
        tf_result_4 = sess.run(tf_result_4)
        p = np.zeros_like(np_result_4)
        for i in tf_result_4:
            p[i] += 1
        print(p * 1.0 / sample_number)
        print(np_result_4)
        
        print("[INFO] case_5.")
        tf_result_5 = sess.run(tf_result_5)
        p = np.zeros_like(np_result_5)
        print(tf_result_5)
        for i in tf_result_5:
            p[0][i] += 1
        for i, j in zip(p[0]/sample_number, np_result_5[0]):
            if i != 0 or j != 0:
                print(i, j)
    
        

def top_p_sampling_unit_test():
    top_p_sampling = Sampling("top_p")

    np_probs_1 = [0.3, 0.01, 0.4, 0.25, 0.05]
    np_result_1 = [0, 0, 1, 0, 0]
    p_1 = 0.3
    tf_result_1 = top_p_sampling.sample(tf.convert_to_tensor(np.log([np_probs_1])), p_1, sample_number)
    
    np_probs_2 = [0.3, 0.01, 0.4, 0.25, 0.05]
    np_result_2 = [3./7, 0, 4./7, 0, 0]
    p_2 = 0.5
    tf_result_2 = top_p_sampling.sample(tf.convert_to_tensor(np.log([np_probs_2])), p_2, sample_number)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True    
    print("[INFO] start the top p sampling unit test.")
    with tf.Session(config=config) as sess:
        
        print("[INFO] case_1.")
        tf_result_1 = sess.run(tf_result_1)
        p = np.zeros_like(np_result_1)
        for i in tf_result_1:
            for j in range(len(p)):
                if i == j:
                    p[j] += 1
        print(p * 1.0 / sample_number)
        print(np_result_1)
        
        print("[INFO] case_2.")
        tf_result_2 = sess.run(tf_result_2)
        p = np.zeros_like(np_result_2)
        for i in tf_result_2:
            for j in range(len(p)):
                if i == j:
                    p[j] += 1
        print(p * 1.0 / sample_number)
        print(np_result_2)
    
if __name__ == "__main__":
    top_k_sampling_unit_test()
    top_p_sampling_unit_test()
    

