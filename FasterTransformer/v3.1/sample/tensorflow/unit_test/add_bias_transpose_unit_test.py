from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import sys
import os
import math
sys.path.append("./tensorflow/")
from utils.encoder import build_sequence_mask
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class TestMutiheadAttention(unittest.TestCase):
    count = 0

    def test_attn(self):
        for valid_word_num in [10, 32, 100, 156, 400]:
            for head in [12, 16]:
                self.run_attn(200, 12, 64)

    def run_attn(self, valid_word_num, head_num, size_per_head):
        hidden_dim = head_num * size_per_head
        np.random.seed(1)
        tf.set_random_seed(1)

        q_input = np.random.rand(valid_word_num, hidden_dim).astype(np.float16)
        k_input = np.random.rand(valid_word_num, hidden_dim).astype(np.float16)
        v_input = np.random.rand(valid_word_num, hidden_dim).astype(np.float16)

        q_b = np.random.rand(hidden_dim).astype(np.float16)
        k_b = np.random.rand(hidden_dim).astype(np.float16)
        v_b = np.random.rand(hidden_dim).astype(np.float16)

        qkv = tf.concat([q_input + q_b, k_input + k_b, v_input + v_b], axis=0) # [3 * valid_word_num, 768]
        qkv_s = tf.reshape(qkv, [3, -1, head_num, size_per_head])
        qkv_t = tf.transpose(qkv_s, [1, 2, 0, 3])
        tf_output = tf.reshape(qkv_t, [-1, 3 * hidden_dim])

        fused_multihead_attention_op = tf.load_op_library(os.path.join('./lib/libtf_add_bias_transpose.so'))
        op_output = fused_multihead_attention_op.add_bias_transpose(q_input, q_b, 
                                                                    k_input, k_b,
                                                                    v_input, v_b,
                                                                    head_num, size_per_head)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())
            tf_result = sess.run(tf_output).flatten()
            op_result = sess.run(op_output).flatten()

            assert(abs(tf_result - op_result).max() < 0.03)

if __name__ == "__main__":
    unittest.main()
