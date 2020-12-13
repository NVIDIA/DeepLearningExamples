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

    def test_attn_batch(self):
        for b in [1, 4, 32, 128]:
            self.run_attn(b, 128, 12, 64)
    
    def test_attn_seq(self):
        for seq in [64, 96, 128, 384]:
            self.run_attn(4, seq, 12, 64)
    
    def test_attn_head(self):
        for head in [8, 12, 16]:
            self.run_attn(4, 128, head, 64)

    def run_attn(self, batch_size, seq_len, head_num, size_per_head):
        hidden_dim = head_num * size_per_head
        np.random.seed(1)
        tf.set_random_seed(1)

        sequence_length = np.random.randint(1, seq_len + 1, size=batch_size).astype(np.int32)
        attention_mask = build_sequence_mask(sequence_length, num_heads=head_num, maximum_length=seq_len, dtype=tf.float16)

        q_input = np.random.rand(batch_size, seq_len, hidden_dim).astype(np.float16)
        k_input = np.random.rand(batch_size, seq_len, hidden_dim).astype(np.float16)
        v_input = np.random.rand(batch_size, seq_len, hidden_dim).astype(np.float16)

        query_layer = tf.convert_to_tensor(q_input, dtype=tf.float16)
        key_layer = tf.convert_to_tensor(k_input, dtype=tf.float16)
        value_layer = tf.convert_to_tensor(v_input, dtype=tf.float16)
    
        def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
            output_tensor = tf.reshape(
                input_tensor, [batch_size, seq_length, num_attention_heads, width])

            output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
            return output_tensor
            
        # `query_layer` = [B, N, F, H]
        query_layer = transpose_for_scores(query_layer, batch_size,head_num, seq_len,size_per_head)

        # `key_layer` = [B, N, T, H]
        key_layer = transpose_for_scores(key_layer, batch_size, head_num,seq_len, size_per_head)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            if tf.rank(attention_mask) == 3:
                attention_mask = tf.expand_dims(attention_mask, axis=[1])
                
            adder = (1.0 - tf.cast(attention_mask, tf.float16)) * -10000.0

            attention_scores += adder

        attention_probs = tf.nn.softmax(attention_scores)

        value_layer = tf.reshape(
            value_layer,
            [batch_size, seq_len, head_num, size_per_head])

        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, [batch_size, seq_len, head_num * size_per_head])
        
        # remove the padding of tf output to compare with op output
        tf_output = []
        for i in range(batch_size):
            tf_output.append(context_layer[i][0:sequence_length[i]])
        tf_output = tf.concat(tf_output, axis=0)

        # Attention op
        fused_multihead_attention_op = tf.load_op_library(os.path.join('./lib/libtf_fused_multihead_attention.so'))

        # remove padding of q, k and v
        q_input_remove_pad = []
        for i in range(batch_size):
            q_input_remove_pad.append(q_input[i][0:sequence_length[i]])
        q_input_remove_pad = tf.concat(q_input_remove_pad, axis=0)

        k_input_remove_pad = []
        for i in range(batch_size):
            k_input_remove_pad.append(k_input[i][0:sequence_length[i]])
        k_input_remove_pad = tf.concat(k_input_remove_pad, axis=0)

        v_input_remove_pad = []
        for i in range(batch_size):
            v_input_remove_pad.append(v_input[i][0:sequence_length[i]])
        v_input_remove_pad = tf.concat(v_input_remove_pad, axis=0)
        
        qkv_input = tf.concat([q_input_remove_pad, k_input_remove_pad, v_input_remove_pad], axis=1)
        qkv_input = tf.reshape(qkv_input, [-1, 3, head_num, size_per_head])
        qkv_input = tf.transpose(qkv_input, [0, 2, 1, 3])
        qkv_input = tf.reshape(qkv_input, [-1, 3 * head_num * size_per_head])
        
        op_output = fused_multihead_attention_op.multi_head_attention(qkv_input, 
                                                                      np.cumsum(np.insert(sequence_length, 0, 0), axis=0),
                                                                      head_num,
                                                                      size_per_head,
                                                                      seq_len)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            print(batch_size, seq_len, head_num, size_per_head)

            sess.run(tf.global_variables_initializer())
            tf_result = sess.run(tf_output).flatten()
            op_result = sess.run(op_output).flatten()

            assert(abs(tf_result - op_result).max() < 0.03)

if __name__ == "__main__":
    unittest.main()
