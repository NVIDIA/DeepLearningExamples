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

from __future__ import print_function
import unittest
import argparse
import os
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import sys
sys.path.append("./tensorflow")
from encoder_sample import encoder_sample

class TestEncoder(unittest.TestCase):
    
    common_args_dict = {'batch_size' : 4,
                        'num_layer' : 12,
                        'max_seq_len': 32,
                        'head_number': 12,
                        'size_per_head': 64,
                        'int8_mode': 0,
                        'allow_gemm_test': 'False',
                        'test_time': 0,
                        'data_type': 'fp32',
                        'remove_padding': 'False',
                        'avg_seq_len': -1
                        }
    threshold = {'fp32': 3e-5, 'fp16': 3e-2 }

    def test_batch_fp32(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp32'
        
        for batch in [1, 8, 64, 128]:
            args_dict['batch_size'] = batch
            max_diff = encoder_sample(args_dict)
            tf.reset_default_graph()
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_batch_fp16(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp16'
        
        for batch in [1, 8, 64, 128]:
            args_dict['batch_size'] = batch
            max_diff = encoder_sample(args_dict)
            tf.reset_default_graph()
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_hidden_fp32(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp32'
        
        for p in [tuple([12, 64]), tuple([16, 64]), tuple([4, 32]), tuple([8, 96])]:
            args_dict['head_number'] = p[0]
            args_dict['size_per_head'] = p[1]
            max_diff = encoder_sample(args_dict)
            tf.reset_default_graph()
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_hidden_fp16(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp16'
        
        for p in [tuple([12, 64]), tuple([16, 64]), tuple([4, 32]), tuple([8, 96])]:
            args_dict['head_number'] = p[0]
            args_dict['size_per_head'] = p[1]
            max_diff = encoder_sample(args_dict)
            tf.reset_default_graph()
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_seqlen_fp32(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp32'
        
        for seqlen in [32, 128, 512]:
            args_dict['max_seq_len'] = seqlen
            max_diff = encoder_sample(args_dict)
            tf.reset_default_graph()
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_seqlen_fp16(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp16'
        
        for seqlen in [32, 128, 512]:
            args_dict['max_seq_len'] = seqlen
            max_diff = encoder_sample(args_dict)
            tf.reset_default_graph()
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_remove_padding_fp32(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp32'
        
        for avg_seq_len in [32, 64, 128]:
            args_dict['max_seq_len'] = 256
            args_dict['remove_padding'] = 'True'
            args_dict['avg_seq_len'] = avg_seq_len

            max_diff = encoder_sample(args_dict)
            tf.reset_default_graph()
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

    def test_remove_padding_fp16(self):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['data_type'] = 'fp16'
        
        for avg_seq_len in [32, 64, 128]:
            args_dict['max_seq_len'] = 256
            args_dict['remove_padding'] = 'True'
            args_dict['avg_seq_len'] = avg_seq_len

            max_diff = encoder_sample(args_dict)
            tf.reset_default_graph()
            self.assertTrue(max_diff < self.threshold[args_dict['data_type']])

if __name__ == "__main__":
    unittest.main()
