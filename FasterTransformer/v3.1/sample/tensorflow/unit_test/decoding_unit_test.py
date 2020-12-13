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
from translate_sample import translate_sample

class TestDecoding(unittest.TestCase):
    
    common_args_dict = {'batch_size' : 128,
                        'max_seq_len': 200,
                        'encoder_head_number': 8,
                        'encoder_size_per_head': 64,
                        'decoder_head_number': 8,
                        'decoder_size_per_head': 64,
                        'encoder_num_layer': 6,
                        'decoder_num_layer': 6,
                        'beam_search_diversity_rate': 0.0,
                        'sampling_topk': 1,
                        'sampling_topp': 0.0,
                        'source_vocabulary': "./tensorflow/utils/translation/wmtende.vocab",
                        'target_vocabulary': "./tensorflow/utils/translation/wmtende.vocab",
                        'source': "./tensorflow/utils/translation/test.en",
                        'target': "./tensorflow/utils/translation/test.de"
                        }
    
    def run_translate(self, beam_width, datatype, test_time, topk=4, topp=0.0):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['beam_width'] = beam_width
        args_dict['data_type'] = datatype
        args_dict['test_time'] = test_time
        args_dict['sampling_topk'] = topk
        args_dict['sampling_topp'] = topp
        
        translation_result_list = translate_sample(args_dict)
        tf_bleu_score = translation_result_list[0].bleu_score.score
        op_decoder_bleu_score = translation_result_list[1].bleu_score.score
        op_decoding_bleu_score = translation_result_list[2].bleu_score.score
        tf.reset_default_graph()
        
        self.assertTrue(op_decoder_bleu_score >= tf_bleu_score - 1.0)
        self.assertTrue(op_decoding_bleu_score >= tf_bleu_score - 1.0)
    
    def test_decoding_beamsearch_fp32(self):
        os.system("./bin/decoding_gemm 128 4 8 64 32001 200 512 0")
        self.run_translate(4, 'fp32', '012')
        
    def test_decoding_beamsearch_fp16(self):
        os.system("./bin/decoding_gemm 128 4 8 64 32001 200 512 1")
        self.run_translate(4, 'fp16', '012')
    
    def test_decoding_topk_sampling_fp32(self):
        os.system("./bin/decoding_gemm 128 1 8 64 32001 200 512 0")
        self.run_translate(1, 'fp32', '345', 4, 0.0)
        
    def test_decoding_topk_sampling_fp16(self):
        os.system("./bin/decoding_gemm 128 1 8 64 32001 200 512 1")
        self.run_translate(1, 'fp16', '345', 4, 0.0)
    
    def test_decoding_topp_sampling_fp32(self):
        os.system("./bin/decoding_gemm 128 1 8 64 32001 200 512 0")
        self.run_translate(1, 'fp32', '345', 0, 0.5)
        
    def test_decoding_topp_sampling_fp16(self):
        os.system("./bin/decoding_gemm 128 1 8 64 32001 200 512 1")
        self.run_translate(1, 'fp16', '345', 0, 0.5)

if __name__ == "__main__":
    unittest.main()
