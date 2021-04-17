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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import os.path
import json
import copy
import sys
sys.path.append("./tensorflow/tensorflow_bert")
import tensorflow as tf
from squad_evaluate_v1_1 import evaluate
# from ckpt_type_convert import checkpoint_dtype_cast

class TestDecoding(unittest.TestCase):

    is_init = False
    
    @classmethod
    def setUpClass(cls):
        super(TestDecoding, cls).setUpClass()
        if cls.is_init == False:
            cls.expected_version = '1.1'
            cls.truth_dataset = "squad_data/dev-v1.1.json"
            cls.fp32_model_path = "squad_model/model.ckpt"
            cls.fp16_model_path = "squad_fp16_model/model.ckpt"
            assert(os.path.isfile(cls.truth_dataset))
            assert(os.path.isfile(cls.fp32_model_path + ".index"))
            if(not os.path.isfile(cls.fp16_model_path + ".index")):
                os.system("python tensorflow/tensorflow_bert/ckpt_type_convert.py --init_checkpoint={} --fp16_checkpoint={}".format(cls.fp32_model_path, cls.fp16_model_path))
            cls.tf_fp32_output_path = "squad_tf_output/fp32/"
            cls.ft_fp32_output_path = "squad_ft_output/fp32/"
            cls.ft_fp16_output_path = "squad_ft_output/fp16/"
            cls.predict_filename = "predictions.json"

            os.system("python tensorflow/tensorflow_bert/bert/run_squad.py \
                        --predict_batch_size=8 \
                        --vocab_file=squad_model/vocab.txt \
                        --bert_config_file=squad_model/bert_config.json \
                        --init_checkpoint={} \
                        --train_file=squad_data/train-v1.1.json \
                        --do_predict=True \
                        --predict_file=squad_data/dev-v1.1.json \
                        --max_seq_length=384 \
                        --output_dir={}".format(cls.fp32_model_path, cls.tf_fp32_output_path))
            cls.tf_fp32_score = cls.run_evaluate(cls, cls.tf_fp32_output_path + cls.predict_filename)
            print("[INFO] tensorflow results: {}".format(cls.tf_fp32_score))
            cls.is_init = True

    def run_evaluate(self, file_path):
        with open(file_path) as f, open(self.truth_dataset) as b:
            f_json = json.load(f)
            b_json = json.load(b)

            if (b_json['version'] != self.expected_version):
                print('Evaluation expects v-' + self.expected_version +
                    ', but got dataset with v-' + b_json['version'],
                    file=sys.stderr)
            dataset = b_json['data']
            score = evaluate(dataset, f_json)
            return score

    def test_squad_fp32(self):
        print("{INFO] test_squad_fp32")
        os.system("./bin/encoder_gemm 8 384 12 64 0 0")
        os.system("python tensorflow/tensorflow_bert/run_squad_wrap.py \
                    --floatx=float32 \
                    --predict_batch_size=8 \
                    --vocab_file=squad_model/vocab.txt \
                    --bert_config_file=squad_model/bert_config.json \
                    --init_checkpoint={} \
                    --train_file=squad_data/train-v1.1.json \
                    --do_predict=True \
                    --predict_file=squad_data/dev-v1.1.json \
                    --max_seq_length=384 \
                    --output_dir={}".format(self.fp32_model_path, self.ft_fp32_output_path))
        os.system("rm gemm_config.in")
        self.ft_fp32_score = self.run_evaluate(self.ft_fp32_output_path + self.predict_filename)
        print("[INFO] fp32 results: {}".format(self.ft_fp32_score))
        assert(self.ft_fp32_score['f1'] > self.tf_fp32_score['f1'] - 0.1)
        assert(self.ft_fp32_score['exact_match'] > self.tf_fp32_score['exact_match'] - 0.1)
    
    def test_squad_fp16(self):
        print("[INFO] test_squad_fp16")
        os.system("./bin/encoder_gemm 8 384 12 64 1 0")
        os.system("python tensorflow/tensorflow_bert/run_squad_wrap.py \
                    --floatx=float16 \
                    --predict_batch_size=8 \
                    --vocab_file=squad_model/vocab.txt \
                    --bert_config_file=squad_model/bert_config.json \
                    --init_checkpoint={} \
                    --train_file=squad_data/train-v1.1.json \
                    --do_predict=True \
                    --predict_file=squad_data/dev-v1.1.json \
                    --max_seq_length=384 \
                    --output_dir={}".format(self.fp16_model_path, self.ft_fp16_output_path))
        os.system("rm gemm_config.in")
        self.ft_fp16_score = self.run_evaluate(self.ft_fp16_output_path + self.predict_filename)
        print("[INFO] fp16 results: {}".format(self.ft_fp16_score))
        assert(self.ft_fp16_score['f1'] > self.tf_fp32_score['f1'] - 0.1)
        assert(self.ft_fp16_score['exact_match'] > self.tf_fp32_score['exact_match'] - 0.1)

    def test_squad_fp16_varSeqlen(self):
        print("[INFO] test_squad_fp16_varSeqlen")
        os.system("python tensorflow/tensorflow_bert/run_squad_wrap.py \
                    --floatx=float16 \
                    --predict_batch_size=8 \
                    --vocab_file=squad_model/vocab.txt \
                    --bert_config_file=squad_model/bert_config.json \
                    --init_checkpoint={} \
                    --train_file=squad_data/train-v1.1.json \
                    --do_predict=True \
                    --predict_file=squad_data/dev-v1.1.json \
                    --max_seq_length=384 \
                    --remove_padding=True \
                    --output_dir={}".format(self.fp16_model_path, self.ft_fp16_output_path))
        self.ft_fp16_score_var_seqlen = self.run_evaluate(self.ft_fp16_output_path + self.predict_filename)
        print("[INFO] fp16 var seqlen results: {}".format(self.ft_fp16_score_var_seqlen))
        assert(self.ft_fp16_score_var_seqlen['f1'] > self.tf_fp32_score['f1'] - 0.1)
        assert(self.ft_fp16_score_var_seqlen['exact_match'] > self.tf_fp32_score['exact_match'] - 0.1)
     
if __name__ == "__main__":
    unittest.main()
