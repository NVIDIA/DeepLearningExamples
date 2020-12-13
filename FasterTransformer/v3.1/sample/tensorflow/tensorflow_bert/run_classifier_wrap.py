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

# usage example
# export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
# export GLUE_DIR=/path/to/glue
# python run_classifier_wrap.py   --floatx=float16   --task_name=MRPC   --do_eval=true   --data_dir=$GLUE_DIR/MRPC   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=mrpc_output/fp16_model.ckpt   --max_seq_length=128   --eval_batch_size=8   --output_dir=mrpc_output

# FP32 Tensorflow Transformer MRPC result
# INFO:tensorflow:  eval_accuracy = 0.877451
# INFO:tensorflow:  eval_loss = 0.44744828
# INFO:tensorflow:  global_step = 0
# INFO:tensorflow:  loss = 0.44744828

# FP32 Faster Transformer MRPC result
# INFO:tensorflow:  eval_accuracy = 0.877451
# INFO:tensorflow:  eval_loss = 0.4474482
# INFO:tensorflow:  global_step = 0
# INFO:tensorflow:  loss = 0.4474482

# FP16 Tensorflow Transformer MRPC result
# INFO:tensorflow:  eval_accuracy = 0.875
# INFO:tensorflow:  eval_loss = 0.44760832
# INFO:tensorflow:  global_step = 0
# INFO:tensorflow:  loss = 0.44760215

# FP16 Faster Transformer MRPC result
# INFO:tensorflow:  eval_accuracy = 0.875
# INFO:tensorflow:  eval_loss = 0.44731623
# INFO:tensorflow:  global_step = 0
# INFO:tensorflow:  loss = 0.44728807

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
bert_submodule = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bert')
sys.path.insert(0, bert_submodule)
import tensorflow as tf
import run_classifier as rc
import fast_infer_util as fiu
import my_modeling

flags = tf.flags
FLAGS = flags.FLAGS

# replace transformer implementation
my_modeling.transformer_model = fiu.fast_transformer_model_trans
# replace the model to support fp16 data type
rc.create_model = fiu.create_model
# replace the input function to drop remainder
rc.file_based_input_fn_builder = fiu.file_based_input_fn_builder_drop
main = rc.main

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.DEFINE_string("floatx", None, "float32 or float16")
    flags.mark_flag_as_required("floatx")
    flags.DEFINE_bool("remove_padding", False, "Whether remove the padding of sentences")
    flags.DEFINE_integer("int8_mode", 0, "whether use int8 or not; and how to use int8")
    flags.DEFINE_bool("allow_gemm_test", False, "whether allow gemm test inside FT.")
    tf.app.run()
