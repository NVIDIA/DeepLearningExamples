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
# python profile_bert_inference.py   --task_name=MRPC   --data_dir=$GLUE_DIR/MRPC   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --predict_batch_size=8   --max_seq_length=128   --output_dir=mrpc_output --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --tf_profile=true  --profiling_output_file=time_elapsed --xla=false --floatx=float32
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import numpy as np
import fast_infer_util as fiu
import profile_util
import tensorflow as tf
import os
from tensorflow.python.client import timeline
import contextlib
import time
from tensorflow.python.client import device_lib
import my_modeling
bert_submodule = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bert')
sys.path.insert(0, bert_submodule)
import tokenization
import run_classifier as rc

flags = tf.flags
FLAGS = flags.FLAGS


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids):
    """Creates a classification model."""
    model = my_modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)
    seq_output = model.get_sequence_output()
    return seq_output


def model_fn_builder(bert_config):

    def model_fn(features):

        # print features
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        fetches = create_model(
            bert_config, False, input_ids, input_mask, segment_ids)

        # # fetch mrpc logits for prediction
        # num_labels = 2  # for mrpc
        # _, _, fetches, _ = fiu.create_model(bert_config, False, input_ids, input_mask, segment_ids, label_ids,
        # num_labels, False)

        return fetches

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    num_iter = 20
    jit_xla = tf.OptimizerOptions.ON_1 if FLAGS.xla else 0

    processors = {
        "cola": rc.ColaProcessor,
        "mnli": rc.MnliProcessor,
        "mrpc": rc.MrpcProcessor,
        "xnli": rc.XnliProcessor,
    }

    # sanity check
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)
    bert_config = my_modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    tf.gfile.MakeDirs(FLAGS.output_dir)
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # prepare data
    processor = processors[task_name]()
    label_list = processor.get_labels()
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    rc.file_based_convert_examples_to_features(predict_examples, label_list,
                                               FLAGS.max_seq_length, tokenizer,
                                               predict_file)

    # get model function and input function
    # drop_remainder option should be turned on for fast transformer inference
    drop_remainder = True
    predict_input_fn = rc.file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=drop_remainder)

    def graph_fn():
        model_fn = model_fn_builder(bert_config=bert_config)
        dataset = predict_input_fn({'batch_size': FLAGS.predict_batch_size})
        next_item = dataset.make_one_shot_iterator().get_next()
        output_var = model_fn(next_item)
        return output_var

    if FLAGS.tf_profile:
        tf.logging.info("***** Running tensorflow transformer*****")
        p1 = profile_util.Profiler(os.path.join(
            FLAGS.output_dir, 'prof/bert_origin'))
        t1, r1 = profile_util.run_profile(
            graph_fn, jit_xla, num_iter, p1, init_checkpoint=FLAGS.init_checkpoint)
        tf.reset_default_graph()

        my_modeling.transformer_model = fiu.fast_transformer_model_trans
        tf.logging.info("***** Running fast transformer*****")
        p2 = profile_util.Profiler(os.path.join(
            FLAGS.output_dir, 'prof/bert_fastinfer'))
        t2, r2 = profile_util.run_profile(
            graph_fn, jit_xla, num_iter, p2, init_checkpoint=FLAGS.init_checkpoint)

    else:
        tf.logging.info("***** Running tensorflow transformer*****")
        t1, r1 = profile_util.run_profile(
            graph_fn, jit_xla, num_iter, check_result=False, init_checkpoint=FLAGS.init_checkpoint)
        tf.reset_default_graph()

        my_modeling.transformer_model = fiu.fast_transformer_model_trans
        tf.logging.info("***** Running fast transformer*****")
        t2, r2 = profile_util.run_profile(
            graph_fn, jit_xla, num_iter, check_result=False, init_checkpoint=FLAGS.init_checkpoint)

    print('average time (seconds) elasped original tensorflow:', t1)
    print('average time (seconds) elasped fast transformer:', t2)
    if len(r1) + len(r2) > 0:
        check_res = np.asarray([np.allclose(
            r1[i], r2[i], atol=1e-4, rtol=0) for i in range(num_iter)])
        if check_res.all():
            print('Pass')
            print(np.mean(r1))
            print(np.mean(r2))
        else:
            for i in np.where(np.logical_not(check_res))[0]:
                diff = np.fabs(r1[i] - r2[i])
                idx = np.unravel_index(diff.argmax(), diff.shape)
                print('Failed iter:', i, "max diff:",
                      diff[idx], idx, r1[i][idx], r2[i][idx])


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.DEFINE_string("profiling_output_file", None,
                        "The output file for profiling results.")
    flags.mark_flag_as_required("profiling_output_file")
    flags.DEFINE_string("floatx", "float32", "float32 or float16")
    flags.mark_flag_as_required("floatx")
    flags.DEFINE_bool("xla", False, "whether to turn on XLA")
    flags.mark_flag_as_required("xla")
    flags.DEFINE_bool("tf_profile", False,
                      "whether to use tensorflow profiling")
    tf.app.run()
