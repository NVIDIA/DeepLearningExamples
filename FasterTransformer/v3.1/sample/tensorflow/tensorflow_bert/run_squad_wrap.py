# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
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
# import run_classifier as rc
import run_squad as rs
import fast_infer_util as fiu
import my_modeling

flags = tf.flags
FLAGS = flags.FLAGS

# replace transformer implementation
my_modeling.transformer_model = fiu.fast_transformer_model_trans
# replace the model to support fp16 data type
rs.create_model = fiu.create_model_squad
# replace the input function to drop remainder
rs.file_based_input_fn_builder = fiu.file_based_input_fn_builder_drop


def get_act_seq_len(examples, tokenizer, max_seq_length,
                    doc_stride, max_query_length):

  act_seq_len = []
  for (example_index, example) in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = rs.collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = rs._check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      act_seq_len.append(len(input_ids))
  
  return act_seq_len

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = rs.modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  rs.validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tokenizer = rs.tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = rs.read_squad_examples(
        input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

  model_fn = rs.model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    # We write to a temporary file to avoid storing very large constant tensors
    # in memory.
    train_writer = rs.FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
        is_training=True)
    rs.convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=True,
        output_fn=train_writer.process_feature)
    train_writer.close()

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", len(train_examples))
    tf.logging.info("  Num split examples = %d", train_writer.num_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    del train_examples

    train_input_fn = rs.input_fn_builder(
        input_file=train_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_predict:
    eval_examples = rs.read_squad_examples(
        input_file=FLAGS.predict_file, is_training=False)

    act_seq_len = get_act_seq_len(eval_examples, tokenizer, FLAGS.max_seq_length,
                    FLAGS.doc_stride, FLAGS.max_query_length)

    eval_writer = rs.FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
        is_training=False)
    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    rs.convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    all_results = []

    predict_input_fn = rs.input_fn_builder(
        input_file=eval_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)
    
    # If running eval on the TPU, you will need to specify the number of
    # steps.
    all_results = []
    for idx, result in enumerate(estimator.predict(
        predict_input_fn, yield_single_examples=True)):
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      unique_id = int(result["unique_ids"])
      start_logits = [float(x) for x in result["start_logits"].flat]
      end_logits = [float(x) for x in result["end_logits"].flat]
      all_results.append(
          rs.RawResult(
              unique_id=unique_id,
              start_logits=start_logits[:act_seq_len[idx]],
              end_logits=end_logits[:act_seq_len[idx]]))

    output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
    output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")

    rs.write_predictions(eval_examples, eval_features, all_results,
                         FLAGS.n_best_size, FLAGS.max_answer_length,
                         FLAGS.do_lower_case, output_prediction_file,
                         output_nbest_file, output_null_log_odds_file)

if __name__ == "__main__":
    # flags.mark_flag_as_required("data_dir")
    # flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.DEFINE_string("floatx", None, "float32 or float16")
    flags.mark_flag_as_required("floatx")
    flags.DEFINE_bool("remove_padding", False, "Remove padding or Not")
    flags.DEFINE_integer("int8_mode", 0, "whether use int8 or not; and how to use int8")
    flags.DEFINE_bool("allow_gemm_test", False, "whether allow gemm test inside FT.")
    tf.app.run()
