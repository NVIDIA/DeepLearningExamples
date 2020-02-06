# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import logging
import os, sys
import numpy as np

import tensorflow as tf

sys.path.append("/workspace/bert")

import modeling
import optimization
import tokenization

import time
import horovod.tensorflow as hvd
from utils.utils import LogEvalRunHook, LogTrainRunHook

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-6, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_bool("horovod", False, "Whether to use Horovod for multi-gpu runs")
flags.DEFINE_bool("use_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU.")
flags.DEFINE_bool("use_xla", False, "Whether to enable XLA JIT compilation.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.io.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class BioBERTChemprotProcessor(DataProcessor):
  """Processor for the BioBERT data set obtained from
   (https://github.com/arwhirang/recursive_chemprot/tree/master/Demo/tree_LSTM/data).
   """

  def get_train_examples(self, data_dir, file_name="trainingPosit_chem"):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, file_name)), "train")

  def get_dev_examples(self, data_dir, file_name="developPosit_chem"):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, file_name)), "dev")

  def get_test_examples(self, data_dir, file_name="testPosit_chem"):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, file_name)), "test")

  def get_labels(self):
    """See base class."""
    return ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "False"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "False"
      else:
        text_a = tokenization.convert_to_unicode(line[1])
        label = tokenization.convert_to_unicode(line[2])
        if label == "True":
            label = tokenization.convert_to_unicode(line[3])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


class _ChemProtProcessor(DataProcessor):
    """Processor for the ChemProt data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, file_name="dev.tsv"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_test_examples(self, data_dir, file_name="test.tsv"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # skip header
            if i == 0:
                continue
            guid = line[0]
            text_a = tokenization.convert_to_unicode(line[1])
            if set_type == "test":
                label = self.get_labels()[-1]
            else:
                try:
                    label = tokenization.convert_to_unicode(line[2])
                except IndexError:
                    logging.exception(line)
                    exit(1)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ChemProtProcessor(_ChemProtProcessor):
    def get_labels(self):
        """See base class."""
        return ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "false"]



class MedNLIProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, file_name="dev.tsv"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_test_examples(self, data_dir, file_name="test.tsv"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "test")

    def get_labels(self):
        """See base class."""
        return ['contradiction', 'entailment', 'neutral']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line[1]
            text_a = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[3])
            if set_type == "test":
                label = self.get_labels()[-1]
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.compat.v1.logging.info("*** Example ***")
        tf.compat.v1.logging.info("guid: %s" % (example.guid))
        tf.compat.v1.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.compat.v1.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.compat.v1.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.compat.v1.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, batch_size, seq_length, is_training,
                                drop_remainder, hvd=None):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([], tf.int64),
        "is_real_example": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        #batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            if hvd is not None: d = d.shard(hvd.size(), hvd.rank())
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate=None,
                     num_train_steps=None, num_warmup_steps=None,
                     use_one_hot_embeddings=False, hvd=None, use_fp16=False):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.compat.v1.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint and (hvd is None or hvd.rank() == 0):
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.compat.v1.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, hvd, False, use_fp16)

            output_spec = tf.estimator.EstimatorSpec(
              mode=mode,
              loss=total_loss,
              train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metric_ops = metric_fn(per_example_loss, label_ids, logits, is_real_example)
            output_spec = tf.estimator.EstimatorSpec(
              mode=mode,
              loss=total_loss,
              eval_metric_ops=eval_metric_ops)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                    mode=mode, predictions={"probabilities": probabilities})#predicts)#probabilities)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    if FLAGS.horovod:
      hvd.init()
    if FLAGS.use_fp16:
        os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"

    processors = {
        "chemprot": BioBERTChemprotProcessor,
        'mednli': MedNLIProcessor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.io.gfile.makedirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    master_process = True
    training_hooks = []
    global_batch_size = FLAGS.train_batch_size
    hvd_rank = 0

    config = tf.compat.v1.ConfigProto()
    if FLAGS.horovod:
      global_batch_size = FLAGS.train_batch_size * hvd.size()
      master_process = (hvd.rank() == 0)
      hvd_rank = hvd.rank()
      config.gpu_options.visible_device_list = str(hvd.local_rank())
      if hvd.size() > 1:
        training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

    if FLAGS.use_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
    run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir if master_process else None,
      session_config=config,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps if master_process else None,
      keep_checkpoint_max=1)

    if master_process:
      tf.compat.v1.logging.info("***** Configuaration *****")
      for key in FLAGS.__flags.keys():
          tf.compat.v1.logging.info('  {}: {}'.format(key, getattr(FLAGS, key)))
      tf.compat.v1.logging.info("**************************")

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    training_hooks.append(LogTrainRunHook(global_batch_size, hvd_rank))

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / global_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        start_index = 0
        end_index = len(train_examples)
        tmp_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record")]

        if FLAGS.horovod:
          tmp_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record{}".format(i)) for i in range(hvd.size())]
          num_examples_per_rank = len(train_examples) // hvd.size()
          remainder = len(train_examples) % hvd.size()
          if hvd.rank() < remainder:
            start_index = hvd.rank() * (num_examples_per_rank+1)
            end_index = start_index + num_examples_per_rank + 1
          else:
            start_index = hvd.rank() * num_examples_per_rank + remainder
            end_index = start_index + (num_examples_per_rank)


    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate if not FLAGS.horovod else FLAGS.learning_rate * hvd.size(),
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_one_hot_embeddings=False,
        hvd=None if not FLAGS.horovod else hvd,
        use_fp16=FLAGS.use_fp16)

    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)


    if FLAGS.do_train:
        file_based_convert_examples_to_features(
          train_examples[start_index:end_index], label_list, FLAGS.max_seq_length, tokenizer, tmp_filenames[hvd_rank])
        tf.compat.v1.logging.info("***** Running training *****")
        tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=tmp_filenames,
            batch_size=FLAGS.train_batch_size,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            hvd=None if not FLAGS.horovod else hvd)

        train_start_time = time.time()
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=training_hooks)
        train_time_elapsed = time.time() - train_start_time
        train_time_wo_overhead = training_hooks[-1].total_time
        avg_sentences_per_second = num_train_steps * global_batch_size * 1.0 / train_time_elapsed
        ss_sentences_per_second = (num_train_steps - training_hooks[-1].skipped) * global_batch_size * 1.0 / train_time_wo_overhead

        if master_process:
          tf.compat.v1.logging.info("-----------------------------")
          tf.compat.v1.logging.info("Total Training Time = %0.2f for Sentences = %d", train_time_elapsed,
                        num_train_steps * global_batch_size)
          tf.compat.v1.logging.info("Total Training Time W/O Overhead = %0.2f for Sentences = %d", train_time_wo_overhead,
                        (num_train_steps - training_hooks[-1].skipped) * global_batch_size)
          tf.compat.v1.logging.info("Throughput Average (sentences/sec) with overhead = %0.2f", avg_sentences_per_second)
          tf.compat.v1.logging.info("Throughput Average (sentences/sec) = %0.2f", ss_sentences_per_second)
          tf.compat.v1.logging.info("-----------------------------")


    if FLAGS.do_eval and master_process:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)


        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.compat.v1.logging.info("***** Running evaluation *****")
        tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None

        eval_drop_remainder = False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            batch_size=FLAGS.eval_batch_size,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
            tf.compat.v1.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.compat.v1.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict and master_process:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.compat.v1.logging.info("***** Running prediction*****")
        tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            batch_size=FLAGS.predict_batch_size,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        eval_hooks = [LogEvalRunHook(FLAGS.predict_batch_size)]
        eval_start_time = time.time()


        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.io.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.compat.v1.logging.info("***** Predict results *****")
            for prediction in estimator.predict(input_fn=predict_input_fn, hooks=eval_hooks,
                                                     yield_single_examples=True):
                probabilities = prediction["probabilities"]
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples

        eval_time_elapsed = time.time() - eval_start_time
        eval_time_wo_overhead = eval_hooks[-1].total_time

        time_list = eval_hooks[-1].time_list
        time_list.sort()
        num_sentences = (eval_hooks[-1].count - eval_hooks[-1].skipped) * FLAGS.predict_batch_size

        avg = np.mean(time_list)
        cf_50 = max(time_list[:int(len(time_list) * 0.50)])
        cf_90 = max(time_list[:int(len(time_list) * 0.90)])
        cf_95 = max(time_list[:int(len(time_list) * 0.95)])
        cf_99 = max(time_list[:int(len(time_list) * 0.99)])
        cf_100 = max(time_list[:int(len(time_list) * 1)])
        ss_sentences_per_second = num_sentences * 1.0 / eval_time_wo_overhead

        tf.compat.v1.logging.info("-----------------------------")
        tf.compat.v1.logging.info("Total Inference Time = %0.2f for Sentences = %d", eval_time_elapsed,
                        eval_hooks[-1].count * FLAGS.predict_batch_size)
        tf.compat.v1.logging.info("Total Inference Time W/O Overhead = %0.2f for Sentences = %d", eval_time_wo_overhead,
                        (eval_hooks[-1].count - eval_hooks[-1].skipped) * FLAGS.predict_batch_size)
        tf.compat.v1.logging.info("Summary Inference Statistics")
        tf.compat.v1.logging.info("Batch size = %d", FLAGS.predict_batch_size)
        tf.compat.v1.logging.info("Sequence Length = %d", FLAGS.max_seq_length)
        tf.compat.v1.logging.info("Precision = %s", "fp16" if FLAGS.use_fp16 else "fp32")
        tf.compat.v1.logging.info("Latency Confidence Level 50 (ms) = %0.2f", cf_50 * 1000)
        tf.compat.v1.logging.info("Latency Confidence Level 90 (ms) = %0.2f", cf_90 * 1000)
        tf.compat.v1.logging.info("Latency Confidence Level 95 (ms) = %0.2f", cf_95 * 1000)
        tf.compat.v1.logging.info("Latency Confidence Level 99 (ms) = %0.2f", cf_99 * 1000)
        tf.compat.v1.logging.info("Latency Confidence Level 100 (ms) = %0.2f", cf_100 * 1000)
        tf.compat.v1.logging.info("Latency Average (ms) = %0.2f", avg * 1000)
        tf.compat.v1.logging.info("Throughput Average (sentences/sec) = %0.2f", ss_sentences_per_second)
        tf.compat.v1.logging.info("-----------------------------")

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
