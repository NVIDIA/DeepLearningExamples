#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os, sys
import pickle

import tensorflow as tf
import numpy as np

sys.path.append("/workspace/bert")

from biobert.conlleval import evaluate, report_notprint
import modeling
import optimization
import tokenization
import tf_metrics

import time
import horovod.tensorflow as hvd
from utils.utils import LogEvalRunHook, LogTrainRunHook

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "task_name", "NER", "The name of the task to train."
)

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", False,
    "Whether to run training."
)

flags.DEFINE_bool(
    "do_eval", False,
    "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer(
    "train_batch_size", 64,
    "Total batch size for training.")

flags.DEFINE_integer(
    "eval_batch_size", 16,
    "Total batch size for eval.")

flags.DEFINE_integer(
    "predict_batch_size", 16,
    "Total batch size for predict.")

flags.DEFINE_float(
    "learning_rate", 5e-6,
    "The initial learning rate for Adam.")

flags.DEFINE_float(
    "num_train_epochs", 10.0,
    "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer(
    "save_checkpoints_steps", 1000,
    "How often to save the model checkpoint.")

flags.DEFINE_integer(
    "iterations_per_loop", 1000,
    "How many steps to make in each estimator call.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_bool("horovod", False, "Whether to use Horovod for multi-gpu runs")
flags.DEFINE_bool("use_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU.")
flags.DEFINE_bool("use_xla", False, "Whether to enable XLA JIT compilation.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with tf.io.gfile.Open(input_file, "r") as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                if len(contends) == 0:
                    assert len(words) == len(labels)
                    if len(words) > 30:
                        # split if the sentence is longer than 30
                        while len(words) > 30:
                            tmplabel = labels[:30]
                            for iidx in range(len(tmplabel)):
                                if tmplabel.pop() == 'O':
                                    break
                            l = ' '.join(
                                [label for label in labels[:len(tmplabel) + 1] if len(label) > 0])
                            w = ' '.join(
                                [word for word in words[:len(tmplabel) + 1] if len(word) > 0])
                            lines.append([l, w])
                            words = words[len(tmplabel) + 1:]
                            labels = labels[len(tmplabel) + 1:]

                    if len(words) == 0:
                        continue
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue

                word = line.strip().split()[0]
                label = line.strip().split()[-1]
                words.append(word)
                labels.append(label)
            return lines


class BC5CDRProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        l1 = self._read_data(os.path.join(data_dir, "train.tsv"))
        l2 = self._read_data(os.path.join(data_dir, "devel.tsv"))
        return self._create_example(l1 + l2, "train")

    def get_dev_examples(self, data_dir, file_name="devel.tsv"):
        return self._create_example(
            self._read_data(os.path.join(data_dir, file_name)), "dev"
        )

    def get_test_examples(self, data_dir, file_name="test.tsv"):
        return self._create_example(
            self._read_data(os.path.join(data_dir, file_name)), "test")

    def get_labels(self):
        return ["B", "I", "O", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


class CLEFEProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        lines1 = self._read_data2(os.path.join(data_dir, "Training.tsv"))
        lines2 = self._read_data2(os.path.join(data_dir, "Development.tsv"))
        return self._create_example(
            lines1 + lines2, "train"
        )

    def get_dev_examples(self, data_dir, file_name="Development.tsv"):
        return self._create_example(
            self._read_data2(os.path.join(data_dir, file_name)), "dev"
        )

    def get_test_examples(self, data_dir, file_name="Test.tsv"):
        return self._create_example(
            self._read_data2(os.path.join(data_dir, file_name)), "test")

    def get_labels(self):
        return ["B", "I", "O", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    @classmethod
    def _read_data2(cls, input_file):
        with tf.io.gfile.Open(input_file, "r") as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                if len(contends) == 0:
                    assert len(words) == len(labels)
                    if len(words) == 0:
                        continue
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                elif contends.startswith('###'):
                    continue

                word = line.strip().split()[0]
                label = line.strip().split()[-1]
                words.append(word)
                labels.append(label)
            return lines


class I2b22012Processor(CLEFEProcessor):
    def get_labels(self):
        return ['B-CLINICAL_DEPT', 'B-EVIDENTIAL', 'B-OCCURRENCE', 'B-PROBLEM', 'B-TEST', 'B-TREATMENT', 'I-CLINICAL_DEPT', 'I-EVIDENTIAL', 'I-OCCURRENCE', 'I-PROBLEM', 'I-TEST', 'I-TREATMENT', "O", "X", "[CLS]", "[SEP]"]


def write_tokens(tokens, labels, mode):
    if mode == "test":
        path = os.path.join(FLAGS.output_dir, "token_" + mode + ".txt")
        if tf.io.gfile.Exists(path):
            wf = tf.io.gfile.Open(path, 'a')
        else:
            wf = tf.io.gfile.Open(path, 'w')
        for token, label in zip(tokens, labels):
            if token != "**NULL**":
                wf.write(token + ' ' + str(label) + '\n')
        wf.close()


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    label2id_file = os.path.join(FLAGS.output_dir, 'label2id.pkl')
    if not tf.io.gfile.Exists(label2id_file):
        with tf.io.gfile.Open(label2id_file, 'wb') as w:
            pickle.dump(label_map, w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.compat.v1.logging.info("*** Example ***")
        tf.compat.v1.logging.info("guid: %s" % (example.guid))
        tf.compat.v1.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.compat.v1.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.compat.v1.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.compat.v1.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.compat.v1.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    # write_tokens(ntokens, label_ids, mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,
                                         mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, batch_size, seq_length, is_training, drop_remainder, hvd=None):
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        #batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            if hvd is not None: d = d.shard(hvd.size(), hvd.rank())
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = tf.argmax(probabilities, axis=-1)
        return (loss, per_example_loss, logits, predict)
        ##########################################################################


def model_fn_builder(bert_config, num_labels, init_checkpoint=None, learning_rate=None,
                     num_train_steps=None, num_warmup_steps=None,
                     use_one_hot_embeddings=False, hvd=None, use_fp16=False):
    def model_fn(features, labels, mode, params):
        tf.compat.v1.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, predicts) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint and (hvd is None or hvd.rank() == 0):
            (assignment_map,
             initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
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

            def metric_fn(per_example_loss, label_ids, logits):
                # def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                precision = tf_metrics.precision(label_ids, predictions, num_labels, [1, 2], average="macro")
                recall = tf_metrics.recall(label_ids, predictions, num_labels, [1, 2], average="macro")
                f = tf_metrics.f1(label_ids, predictions, num_labels, [1, 2], average="macro")
                #
                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                    # "eval_loss": loss,
                }

            eval_metric_ops = metric_fn(per_example_loss, label_ids, logits)
            output_spec = tf.estimator.EstimatorSpec(
              mode=mode,
              loss=total_loss,
              eval_metric_ops=eval_metric_ops)
        else:
            output_spec = tf.estimator.EstimatorSpec(
              mode=mode, predictions=predicts)#probabilities)
        return output_spec

    return model_fn


def result_to_pair(predict_line, pred_ids, id2label, writer, err_writer):

    words = str(predict_line.text).split(' ')
    labels = str(predict_line.label).split(' ')
    if len(words) != len(labels):
        tf.compat.v1.logging.error('Text and label not equal')
        tf.compat.v1.logging.error(predict_line.text)
        tf.compat.v1.logging.error(predict_line.label)
        exit(1)

    # get from CLS to SEP
    pred_labels = []
    for id in pred_ids:
        if id == 0:
            continue
        curr_label = id2label[id]
        if curr_label == '[CLS]':
            continue
        elif curr_label == '[SEP]':
            break
        elif curr_label == 'X':
            continue
        pred_labels.append(curr_label)
    if len(pred_labels) > len(words):
        err_writer.write(predict_line.guid + '\n')
        err_writer.write(predict_line.text + '\n')
        err_writer.write(predict_line.label + '\n')
        err_writer.write(' '.join([str(i) for i in pred_ids]) + '\n')
        err_writer.write(' '.join([id2label.get(i, '**NULL**') for i in pred_ids]) + '\n\n')
        pred_labels = pred_labels[:len(words)]
    elif len(pred_labels) < len(words):
        err_writer.write(predict_line.guid + '\n')
        err_writer.write(predict_line.text + '\n')
        err_writer.write(predict_line.label + '\n')
        err_writer.write(' '.join([str(i) for i in pred_ids]) + '\n')
        err_writer.write(' '.join([id2label.get(i, '**NULL**') for i in pred_ids]) + '\n\n')
        pred_labels += ['O'] * (len(words) - len(pred_labels))

    for tok, label, pred_label in zip(words, labels, pred_labels):
        writer.write(tok + ' ' + label + ' ' + pred_label + '\n')
    writer.write('\n')


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    if FLAGS.horovod:
      hvd.init()
    if FLAGS.use_fp16:
        os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"

    processors = {
        "bc5cdr": BC5CDRProcessor,
        "clefe": CLEFEProcessor,
        'i2b2': I2b22012Processor
    }
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
       raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    tf.io.gfile.makedirs(FLAGS.output_dir)

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
        num_labels=len(label_list) + 1,
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
        #train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        #filed_based_convert_examples_to_features(
        #    train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        filed_based_convert_examples_to_features(
          train_examples[start_index:end_index], label_list, FLAGS.max_seq_length, tokenizer, tmp_filenames[hvd_rank])
        tf.compat.v1.logging.info("***** Running training *****")
        tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=tmp_filenames, #train_file,
            batch_size=FLAGS.train_batch_size,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            hvd=None if not FLAGS.horovod else hvd)
        
        #estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
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
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.compat.v1.logging.info("***** Running evaluation *****")
        tf.compat.v1.logging.info("  Num examples = %d", len(eval_examples))
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
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
        with tf.io.gfile.Open(output_eval_file, "w") as writer:
            tf.compat.v1.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.compat.v1.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    if FLAGS.do_predict and master_process:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                 FLAGS.max_seq_length, tokenizer,
                                                 predict_file, mode="test")

        with tf.io.gfile.Open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        if tf.io.gfile.Exists(token_path):
            tf.io.gfile.Remove(token_path)

        tf.compat.v1.logging.info("***** Running prediction*****")
        tf.compat.v1.logging.info("  Num examples = %d", len(predict_examples))
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

        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        test_labels_file = os.path.join(FLAGS.output_dir, "test_labels.txt")
        test_labels_err_file = os.path.join(FLAGS.output_dir, "test_labels_errs.txt")
        with tf.io.gfile.Open(output_predict_file, 'w') as writer, \
                tf.io.gfile.Open(test_labels_file, 'w') as tl, \
                tf.io.gfile.Open(test_labels_err_file, 'w') as tle:
            print(id2label)
            i=0
            for prediction in estimator.predict(input_fn=predict_input_fn, hooks=eval_hooks,
                                                yield_single_examples=True):
                output_line = "\n".join(id2label[id] for id in prediction if id != 0) + "\n"
                writer.write(output_line)
                result_to_pair(predict_examples[i], prediction, id2label, tl, tle)
                i = i + 1

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

        tf.compat.v1.logging.info('Reading: %s', test_labels_file)
        with tf.io.gfile.Open(test_labels_file, "r") as f:
            counts = evaluate(f)
        eval_result = report_notprint(counts)
        print(''.join(eval_result))
        with tf.io.gfile.Open(os.path.join(FLAGS.output_dir, 'test_results_conlleval.txt'), 'w') as fd:
            fd.write(''.join(eval_result))



if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
