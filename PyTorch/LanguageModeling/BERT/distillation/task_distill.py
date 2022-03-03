# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team and Huawei Noah's Ark Lab.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
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

"""BERT Distillation finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import time, datetime
import math

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

sys.path.append('/workspace/bert/')
from modeling import BertForSequenceClassification, BertForQuestionAnswering, Project, WEIGHTS_NAME, CONFIG_NAME
from schedulers import LinearWarmUpScheduler, ConstantLR
from tokenization_utils import BertTokenizer
from apex.optimizers import FusedAdam

from hooks import *
from losses import *
from utils.utils import is_main_process, get_rank, get_world_size, unwrap_ddp, set_seed
from utils.squad.squad_metrics import compute_predictions, squad_evaluate
import utils.squad.squad_metrics as squad_metrics
from utils.squad.squad_utils import squad_convert_examples_to_features, SquadResult, RawResult, SquadV1Processor, SquadV2Processor, load_and_cache_examples
from itertools import chain
csv.field_size_limit(sys.maxsize)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

oncloud = True
try:
    import moxing as mox
except:
    oncloud = False


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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


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
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        print("Using train data")
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        print("Using Aug Data")
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))
            logger.info("label_id: {}".format(label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


def result_to_file(result, file_name, step, train_summary_writer):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  {} = {}".format(key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))
            train_summary_writer.add_scalar(key, result[key], step)


def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels, amp):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The teacher model dir.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The student model dir.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--amp',
                        action='store_true',
                        default=False,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=-1,
                        type=int,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=1.,
                        help="Gradient Clipping threshold")

    # added arguments
    parser.add_argument('--aug_train',
                        action='store_true')
    parser.add_argument('--value_state_loss',
                        action='store_true',
                        default=False)
    parser.add_argument('--hidden_state_loss',
                        action='store_true',
                        default=False)
    parser.add_argument('--use_last_layer',
                        action='store_true',
                        default=False)
    parser.add_argument('--use_kld',
                        action='store_true',
                        default=False)
    parser.add_argument('--use_cosine',
                        action='store_true',
                        default=False)
    parser.add_argument('--average_loss',
                        action='store_true',
                        default=False)
    parser.add_argument('--eval_step',
                        type=int,
                        default=50)
    parser.add_argument('--pred_distill',
                        action='store_true')
    parser.add_argument('--data_url',
                        type=str,
                        default="")
    parser.add_argument('--temperature',
                        type=float,
                        default=1.)

    #SQUAD args
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
             "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--distill_config',
                        default="distillation_config.json",
                        type=str,
                        help="path the distillation config")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    args.device = device
    logger.info('device: {}'.format(device))
    logger.info('The args: {}'.format(args))

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor,
        "squadv1.1": SquadV1Processor,
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification",
        "squadv1.1": "classification",
    }

    # intermediate distillation default parameters
    default_params = {
        "cola": {"num_train_epochs": 50, "max_seq_length": 64},
        "mnli": {"num_train_epochs": 5, "max_seq_length": 128},
        "mrpc": {"num_train_epochs": 20, "max_seq_length": 128},
        "sst-2": {"num_train_epochs": 10, "max_seq_length": 64},
        "sts-b": {"num_train_epochs": 20, "max_seq_length": 128},
        "qqp": {"num_train_epochs": 5, "max_seq_length": 128},
        "qnli": {"num_train_epochs": 10, "max_seq_length": 128},
        "rte": {"num_train_epochs": 20, "max_seq_length": 128},
        "squadv1.1": {"num_train_epochs": 6, "max_seq_length": 384},
    }

    acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte", "squadv1.1"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]

    # Prepare devices
    #device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    set_seed(args.seed, n_gpu)

    # Prepare task settings
    # Set up tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(args.output_dir, current_time,
                                 'train_' + str(get_rank()) + '_of_' + str(get_world_size()))
    train_summary_writer = SummaryWriter(train_log_dir)

    task_name = args.task_name.lower()

    if task_name in default_params:
        args.max_seq_len = default_params[task_name]["max_seq_length"]

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels() if task_name != "squadv1.1" else []
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)

    if not args.do_eval:
        if task_name == "squadv1.1":
            train_data = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        else:
            if not args.aug_train:
                train_examples = processor.get_train_examples(args.data_dir)
            else:
                train_examples = processor.get_aug_examples(args.data_dir)
        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        if task_name != "squadv1.1":
            num_train_optimization_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

            train_features = convert_examples_to_features(train_examples, label_list,
                                                          args.max_seq_length, tokenizer, output_mode)
            train_data, _ = get_tensor_data(output_mode, train_features)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

        if args.max_steps > 0:
            num_train_optimization_steps = args.max_steps
        else:
            num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if "SQuAD" not in args.task_name:
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    if not args.do_eval:
        if "SQuAD" not in args.task_name:
            teacher_model, teacher_config = BertForSequenceClassification.from_pretrained(args.teacher_model, distill_config=args.distill_config, num_labels=num_labels, pooler=True)
        else:
            teacher_model, teacher_config = BertForQuestionAnswering.from_pretrained(args.teacher_model, distill_config=args.distill_config, pooler=False)

    if "SQuAD" not in args.task_name:
        student_model, student_config = BertForSequenceClassification.from_pretrained(args.student_model, distill_config=args.distill_config, num_labels=num_labels, pooler=True)
    else:
        student_model, student_config = BertForQuestionAnswering.from_pretrained(args.student_model, distill_config=args.distill_config, pooler=False)

    # We need a projection layer since teacher.hidden_size != student.hidden_size
    if not args.do_eval:
        use_projection = student_config.hidden_size != teacher_config.hidden_size
        if student_config.distillation_config["use_hidden_states"] and use_projection:
            project = Project(student_config, teacher_config)
            project_model_file = os.path.join(args.student_model, "project.bin")
            project_ckpt = torch.load(project_model_file, map_location="cpu")
            project.load_state_dict(project_ckpt)
    else:
        use_projection = False

    distill_config = {"nn_module_names": []} #Empty list since we don't want to use nn module hooks here
    distill_hooks_student, distill_hooks_teacher = DistillHooks(distill_config), DistillHooks(distill_config)

    student_model.register_forward_hook(distill_hooks_student.child_to_main_hook)
    if not args.do_eval:
        teacher_model.register_forward_hook(distill_hooks_teacher.child_to_main_hook)

    if not args.do_eval:
        teacher_model.to(device)
    student_model.to(device)
    if student_config.distillation_config["use_hidden_states"] and use_projection:
        project.to(device)

    if args.local_rank != -1:
        if not args.do_eval:
            teacher_model = torch.nn.parallel.DistributedDataParallel(
                    teacher_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False
                )
        student_model = torch.nn.parallel.DistributedDataParallel(
            student_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False
        )
        if student_config.distillation_config["use_hidden_states"] and use_projection:
            project = torch.nn.parallel.DistributedDataParallel(
                         project, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False
                     )


    if args.do_eval:
        global_step = 0
        num_examples = 0
        eval_start = time.time()
        logger.info("***** Running evaluation *****")

        student_model.eval()
        if "SQuAD" not in args.task_name:
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            result = do_eval(student_model, task_name, eval_dataloader,
                         device, output_mode, eval_labels, num_labels, args.amp)
            num_examples = len(eval_examples)
        else:
            dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
            logger.info("  Num examples = %d", len(features))
            logger.info("  Batch size = %d", args.eval_batch_size)
            result = squad_metrics.evaluate(args, student_model, dataset, examples, features, prefix="final")#global_step)
            result["global_step"] = global_step
            num_examples = len(features)

        eval_end = time.time()
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        logger.info("time for inference {} perf {}".format(eval_end - eval_start, num_examples * 100 / (eval_end - eval_start)))
    else:
        scaler = torch.cuda.amp.GradScaler()

        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        # Prepare optimizer
        param_optimizer = list(student_model.named_parameters())
        if student_config.distillation_config["use_hidden_states"] and use_projection:
            param_optimizer += list(project.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()

        logger.info('Total parameters: {}'.format(size))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False)
        schedule = 'warmup_linear'
        if not student_config.distillation_config["use_pred_states"]:
            scheduler = ConstantLR(optimizer)
        else:
            scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion, total_steps=num_train_optimization_steps)
        transformer_losses = TransformerLosses(student_config, teacher_config, device, args)
        def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (- targets_prob * student_likelihood).mean()

        # Train and evaluate
        global_step = 0
        best_dev_acc = 0.0
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

        iter_start = time.time()
        for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_value_loss = 0.
            tr_cls_loss = 0.

            student_model.train()
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                batch = tuple(t.to(device) for t in batch)

                if "squad" in task_name:
                    input_ids, input_mask, segment_ids = batch[:3]
                    label_ids = None
                else:
                    input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
                if input_ids.size()[0] != args.train_batch_size:
                    continue

                att_loss = 0.
                rep_loss = 0.
                value_loss = 0.
                cls_loss = 0.

                with torch.cuda.amp.autocast(enabled=args.amp):
                    student_model(input_ids, segment_ids, input_mask)

                    # Gather student states extracted by hooks
                    temp_model = unwrap_ddp(student_model)
                    student_atts = flatten_states(temp_model.distill_states_dict, "attention_scores")
                    student_reps = flatten_states(temp_model.distill_states_dict, "hidden_states")
                    student_values = flatten_states(temp_model.distill_states_dict, "value_states")
                    student_embeddings = flatten_states(temp_model.distill_states_dict, "embedding_states")
                    student_logits = flatten_states(temp_model.distill_states_dict, "pred_states")
                    if student_config.distillation_config["use_attention_scores"]:
                        bsz, attn_heads, seq_len, _  = student_atts[0].shape

                    #No gradient for teacher training
                    with torch.no_grad():
                        teacher_model(input_ids, segment_ids, input_mask)

                    # Gather teacher states extracted by hooks
                    temp_model = unwrap_ddp(teacher_model)
                    teacher_atts = [i.detach() for i in flatten_states(temp_model.distill_states_dict, "attention_scores")]
                    teacher_reps = [i.detach() for i in flatten_states(temp_model.distill_states_dict, "hidden_states")]
                    teacher_values = [i.detach() for i in flatten_states(temp_model.distill_states_dict, "value_states")]
                    if student_config.distillation_config["use_pred_states"]:
                        if "squad" in task_name:
                            teacher_logits = [[i.detach() for i in flatten_states(temp_model.distill_states_dict, "pred_states")[0]]]
                        else:
                            teacher_logits = [flatten_states(temp_model.distill_states_dict, "pred_states")[0].detach()]
                    teacher_embeddings = [i.detach() for i in flatten_states(temp_model.distill_states_dict, "embedding_states")]

                    #MiniLM
                    if student_config.distillation_config["student_teacher_layer_mapping"] == "last_layer":
                        if student_config.distillation_config["use_attention_scores"]:
                            student_atts = [student_atts[-1]]
                            new_teacher_atts = [teacher_atts[-1]]

                        if student_config.distillation_config["use_value_states"]:
                            student_values = [student_values[-1]]
                            new_teacher_values = [teacher_values[-1]]

                        if student_config.distillation_config["use_hidden_states"]:
                            new_teacher_reps = [teacher_reps[-1]]
                            new_student_reps = [student_reps[-1]]
                    else:
                        if student_config.distillation_config["use_attention_scores"]:
                            teacher_layer_num = len(teacher_atts)
                            student_layer_num = len(student_atts)
                            assert teacher_layer_num % student_layer_num == 0
                            layers_per_block = int(teacher_layer_num / student_layer_num)
                            new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                                for i in range(student_layer_num)]

                        if student_config.distillation_config["use_value_states"]:
                            teacher_layer_num = len(teacher_values)
                            student_layer_num = len(student_values)
                            assert teacher_layer_num % student_layer_num == 0
                            layers_per_block = int(teacher_layer_num / student_layer_num)
                            new_teacher_values = [teacher_values[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]

                        if student_config.distillation_config["use_hidden_states"]:
                            teacher_layer_num = len(teacher_reps)
                            student_layer_num = len(student_reps)
                            assert teacher_layer_num % student_layer_num == 0
                            layers_per_block = int(teacher_layer_num / student_layer_num)
                            new_teacher_reps = [teacher_reps[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]
                            new_student_reps = student_reps

                    if student_config.distillation_config["use_attention_scores"]:
                        att_loss = transformer_losses.compute_loss(student_atts, new_teacher_atts, loss_name="attention_loss")

                    if student_config.distillation_config["use_hidden_states"]:
                        if student_config.distillation_config["use_hidden_states"] and use_projection:
                            rep_loss = transformer_losses.compute_loss(project(new_student_reps), new_teacher_reps, loss_name="hidden_state_loss")
                        else:
                            rep_loss = transformer_losses.compute_loss(new_student_reps, new_teacher_reps, loss_name="hidden_state_loss")

                    if student_config.distillation_config["use_embedding_states"]:
                        if student_config.distillation_config["use_hidden_states"] and use_projection:
                            rep_loss += transformer_losses.compute_loss(project(student_embeddings), teacher_embeddings, loss_name="embedding_state_loss")
                        else:
                            rep_loss += transformer_losses.compute_loss(student_embeddings, teacher_embeddings, loss_name="embedding_state_loss")

                    if student_config.distillation_config["use_value_states"]:
                        value_loss = transformer_losses.compute_loss(student_values, new_teacher_values, loss_name="value_state_loss")

                    if not args.average_loss:
                        loss = rep_loss + att_loss + value_loss
                    else:
                        loss = (rep_loss / len(new_student_reps)) + (att_loss / len(student_atts)) + (value_loss / len(student_values))

                    if student_config.distillation_config["use_attention_scores"]:
                        tr_att_loss += att_loss.item()
                    if student_config.distillation_config["use_hidden_states"]:
                        tr_rep_loss += rep_loss.item()
                    if student_config.distillation_config["use_value_states"]:
                        tr_value_loss += value_loss.item()

                    #pred layer specific
                    if student_config.distillation_config["use_pred_states"]:
                        if output_mode == "classification":
                            if "squad" in task_name:
                                cls_loss = 0.
                                # Iterate over start and end logits
                                for index, student_logit in enumerate(student_logits[0]):
                                    cls_loss += soft_cross_entropy(student_logit / args.temperature,
                                                          teacher_logits[0][index] / args.temperature)
                            else:
                                cls_loss = soft_cross_entropy(student_logits[0] / args.temperature,
                                                          teacher_logits[0] / args.temperature)
                        elif output_mode == "regression":
                            loss_mse = MSELoss()
                            cls_loss = loss_mse(student_logits[0].view(-1), label_ids.view(-1))

                        loss = cls_loss
                        tr_cls_loss += cls_loss.item()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                if student_config.distillation_config["use_hidden_states"] and use_projection:
                    torch.nn.utils.clip_grad_norm_(chain(student_model.parameters(), project.parameters()), args.max_grad_norm, error_if_nonfinite=False)
                else:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm, error_if_nonfinite=False)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()
                    if args.amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()
                    global_step += 1

                if (global_step + 1) % args.eval_step == 0 and is_main_process():
                    logger.info("***** Running evaluation *****")
                    logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                    if not "squad" in task_name:
                        logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    student_model.eval()

                    loss = tr_loss / (step + 1)
                    cls_loss = tr_cls_loss / (step + 1)
                    att_loss = tr_att_loss / (step + 1)
                    rep_loss = tr_rep_loss / (step + 1)
                    value_loss = tr_value_loss / (step + 1)

                    result = {}
                    if student_config.distillation_config["use_pred_states"]:
                        if "SQuAD" not in args.task_name:
                            result = do_eval(student_model, task_name, eval_dataloader,
                                             device, output_mode, eval_labels, num_labels, args.amp)
                        else:
                            dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
                            result = squad_metrics.evaluate(args, student_model, dataset, examples, features, prefix="final")#global_step)

                    result['global_step'] = global_step
                    result['lr'] = optimizer.param_groups[0]["lr"]
                    result['cls_loss'] = cls_loss
                    result['att_loss'] = att_loss
                    result['rep_loss'] = rep_loss
                    result['value_loss'] = value_loss
                    result['loss'] = loss
                    result['perf'] = (global_step + 1) * get_world_size() * args.train_batch_size * args.gradient_accumulation_steps / (time.time() - iter_start)

                    if is_main_process():
                        result_to_file(result, output_eval_file, global_step, train_summary_writer)

                    if not student_config.distillation_config["use_pred_states"]:
                        save_model = True
                    else:
                        save_model = False

                        if task_name in acc_tasks and result['acc'] > best_dev_acc:
                            best_dev_acc = result['acc']
                            save_model = True

                        if task_name in corr_tasks and result['corr'] > best_dev_acc:
                            best_dev_acc = result['corr']
                            save_model = True

                        if task_name in mcc_tasks and result['mcc'] > best_dev_acc:
                            best_dev_acc = result['mcc']
                            save_model = True

                    if save_model and is_main_process():
                        logger.info("***** Save model *****")

                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

                        model_name = WEIGHTS_NAME
                        output_model_file = os.path.join(args.output_dir, model_name)
                        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir)

                        # Test mnli-mm
                        if student_config.distillation_config["use_pred_states"] and task_name == "mnli":
                            task_name = "mnli-mm"
                            processor = processors[task_name]()
                            if not os.path.exists(args.output_dir + '-MM'):
                                os.makedirs(args.output_dir + '-MM')

                            eval_examples = processor.get_dev_examples(args.data_dir)

                            eval_features = convert_examples_to_features(
                                eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
                            eval_data, eval_labels = get_tensor_data(output_mode, eval_features)

                            logger.info("***** Running mm evaluation *****")
                            logger.info("  Num examples = %d", len(eval_examples))
                            logger.info("  Batch size = %d", args.eval_batch_size)

                            eval_sampler = SequentialSampler(eval_data)
                            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                                         batch_size=args.eval_batch_size, num_workers=4)

                            result = do_eval(student_model, task_name, eval_dataloader,
                                             device, output_mode, eval_labels, num_labels, args.amp)

                            result['global_step'] = global_step

                            tmp_output_eval_file = os.path.join(args.output_dir + '-MM', "eval_results.txt")
                            result_to_file(result, tmp_output_eval_file, global_step, train_summary_writer)

                            task_name = 'mnli'

                        if oncloud:
                            logging.info(mox.file.list_directory(args.output_dir, recursive=True))
                            logging.info(mox.file.list_directory('.', recursive=True))
                            mox.file.copy_parallel(args.output_dir, args.data_url)
                            mox.file.copy_parallel('.', args.data_url)

                    student_model.train()
    train_summary_writer.flush()
    train_summary_writer.close()

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total time taken:", time.time() - start_time)
