# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

import os
import sys
import subprocess
import time
import argparse
import json
import logging
import collections

import tensorflow as tf

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from configuration import ElectraConfig
from modeling import TFElectraForQuestionAnswering
from tokenization import ElectraTokenizer
from squad_utils import SquadResult, RawResult, _get_best_indices

TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/electra-small-generator",
    "google/electra-base-generator",
    "google/electra-large-generator",
    "google/electra-small-discriminator",
    "google/electra-base-discriminator",
    "google/electra-large-discriminator",
    # See all ELECTRA models at https://huggingface.co/models?filter=electra
]

_PrelimPrediction = collections.namedtuple(
    "PrelimPrediction",
    ["start_index", "end_index", "start_logit", "end_logit"])


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--electra_model", default=None, type=str, required=True,
                        help="Model selected in the list: " + ", ".join(TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST))
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="The checkpoint file from pretraining")
    parser.add_argument("--question",
                        default=None,
                        type=str,
                        required=True,
                        help="Question")
    parser.add_argument("--context",
                        default=None,
                        type=str,
                        required=True,
                        help="Context")
    parser.add_argument(
        "--joint_head",
        default=True,
        type=bool,
        help="Jointly predict the start and end positions",
    )
    parser.add_argument(
        "--beam_size",
        default=4,
        type=int,
        help="Beam size when doing joint predictions",
    )
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    args = parser.parse_args()

    return args


def get_predictions_joint_head(start_indices, end_indices, result, max_len, args):
    predictions = []
    for i in range(args.beam_size):
        start_index = start_indices[i]
        for j in range(args.beam_size):
            # for end_index in end_indices:
            end_index = end_indices[i * args.beam_size + j]
            if start_index >= max_len:
                continue
            if end_index >= max_len:
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > args.max_answer_length:
                continue
            predictions.append(
                _PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=result.start_logits[i],
                    end_logit=result.end_logits[i * args.beam_size + j]))
    return predictions


def get_predictions(start_indices, end_indices, result, max_len, args):
    predictions = []
    for start_index in start_indices:
        for end_index in end_indices:
            if start_index >= max_len:
                continue
            if end_index >= max_len:
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > args.max_answer_length:
                continue
            predictions.append(
                _PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=result.start_logits[start_index],
                    end_logit=result.end_logits[end_index]))
    return predictions


def main():
    args = parse_args()
    print("***** Loading tokenizer and model *****")
    electra_model = args.electra_model
    config = ElectraConfig.from_pretrained(electra_model)
    tokenizer = ElectraTokenizer.from_pretrained(electra_model)
    model = TFElectraForQuestionAnswering.from_pretrained(electra_model, config=config, args=args)

    print("***** Loading fine-tuned checkpoint: {} *****".format(args.init_checkpoint))
    model.load_weights(args.init_checkpoint, by_name=False, skip_mismatch=False).expect_partial()

    question, text = args.question, args.context
    encoding = tokenizer.encode_plus(question, text, return_tensors='tf')
    input_ids, token_type_ids, attention_mask = encoding["input_ids"], encoding["token_type_ids"], \
                                                encoding["attention_mask"]
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids.numpy()[0])
    if not args.joint_head:
        start_logits, end_logits = model(input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids,
                                         )[:2]
        start_logits = start_logits[0].numpy().tolist()
        end_logits = end_logits[0].numpy().tolist()
        result = RawResult(unique_id=0,
                           start_logits=start_logits,
                           end_logits=end_logits)

        start_indices = _get_best_indices(result.start_logits, args.n_best_size)
        end_indices = _get_best_indices(result.end_logits, args.n_best_size)
        predictions = get_predictions(start_indices, end_indices, result, len(all_tokens), args)
        null_score = result.start_logits[0] + result.end_logits[0]

    else:
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = [output[0].numpy().tolist() for output in outputs]
        start_logits = output[0]
        start_top_index = output[1]
        end_logits = output[2]
        end_top_index = output[3]
        cls_logits = output[4]
        result = SquadResult(
            0,
            start_logits,
            end_logits,
            start_top_index=start_top_index,
            end_top_index=end_top_index,
            cls_logits=cls_logits,
        )
        predictions = get_predictions_joint_head(result.start_top_index, result.end_top_index, result, len(all_tokens), args)
        null_score = result.cls_logits

    predictions = sorted(predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
    answer = predictions[0]
    answer = ' '.join(all_tokens[answer.start_index: answer.end_index + 1])
    if args.null_score_diff_threshold > null_score and args.version_2_with_negative:
        answer = ''

    print(answer)

    return answer


if __name__ == "__main__":
    main()
