# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" BERT inference script. Does not depend on dataset. """

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open

import numpy as np
import torch
from tqdm import tqdm, trange
from types import SimpleNamespace

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertForQuestionAnswering, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from tokenization import (BasicTokenizer, BertTokenizer, whitespace_tokenize)

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


import math
import json
import numpy as np
import collections


def preprocess_tokenized_text(doc_tokens, query_tokens, tokenizer, 
                              max_seq_length, max_query_length):
    """ converts an example into a feature """
    
    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]
    
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    
    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
    
    # truncate if too long
    length = len(all_doc_tokens)
    length = min(length, max_tokens_for_doc)
    
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
    
    for i in range(length):
        token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
        token_is_max_context[len(tokens)] = True
        tokens.append(all_doc_tokens[i])
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
    
    tensors_for_inference = {
                             'input_ids': input_ids, 
                             'input_mask': input_mask, 
                             'segment_ids': segment_ids
                            }
    tensors_for_inference = SimpleNamespace(**tensors_for_inference)
    
    tokens_for_postprocessing = {
                                 'tokens': tokens,
                                 'token_to_orig_map': token_to_orig_map,
                                 'token_is_max_context': token_is_max_context
                                }
    tokens_for_postprocessing = SimpleNamespace(**tokens_for_postprocessing)
    
    return tensors_for_inference, tokens_for_postprocessing


RawResult = collections.namedtuple("RawResult", ["start_logits", "end_logits"])


def get_predictions(doc_tokens, tokens_for_postprocessing, 
                    start_logits, end_logits, n_best_size, 
                    max_answer_length, do_lower_case, 
                    can_give_negative_answer, null_score_diff_threshold):
    """ Write final predictions to the json file and log-odds of null if needed. """
    result = RawResult(start_logits=start_logits, end_logits=end_logits)
    
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", 
        ["start_index", "end_index", "start_logit", "end_logit"])
    
    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    
    start_indices = _get_indices_of_largest_logits(result.start_logits)
    end_indices = _get_indices_of_largest_logits(result.end_logits)
    # if we could have irrelevant answers, get the min score of irrelevant
    if can_give_negative_answer:
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
            score_null = feature_null_score
            null_start_logit = result.start_logits[0]
            null_end_logit = result.end_logits[0]
    for start_index in start_indices:
        for end_index in end_indices:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if start_index >= len(tokens_for_postprocessing.tokens):
                continue
            if end_index >= len(tokens_for_postprocessing.tokens):
                continue
            if start_index not in tokens_for_postprocessing.token_to_orig_map:
                continue
            if end_index not in tokens_for_postprocessing.token_to_orig_map:
                continue
            if not tokens_for_postprocessing.token_is_max_context.get(start_index, False):
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue
            prelim_predictions.append(
                _PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=result.start_logits[start_index],
                    end_logit=result.end_logits[end_index]
                    )
            )
    if can_give_negative_answer:
        prelim_predictions.append(
            _PrelimPrediction(
                start_index=0,
                end_index=0,
                start_logit=null_start_logit,
                end_logit=null_end_logit
            )
        )
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True
    )
    
    _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])
    
    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = tokens_for_postprocessing.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = tokens_for_postprocessing.token_to_orig_map[pred.start_index]
            orig_doc_end = tokens_for_postprocessing.token_to_orig_map[pred.end_index]
            orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)
            
            # de-tokenize WordPieces that have been split off. 
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")
            
            # clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)
            
            # get final text
            final_text = get_final_text(tok_text, orig_text, do_lower_case)
            if final_text in seen_predictions:
                continue
            
            # mark it
            seen_predictions[final_text] = True
            
        else: # this is a null prediction
            final_text = ""
            seen_predictions[final_text] = True
        
        nbest.append(
            _NbestPrediction(
                text=final_text, 
                start_logit=pred.start_logit, 
                end_logit=pred.end_logit
            )
        )
    # if we didn't include the empty option in the n-best, include it 
    if can_give_negative_answer:
        if "" not in seen_predictions:
            nbest.append(
                _NbestPrediction(
                    text="", 
                    start_logit=null_start_logit, 
                    end_logit=null_end_logit
                )
            )
        # In very rare edge cases we could only have single null prediction. 
        # So we just create a nonce prediction in this case to avoid failure. 
        if len(nbest) == 1:
            nbest.insert(0, _NbestPrediction(text="", start_logit=0.0, end_logit=0.0))
    
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure. 
    if not nbest:
        nbest.append(_NbestPrediction(text="", start_logit=0.0, end_logit=0.0))
    
    assert len(nbest) >= 1
    
    # scoring
    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry:
            if entry.text:
                best_non_null_entry = entry
    
    # get probabilities
    probs = _compute_softmax(total_scores)
    
    # nbest predictions into json format
    nbest_json = []
    for (i, entry) in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit
        output["end_logit"] = entry.end_logit
        nbest_json.append(output)
    
    assert len(nbest_json) >= 1
    
    if can_give_negative_answer:
        # predict "unknown" iff ((score_null - score_of_best_non-null_entry) > threshold)
        score = best_non_null_entry.start_logit + best_non_null_entry.end_logit
        score_diff = score_null - score
        if score_diff > null_score_diff_threshold:
            nbest_json[0]['text'] = "unknown"
            # best_non_null_entry.text = "unknown"
    # 
    return nbest_json


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def _get_indices_of_largest_logits(logits):
    """ sort logits and return the indices of the sorted array """
    indices_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    indices = map(lambda x: x[0], indices_and_score)
    indices = list(indices)
    return indices


def main():
    parser = argparse.ArgumentParser()
    
    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="The checkpoint file from pretraining")
    
    ## Other parameters
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--question", default="Most antibiotics target bacteria and don't affect what class of organisms? ", 
                                              type=str, help="question")
    parser.add_argument("--context", default="Within the genitourinary and gastrointestinal tracts, commensal flora serve as biological barriers by competing with pathogenic bacteria for food and space and, in some cases, by changing the conditions in their environment, such as pH or available iron. This reduces the probability that pathogens will reach sufficient numbers to cause illness. However, since most antibiotics non-specifically target bacteria and do not affect fungi, oral antibiotics can lead to an overgrowth of fungi and cause conditions such as a vaginal candidiasis (a yeast infection). There is good evidence that re-introduction of probiotic flora, such as pure cultures of the lactobacilli normally found in unpasteurized yogurt, helps restore a healthy balance of microbial populations in intestinal infections in children and encouraging preliminary data in studies on bacterial gastroenteritis, inflammatory bowel diseases, urinary tract infection and post-surgical infections. ", 
                                              type=str, help="context")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--n_best_size", default=1, type=int,
                        help="The total number of n-best predictions to generate. ")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument('--can_give_negative_answer',
                        action='store_true',
                        help='If true, then the model can reply with "unknown". ')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=-11.0,
                        help="If null_score - best_non_null is greater than the threshold predict 'unknown'. ")
    parser.add_argument('--vocab_file',
                        type=str, default=None, required=True,
                        help="Vocabulary mapping/file BERT was pretrainined on")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="use mixed-precision")
    parser.add_argument("--local_rank", default=-1, help="ordinal of the GPU to use")
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    
    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=512) # for bert large
    
    # Prepare model
    config = BertConfig.from_json_file(args.config_file)
    
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    
    # initialize model
    model = BertForQuestionAnswering(config)
    model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu')["model"])
    model.to(device)
    if args.fp16:
        model.half()
    model.eval()
    
    print("question: ", args.question)
    print("context: ", args.context)
    print()
    
    # preprocessing
    doc_tokens = args.context.split()
    query_tokens = tokenizer.tokenize(args.question)
    feature = preprocess_tokenized_text(doc_tokens, 
                                        query_tokens, 
                                        tokenizer, 
                                        max_seq_length=args.max_seq_length, 
                                        max_query_length=args.max_query_length)
    
    tensors_for_inference, tokens_for_postprocessing = feature
    
    input_ids = torch.tensor(tensors_for_inference.input_ids, dtype=torch.long).unsqueeze(0)
    segment_ids = torch.tensor(tensors_for_inference.segment_ids, dtype=torch.long).unsqueeze(0)
    input_mask = torch.tensor(tensors_for_inference.input_mask, dtype=torch.long).unsqueeze(0)
    
    # load tensors to device
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    
    # run prediction
    with torch.no_grad():
        start_logits, end_logits = model(input_ids, segment_ids, input_mask)
    
    # post-processing
    start_logits = start_logits[0].detach().cpu().tolist()
    end_logits = end_logits[0].detach().cpu().tolist()
    answer = get_predictions(doc_tokens, tokens_for_postprocessing, 
                             start_logits, end_logits, args.n_best_size, 
                             args.max_answer_length, args.do_lower_case, 
                             args.can_give_negative_answer, 
                             args.null_score_diff_threshold)
    
    # print result
    print(json.dumps(answer, indent=4))


if __name__ == "__main__":
    main()

