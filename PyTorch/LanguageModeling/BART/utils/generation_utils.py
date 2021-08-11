# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
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

import logging

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from utils.file_utils import ModelOutput
from utils.generation_beam_search import BeamScorer, BeamSearchScorer
from utils.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    HammingDiversityLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

logger = logging.getLogger(__name__)


# class GenerationMixin:
#     """
#     A class contraining all of the functions supporting generation, to be used as a mixin in
#     :class:`~transfomers.PreTrainedModel`.
#     """

#     def prepare_inputs_for_generation(self, input_ids, **kwargs):
#         """
#         Implement in subclasses of :class:`~transfomers.PreTrainedModel` for custom behavior to prepare inputs in the
#         generate method.
#         """
#         return {"input_ids": input_ids}

#     def adjust_logits_during_generation(self, logits, **kwargs):
#         """
#         Implement in subclasses of :class:`~transfomers.PreTrainedModel` for custom behavior to adjust the logits in
#         the generate method.
#         """
#         return logits

#     def _use_cache(self, outputs, use_cache):
#         """During generation, decide whether to pass the `past` variable to the next forward pass."""
#         if len(outputs) <= 1 or use_cache is False:
#             return False
#         if hasattr(self.config, "mem_len") and self.config.mem_len == 0:
#             return False
#         return True

#     def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
#         """
#         Enforce the repetition penalty (from the `CTRL paper <https://arxiv.org/abs/1909.05858>`__).
#         """
#         for i in range(batch_size * num_beams):
#             for previous_token in set(prev_output_tokens[i].tolist()):
#                 # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
#                 if lprobs[i, previous_token] < 0:
#                     lprobs[i, previous_token] *= repetition_penalty
#                 else:
#                     lprobs[i, previous_token] /= repetition_penalty

#     def postprocess_next_token_scores(
#         self,
#         scores,
#         input_ids,
#         no_repeat_ngram_size,
#         bad_words_ids,
#         cur_len,
#         min_length,
#         max_length,
#         eos_token_id,
#         repetition_penalty,
#         batch_size,
#         num_beams,
#     ):
#         # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
#         if repetition_penalty != 1.0:
#             self.enforce_repetition_penalty_(
#                 scores, batch_size, num_beams, input_ids, repetition_penalty,
#             )

#         # set eos token prob to zero if min_length is not reached
#         if eos_token_id is not None and cur_len < min_length:
#             scores[:, eos_token_id] = -float("inf")

#         if no_repeat_ngram_size > 0:
#             # calculate a list of banned tokens to prevent repetitively generating the same ngrams
#             num_batch_hypotheses = batch_size * num_beams
#             # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
#             banned_batch_tokens = calc_banned_ngram_tokens(
#                 input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
#             )
#             for i, banned_tokens in enumerate(banned_batch_tokens):
#                 scores[i, banned_tokens] = -float("inf")

#         if bad_words_ids is not None:
#             # Exclude EOS token (already processed)
#             bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], bad_words_ids))
#             # calculate a list of banned tokens according to bad words
#             banned_tokens = calc_banned_bad_words_ids(input_ids.tolist(), bad_words_ids)
#             # Modify the scores in place by setting the banned tokens logits to `-inf`
#             set_scores_to_inf_for_banned_tokens(scores, banned_tokens)

#         return scores

#     @torch.no_grad()
#     def generate(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         max_length: Optional[int] = None,
#         min_length: Optional[int] = None,
#         do_sample: Optional[bool] = None,
#         early_stopping: Optional[bool] = None,
#         num_beams: Optional[int] = None,
#         temperature: Optional[float] = None,
#         top_k: Optional[int] = None,
#         top_p: Optional[float] = None,
#         repetition_penalty: Optional[float] = None,
#         bad_words_ids: Optional[Iterable[int]] = None,
#         bos_token_id: Optional[int] = None,
#         pad_token_id: Optional[int] = None,
#         eos_token_id: Optional[int] = None,
#         length_penalty: Optional[float] = None,
#         no_repeat_ngram_size: Optional[int] = None,
#         num_return_sequences: Optional[int] = None,
#         attention_mask: Optional[torch.LongTensor] = None,
#         decoder_start_token_id: Optional[int] = None,
#         use_cache: Optional[bool] = None,
#         **model_specific_kwargs
#     ) -> torch.LongTensor:
#         r"""
#         Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
#         beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

#         Adapted in part from `Facebook's XLM beam search code
#         <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.

#         Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
#         attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
#         indicated are the default values of those config.

#         Most of these parameters are explained in more detail in `this blog post
#         <https://huggingface.co/blog/how-to-generate>`__.

#         Parameters:

#             input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#                 The sequence used as a prompt for the generation. If :obj:`None` the method initializes
#                 it as an empty :obj:`torch.LongTensor` of shape :obj:`(1,)`.
#             max_length (:obj:`int`, `optional`, defaults to 20):
#                 The maximum length of the sequence to be generated.
#             min_length (:obj:`int`, `optional`, defaults to 10):
#                 The minimum length of the sequence to be generated.
#             do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
#                 Whether or not to use sampling ; use greedy decoding otherwise.
#             early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
#                 Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
#             num_beams (:obj:`int`, `optional`, defaults to 1):
#                 Number of beams for beam search. 1 means no beam search.
#             temperature (:obj:`float`, `optional`, defaults tp 1.0):
#                 The value used to module the next token probabilities.
#             top_k (:obj:`int`, `optional`, defaults to 50):
#                 The number of highest probability vocabulary tokens to keep for top-k-filtering.
#             top_p (:obj:`float`, `optional`, defaults to 1.0):
#                 If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or
#                 higher are kept for generation.
#             repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
#                 The parameter for repetition penalty. 1.0 means no penalty. See `this paper
#                 <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
#             pad_token_id (:obj:`int`, `optional`):
#                 The id of the `padding` token.
#             bos_token_id (:obj:`int`, `optional`):
#                 The id of the `beginning-of-sequence` token.
#             eos_token_id (:obj:`int`, `optional`):
#                 The id of the `end-of-sequence` token.
#             length_penalty (:obj:`float`, `optional`, defaults to 1.0):
#                 Exponential penalty to the length. 1.0 means no penalty.

#                 Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
#                 order to encourage the model to produce longer sequences.
#             no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
#                 If set to int > 0, all ngrams of that size can only occur once.
#             bad_words_ids(:obj:`List[int]`, `optional`):
#                 List of token ids that are not allowed to be generated. In order to get the tokens of the words that
#                 should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
#             num_return_sequences(:obj:`int`, `optional`, defaults to 1):
#                 The number of independently computed returned sequences for each element in the batch.
#             attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#                 Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
#                 tokens that are not masked, and 0 for masked tokens.

#                 If not provided, will default to a tensor the same shape as :obj:`input_ids` that masks the pad token.

#                 `What are attention masks? <../glossary.html#attention-mask>`__
#             decoder_start_token_id (:obj:`int`, `optional`):
#                 If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
#             use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
#                 Whether or not the model should use the past last key/values attentions (if applicable to the model) to
#                 speed up decoding.
#             model_specific_kwargs:
#                 Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.

#         Return:

#             :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`:
#             The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
#             shorter if all batches finished early due to the :obj:`eos_token_id`.

#         Examples::

#             tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
#             model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
#             outputs = model.generate(max_length=40)  # do greedy decoding
#             print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

#             tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
#             model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
#             input_context = 'The dog'
#             input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
#             outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
#             for i in range(3): #  3 output sequences were generated
#                 print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

#             tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
#             model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
#             input_context = 'The dog'
#             input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
#             outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3, do_sample=True)  # generate 3 candidates using sampling
#             for i in range(3): #  3 output sequences were generated
#                 print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

#             tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
#             model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
#             input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
#             input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
#             outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
#             print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

#             tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
#             model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
#             input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
#             bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
#             input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
#             outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
#         """

#         # We cannot generate if the model does not have a LM head
#         if self.get_output_embeddings() is None:
#             raise AttributeError(
#                 "You tried to generate sequences with a model that does not have a LM Head."
#                 "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
#             )

#         max_length = max_length if max_length is not None else self.config.max_length
#         min_length = min_length if min_length is not None else self.config.min_length
#         do_sample = do_sample if do_sample is not None else self.config.do_sample
#         early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         num_beams = num_beams if num_beams is not None else self.config.num_beams
#         temperature = temperature if temperature is not None else self.config.temperature
#         top_k = top_k if top_k is not None else self.config.top_k
#         top_p = top_p if top_p is not None else self.config.top_p
#         repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
#         bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
#         pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
#         eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
#         length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
#         no_repeat_ngram_size = (
#             no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
#         )
#         bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
#         num_return_sequences = (
#             num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
#         )
#         decoder_start_token_id = (
#             decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
#         )

#         if input_ids is not None:
#             batch_size = input_ids.shape[0]  # overriden by the input batch_size
#         else:
#             batch_size = 1

#         assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
#         assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
#         assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
#         assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
#         assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
#         assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
#         assert temperature > 0, "`temperature` should be strictly positive."
#         assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
#         assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
#         assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
#         assert input_ids is not None or (
#             isinstance(bos_token_id, int) and bos_token_id >= 0
#         ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
#         assert pad_token_id is None or (
#             isinstance(pad_token_id, int) and (pad_token_id >= 0)
#         ), "`pad_token_id` should be a positive integer."
#         assert (eos_token_id is None) or (
#             isinstance(eos_token_id, int) and (eos_token_id >= 0)
#         ), "`eos_token_id` should be a positive integer."
#         assert length_penalty > 0, "`length_penalty` should be strictly positive."
#         assert (
#             isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
#         ), "`no_repeat_ngram_size` should be a positive integer."
#         assert (
#             isinstance(num_return_sequences, int) and num_return_sequences > 0
#         ), "`num_return_sequences` should be a strictly positive integer."
#         assert (
#             bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
#         ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

#         if input_ids is None:
#             assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
#                 "you should either supply a context to complete as `input_ids` input "
#                 "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
#             )
#             input_ids = torch.full(
#                 (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
#             )
#         else:
#             assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

#         # not allow to duplicate outputs when greedy decoding
#         if do_sample is False:
#             if num_beams == 1:
#                 # no_beam_search greedy generation conditions
#                 assert (
#                     num_return_sequences == 1
#                 ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

#             else:
#                 # beam_search greedy generation conditions
#                 assert (
#                     num_beams >= num_return_sequences
#                 ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

#         # create attention mask if necessary
#         # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
#         if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
#             attention_mask = input_ids.ne(pad_token_id).long()
#         elif attention_mask is None:
#             attention_mask = input_ids.new_ones(input_ids.shape)

#         # set pad_token_id to eos_token_id if not set. Important that this is done after
#         # attention_mask is created
#         if pad_token_id is None and eos_token_id is not None:
#             logger.warning(
#                 "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
#             )
#             pad_token_id = eos_token_id

#         # current position and vocab size
#         if hasattr(self.config, "vocab_size"):
#             vocab_size = self.config.vocab_size
#         elif (
#             self.config.is_encoder_decoder
#             and hasattr(self.config, "decoder")
#             and hasattr(self.config.decoder, "vocab_size")
#         ):
#             vocab_size = self.config.decoder.vocab_size

#         # set effective batch size and effective batch multiplier according to do_sample
#         if do_sample:
#             effective_batch_size = batch_size * num_return_sequences
#             effective_batch_mult = num_return_sequences
#         else:
#             effective_batch_size = batch_size
#             effective_batch_mult = 1

#         if self.config.is_encoder_decoder:
#             if decoder_start_token_id is None:
#                 # see if BOS token can be used for decoder_start_token_id
#                 if bos_token_id is not None:
#                     decoder_start_token_id = bos_token_id
#                 elif hasattr(self.config, "decoder") and hasattr(self.config.decoder, "bos_token_id"):
#                     decoder_start_token_id = self.config.decoder.bos_token_id
#                 else:
#                     raise ValueError(
#                         "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
#                     )

#             assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
#             assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

#             # get encoder and store encoder outputs
#             encoder = self.get_encoder()
#             encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask)

#         # Expand input ids if num_beams > 1 or num_return_sequences > 1
#         if num_return_sequences > 1 or num_beams > 1:
#             input_ids_len = input_ids.shape[-1]
#             input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
#             attention_mask = attention_mask.unsqueeze(1).expand(
#                 batch_size, effective_batch_mult * num_beams, input_ids_len
#             )

#             input_ids = input_ids.contiguous().view(
#                 effective_batch_size * num_beams, input_ids_len
#             )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
#             attention_mask = attention_mask.contiguous().view(
#                 effective_batch_size * num_beams, input_ids_len
#             )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

#         if self.config.is_encoder_decoder:
#             # create empty decoder_input_ids
#             input_ids = torch.full(
#                 (effective_batch_size * num_beams, 1),
#                 decoder_start_token_id,
#                 dtype=torch.long,
#                 device=next(self.parameters()).device,
#             )
#             cur_len = 1

#             assert (
#                 batch_size == encoder_outputs[0].shape[0]
#             ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

#             # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
#             expanded_batch_idxs = (
#                 torch.arange(batch_size)
#                 .view(-1, 1)
#                 .repeat(1, num_beams * effective_batch_mult)
#                 .view(-1)
#                 .to(input_ids.device)
#             )
#             # expand encoder_outputs
#             encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

#         else:
#             encoder_outputs = None
#             cur_len = input_ids.shape[-1]

#         assert (
#             cur_len < max_length
#         ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

#         if num_beams > 1:
#             output = self._generate_beam_search(
#                 input_ids,
#                 cur_len=cur_len,
#                 max_length=max_length,
#                 min_length=min_length,
#                 do_sample=do_sample,
#                 early_stopping=early_stopping,
#                 temperature=temperature,
#                 top_k=top_k,
#                 top_p=top_p,
#                 repetition_penalty=repetition_penalty,
#                 no_repeat_ngram_size=no_repeat_ngram_size,
#                 bad_words_ids=bad_words_ids,
#                 pad_token_id=pad_token_id,
#                 eos_token_id=eos_token_id,
#                 batch_size=effective_batch_size,
#                 num_return_sequences=num_return_sequences,
#                 length_penalty=length_penalty,
#                 num_beams=num_beams,
#                 vocab_size=vocab_size,
#                 encoder_outputs=encoder_outputs,
#                 attention_mask=attention_mask,
#                 use_cache=use_cache,
#                 model_specific_kwargs=model_specific_kwargs,
#             )
#         else:
#             output = self._generate_no_beam_search(
#                 input_ids,
#                 cur_len=cur_len,
#                 max_length=max_length,
#                 min_length=min_length,
#                 do_sample=do_sample,
#                 temperature=temperature,
#                 top_k=top_k,
#                 top_p=top_p,
#                 repetition_penalty=repetition_penalty,
#                 no_repeat_ngram_size=no_repeat_ngram_size,
#                 bad_words_ids=bad_words_ids,
#                 pad_token_id=pad_token_id,
#                 eos_token_id=eos_token_id,
#                 batch_size=effective_batch_size,
#                 encoder_outputs=encoder_outputs,
#                 attention_mask=attention_mask,
#                 use_cache=use_cache,
#                 model_specific_kwargs=model_specific_kwargs,
#             )

#         return output

#     def _generate_no_beam_search(
#         self,
#         input_ids,
#         cur_len,
#         max_length,
#         min_length,
#         do_sample,
#         temperature,
#         top_k,
#         top_p,
#         repetition_penalty,
#         no_repeat_ngram_size,
#         bad_words_ids,
#         pad_token_id,
#         eos_token_id,
#         batch_size,
#         encoder_outputs,
#         attention_mask,
#         use_cache,
#         model_specific_kwargs,
#     ):
#         """ Generate sequences for each example without beam search (num_beams == 1).
#             All returned sequence are generated independantly.
#         """
#         # length of generated sentences / unfinished sentences
#         unfinished_sents = input_ids.new(batch_size).fill_(1)
#         sent_lengths = input_ids.new(batch_size).fill_(max_length)

#         past = (encoder_outputs, None) if encoder_outputs is not None else None

#         while cur_len < max_length:
#             model_inputs = self.prepare_inputs_for_generation(
#                 input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
#             )

#             outputs = self(**model_inputs)
#             next_token_logits = outputs[0][:, -1, :]

#             scores = self.postprocess_next_token_scores(
#                 scores=next_token_logits,
#                 input_ids=input_ids,
#                 no_repeat_ngram_size=no_repeat_ngram_size,
#                 bad_words_ids=bad_words_ids,
#                 cur_len=cur_len,
#                 min_length=min_length,
#                 max_length=max_length,
#                 eos_token_id=eos_token_id,
#                 repetition_penalty=repetition_penalty,
#                 batch_size=batch_size,
#                 num_beams=1,
#             )

#             # if model has past, then set the past variable to speed up decoding
#             if self._use_cache(outputs, use_cache):
#                 past = outputs[1]

#             if do_sample:
#                 # Temperature (higher temperature => more likely to sample low probability tokens)
#                 if temperature != 1.0:
#                     scores = scores / temperature
#                 # Top-p/top-k filtering
#                 next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
#                 # Sample
#                 probs = F.softmax(next_token_logscores, dim=-1)
#                 next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
#             else:
#                 # Greedy decoding
#                 next_token = torch.argmax(next_token_logits, dim=-1)

#             # update generations and finished sentences
#             if eos_token_id is not None:
#                 # pad finished sentences if eos_token_id exist
#                 tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
#             else:
#                 tokens_to_add = next_token

#             # add token and increase length by one
#             input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
#             cur_len = cur_len + 1

#             if eos_token_id is not None:
#                 eos_in_sents = tokens_to_add == eos_token_id
#                 # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
#                 is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
#                 sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
#                 # unfinished_sents is set to zero if eos in sentence
#                 unfinished_sents.mul_((~eos_in_sents).long())

#             # stop when there is a </s> in each sentence, or if we exceed the maximul length
#             if unfinished_sents.max() == 0:
#                 break

#             # extend attention_mask for new generated input if only decoder
#             if self.config.is_encoder_decoder is False:
#                 attention_mask = torch.cat(
#                     [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
#                 )

#         return input_ids

#     def _generate_beam_search(
#         self,
#         input_ids,
#         cur_len,
#         max_length,
#         min_length,
#         do_sample,
#         early_stopping,
#         temperature,
#         top_k,
#         top_p,
#         repetition_penalty,
#         no_repeat_ngram_size,
#         bad_words_ids,
#         pad_token_id,
#         eos_token_id,
#         batch_size,
#         num_return_sequences,
#         length_penalty,
#         num_beams,
#         vocab_size,
#         encoder_outputs,
#         attention_mask,
#         use_cache,
#         model_specific_kwargs,
#     ):
#         """ Generate sequences for each example with beam search.
#         """

#         # generated hypotheses
#         generated_hyps = [
#             BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
#             for _ in range(batch_size)
#         ]

#         # scores for each sentence in the beam
#         beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

#         # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
#         if do_sample is False:
#             beam_scores[:, 1:] = -1e9
#         beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

#         # cache compute states
#         past = (encoder_outputs, None) if encoder_outputs is not None else None

#         # done sentences
#         done = [False for _ in range(batch_size)]

#         while cur_len < max_length:
#             model_inputs = self.prepare_inputs_for_generation(
#                 input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
#             )
#             print(model_inputs)
#             outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
#             next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

#             # if model has past, then set the past variable to speed up decoding
#             if self._use_cache(outputs, use_cache):
#                 past = outputs[1]
#             if self.config.is_encoder_decoder and do_sample is False:
#                 # TODO (PVP) still a bit hacky here - there might be a better solution
#                 next_token_logits = self.adjust_logits_during_generation(
#                     next_token_logits, cur_len=cur_len, max_length=max_length
#                 )

#             scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

#             scores = self.postprocess_next_token_scores(
#                 scores=scores,
#                 input_ids=input_ids,
#                 no_repeat_ngram_size=no_repeat_ngram_size,
#                 bad_words_ids=bad_words_ids,
#                 cur_len=cur_len,
#                 min_length=min_length,
#                 max_length=max_length,
#                 eos_token_id=eos_token_id,
#                 repetition_penalty=repetition_penalty,
#                 batch_size=batch_size,
#                 num_beams=num_beams,
#             )

#             assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
#                 scores.shape, (batch_size * num_beams, vocab_size)
#             )

#             if do_sample:
#                 _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
#                 # Temperature
#                 if temperature != 1.0:
#                     _scores = _scores / temperature
#                 # Top-p/top-k filtering
#                 _scores = top_k_top_p_filtering(
#                     _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
#                 )  # (batch_size * num_beams, vocab_size)
#                 # re-organize to group the beam together to sample from all beam_idxs
#                 _scores = _scores.contiguous().view(
#                     batch_size, num_beams * vocab_size
#                 )  # (batch_size, num_beams * vocab_size)

#                 # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
#                 probs = F.softmax(_scores, dim=-1)
#                 next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
#                 # Compute next scores
#                 next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
#                 # sort the sampled vector to make sure that the first num_beams samples are the best
#                 next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
#                 next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

#             else:
#                 next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

#                 # re-organize to group the beam together (we are keeping top hypothesis accross beams)
#                 next_scores = next_scores.view(
#                     batch_size, num_beams * vocab_size
#                 )  # (batch_size, num_beams * vocab_size)

#                 next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

#             assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

#             # next batch beam content
#             next_batch_beam = []

#             # for each sentence
#             for batch_idx in range(batch_size):

#                 # if we are done with this sentence, add a pad token
#                 if done[batch_idx]:
#                     assert (
#                         len(generated_hyps[batch_idx]) >= num_beams
#                     ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
#                     assert (
#                         eos_token_id is not None and pad_token_id is not None
#                     ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
#                     next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
#                     continue

#                 # next sentence beam content, this will get added to next_batch_beam
#                 next_sent_beam = []

#                 # next tokens for this sentence
#                 for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
#                     zip(next_tokens[batch_idx], next_scores[batch_idx])
#                 ):
#                     # get beam and token IDs
#                     beam_id = beam_token_id // vocab_size
#                     token_id = beam_token_id % vocab_size

#                     effective_beam_id = batch_idx * num_beams + beam_id
#                     # add to generated hypotheses if end of sentence
#                     if (eos_token_id is not None) and (token_id.item() == eos_token_id):
#                         # if beam_token does not belong to top num_beams tokens, it should not be added
#                         is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
#                         if is_beam_token_worse_than_top_num_beams:
#                             continue
#                         generated_hyps[batch_idx].add(
#                             input_ids[effective_beam_id].clone(), beam_token_score.item(),
#                         )
#                     else:
#                         # add next predicted token since it is not eos_token
#                         next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

#                     # once the beam for next step is full, don't add more tokens to it.
#                     if len(next_sent_beam) == num_beams:
#                         break

#                 # Check if we are done so that we can save a pad step if all(done)
#                 done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
#                     next_scores[batch_idx].max().item(), cur_len
#                 )

#                 # update next beam content
#                 assert len(next_sent_beam) == num_beams, "Beam should always be full"
#                 next_batch_beam.extend(next_sent_beam)
#                 assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

#             # stop when we are done with each sentence
#             if all(done):
#                 break

#             # sanity check / prepare next batch
#             assert len(next_batch_beam) == batch_size * num_beams
#             beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
#             beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
#             beam_idx = input_ids.new([x[2] for x in next_batch_beam])

#             # re-order batch and update current length
#             input_ids = input_ids[beam_idx, :]
#             input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
#             cur_len = cur_len + 1

#             # re-order internal states
#             if past is not None:
#                 past = self._reorder_cache(past, beam_idx)

#             # extend attention_mask for new generated input if only decoder
#             if self.config.is_encoder_decoder is False:
#                 attention_mask = torch.cat(
#                     [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
#                 )

#         # finalize all open beam hypotheses and add to generated hypotheses
#         for batch_idx in range(batch_size):
#             if done[batch_idx]:
#                 continue

#             # test that beam scores match previously calculated scores if not eos and batch_idx not done
#             if eos_token_id is not None and all(
#                 (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
#             ):
#                 assert torch.all(
#                     next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
#                 ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
#                     next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
#                 )

#             # need to add best num_beams hypotheses to generated hyps
#             for beam_id in range(num_beams):
#                 effective_beam_id = batch_idx * num_beams + beam_id
#                 final_score = beam_scores[effective_beam_id].item()
#                 final_tokens = input_ids[effective_beam_id]
#                 generated_hyps[batch_idx].add(final_tokens, final_score)

#         # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
#         output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
#         output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

#         # select the best hypotheses
#         sent_lengths = input_ids.new(output_batch_size)
#         best = []

#         # retrieve best hypotheses
#         for i, hypotheses in enumerate(generated_hyps):
#             sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
#             for j in range(output_num_return_sequences_per_batch):
#                 effective_batch_idx = output_num_return_sequences_per_batch * i + j
#                 best_hyp = sorted_hyps.pop()[1]
#                 sent_lengths[effective_batch_idx] = len(best_hyp)
#                 best.append(best_hyp)

#         # shorter batches are padded
#         if sent_lengths.min().item() != sent_lengths.max().item():
#             assert pad_token_id is not None, "`Pad_token_id` has to be defined"
#             sent_max_len = min(sent_lengths.max().item() + 1, max_length)
#             decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

#             # fill with hypothesis and eos_token_id if necessary
#             for i, hypo in enumerate(best):
#                 decoded[i, : sent_lengths[i]] = hypo
#                 if sent_lengths[i] < max_length:
#                     decoded[i, sent_lengths[i]] = eos_token_id
#         else:
#             # none of the hypotheses have an eos_token
#             assert (len(hypo) == max_length for hypo in best)
#             decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

#         return decoded

#     @staticmethod
#     def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
#         return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)


# def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
#     """Copied from fairseq for no_repeat_ngram in beam_search"""
#     if cur_len + 1 < no_repeat_ngram_size:
#         # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
#         return [[] for _ in range(num_hypos)]
#     generated_ngrams = [{} for _ in range(num_hypos)]
#     for idx in range(num_hypos):
#         gen_tokens = prev_input_ids[idx].tolist()
#         generated_ngram = generated_ngrams[idx]
#         for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
#             prev_ngram_tuple = tuple(ngram[:-1])
#             generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

#     def _get_generated_ngrams(hypo_idx):
#         # Before decoding the next token, prevent decoding of ngrams that have already appeared
#         start_idx = cur_len + 1 - no_repeat_ngram_size
#         ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
#         return generated_ngrams[hypo_idx].get(ngram_idx, [])

#     banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
#     return banned_tokens


# def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
#     banned_tokens = []

#     def _tokens_match(prev_tokens, tokens):
#         if len(tokens) == 0:
#             # if bad word tokens is just one token always ban it
#             return True
#         if len(tokens) > len(prev_tokens):
#             # if bad word tokens are longer than prev tokens they can't be equal
#             return False

#         if prev_tokens[-len(tokens) :] == tokens:
#             # if tokens match
#             return True
#         else:
#             return False

#     for prev_input_ids_slice in prev_input_ids:
#         banned_tokens_slice = []

#         for banned_token_seq in bad_words_ids:
#             assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
#                 bad_words_ids
#             )

#             if _tokens_match(prev_input_ids_slice, banned_token_seq[:-1]) is False:
#                 # if tokens do not match continue
#                 continue

#             banned_tokens_slice.append(banned_token_seq[-1])

#         banned_tokens.append(banned_tokens_slice)

#     return banned_tokens


# def set_scores_to_inf_for_banned_tokens(scores: torch.Tensor, banned_tokens: List[List[int]]) -> None:
#     """ Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be
#     a list of list of banned tokens to ban in the format [[batch index, vocabulary position],...]
#         Args:
#             scores: logits distribution of shape (batch size, vocabulary size)
#             banned_tokens: list of list of tokens to ban of length (batch_size)
#     """
#     banned_mask_list = []
#     for idx, batch_banned_tokens in enumerate(banned_tokens):
#         for token in batch_banned_tokens:
#             banned_mask_list.append([idx, token])
#     if not banned_mask_list:
#         return
#     banned_mask = torch.LongTensor(banned_mask_list)
#     indices = torch.ones(len(banned_mask))
#     # A sparse tensor is generated from a list of coordinates: [[0, 1], [0, 2], [2, 0]]. A conversion to dense tensor generates:
#     # [ 0  1  1 ]
#     # [ 0  0  0 ]
#     # [ 1  0  0 ]

#     banned_mask = torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
#     scores.masked_fill_(banned_mask, -float("inf"))


# def top_k_top_p_filtering(
#     logits: Tensor,
#     top_k: int = 0,
#     top_p: float = 1.0,
#     filter_value: float = -float("Inf"),
#     min_tokens_to_keep: int = 1,
# ) -> Tensor:
#     """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
#         Args:
#             logits: logits distribution shape (batch size, vocabulary size)
#             if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
#             if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
#                 Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
#             Make sure we keep at least min_tokens_to_keep per batch example in the output
#         From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
#     """
#     if top_k > 0:
#         top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
#         # Remove all tokens with a probability less than the last token of the top-k
#         indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
#         logits[indices_to_remove] = filter_value

#     if top_p < 1.0:
#         sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#         cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

#         # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
#         sorted_indices_to_remove = cumulative_probs > top_p
#         if min_tokens_to_keep > 1:
#             # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
#             sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
#         # Shift the indices to the right to keep also the first token above the threshold
#         sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#         sorted_indices_to_remove[..., 0] = 0

#         # scatter sorted tensors to original indexing
#         indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
#         logits[indices_to_remove] = filter_value
#     return logits


# class BeamHypotheses(object):
#     def __init__(self, num_beams, max_length, length_penalty, early_stopping):
#         """
#         Initialize n-best list of hypotheses.
#         """
#         self.max_length = max_length - 1  # ignoring bos_token
#         self.length_penalty = length_penalty
#         self.early_stopping = early_stopping
#         self.num_beams = num_beams
#         self.beams = []
#         self.worst_score = 1e9

#     def __len__(self):
#         """
#         Number of hypotheses in the list.
#         """
#         return len(self.beams)

#     def add(self, hyp, sum_logprobs):
#         """
#         Add a new hypothesis to the list.
#         """
#         score = sum_logprobs / len(hyp) ** self.length_penalty
#         if len(self) < self.num_beams or score > self.worst_score:
#             self.beams.append((score, hyp))
#             if len(self) > self.num_beams:
#                 sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
#                 del self.beams[sorted_scores[0][1]]
#                 self.worst_score = sorted_scores[1][0]
#             else:
#                 self.worst_score = min(score, self.worst_score)

#     def is_done(self, best_sum_logprobs, cur_len):
#         """
#         If there are enough hypotheses and that none of the hypotheses being generated
#         can become better than the worst one in the heap, then we are done with this sentence.
#         """

#         if len(self) < self.num_beams:
#             return False
#         elif self.early_stopping:
#             return True
#         else:
#             cur_score = best_sum_logprobs / cur_len ** self.length_penalty
#             ret = self.worst_score >= cur_score
#             return ret


@dataclass
class GreedySearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. :obj:`(max_length,)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor of
            shape :obj:`(batch_size, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GreedySearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using greedy search. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)


    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. :obj:`(max_length,)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor of
            shape :obj:`(batch_size, config.vocab_size)`).
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer of the decoder) of shape :obj:`(batch_size,
            num_heads, sequence_length, sequence_length)`.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class SampleDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using sampling.


    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. :obj:`(max_length,)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor of
            shape :obj:`(batch_size*num_return_sequences, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(num_return_sequences*batch_size, num_heads, generated_length,
            sequence_length)`.
        hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(num_return_sequences*batch_size, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class SampleEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using sampling. Hidden states and attention weights of
    the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)


    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. :obj:`(max_length,)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor of
            shape :obj:`(batch_size*num_return_sequences, config.vocab_size)`).
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer of the decoder) of shape
            :obj:`(batch_size*num_return_sequences, num_heads, sequence_length, sequence_length)`.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_return_sequences, num_heads, generated_length,
            sequence_length)`.
        decoder_hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_return_sequences, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class BeamSearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using beam search.

    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        sequences_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_return_sequences)`, `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Final beam scores of the generated ``sequences``.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . :obj:`(max_length,)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor of shape
            :obj:`(batch_size*num_beams*num_return_sequences, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams, num_heads, generated_length,
            sequence_length)`.
        hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams*num_return_sequences, generated_length,
            hidden_size)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class BeamSearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights
    of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)

    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        sequences_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_return_sequences)`, `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Final beam scores of the generated ``sequences``.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . :obj:`(max_length,)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor of shape
            :obj:`(batch_size*num_beams, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer of the decoder) of shape :obj:`(batch_size,
            num_heads, sequence_length, sequence_length)`.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams*num_return_sequences, num_heads,
            generated_length, sequence_length)`.
        decoder_hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams*num_return_sequences, generated_length,
            hidden_size)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class BeamSampleDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using beam sample.

    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        sequences_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_return_sequence)`, `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Final beam scores of the generated ``sequences``.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . :obj:`(max_length,)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor of shape
            :obj:`(batch_size*num_beams*num_return_sequences, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams, num_heads, generated_length,
            sequence_length)`.
        hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class BeamSampleEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam sampling. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)

    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_beams, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        sequences_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_return_sequence)`, `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Final beam scores of the generated ``sequences``.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . :obj:`(max_length,)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor of shape
            :obj:`(batch_size*num_beams, config.vocab_size)`).
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer of the decoder) of shape :obj:`(batch_size,
            num_heads, sequence_length, sequence_length)`.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size*num_beams, sequence_length, hidden_size)`.
        decoder_attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams, num_heads, generated_length,
            sequence_length)`.
        decoder_hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]


class GenerationMixin:
    """
    A class containing all of the functions supporting generation, to be used as a mixin in
    :class:`~transformers.PreTrainedModel`.
    """

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to prepare inputs in the
        generate method.
        """
        return {"input_ids": input_ids}

    def adjust_logits_during_generation(self, logits: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to adjust the logits in
        the generate method.
        """
        return logits

    def _prepare_input_ids_for_generation(self, bos_token_id: int) -> torch.LongTensor:
        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")
        return torch.ones((1, 1), dtype=torch.long, device=self.device) * bos_token_id

    def _prepare_attention_mask_for_generation(
        self, input_ids: torch.Tensor, pad_token_id: int, eos_token_id: int
    ) -> torch.LongTensor:
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and (pad_token_id in input_ids)
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            return input_ids.ne(pad_token_id).long()
        return input_ids.new_ones(input_ids.shape)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        # retrieve encoder hidden states
        encoder = self.get_encoder()
        encoder_kwargs = {
            argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
        }
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)
        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self, input_ids: torch.LongTensor, decoder_start_token_id: int = None, bos_token_id: int = None
    ) -> torch.LongTensor:
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        decoder_input_ids = (
            torch.ones((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
            * decoder_start_token_id
        )
        return decoder_input_ids

    def _get_pad_token_id(self, pad_token_id: int = None, eos_token_id: int = None) -> int:
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id
        return pad_token_id

    def _get_decoder_start_token_id(self, decoder_start_token_id: int = None, bos_token_id: int = None) -> int:
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "decoder_start_token_id")
            and self.config.decoder.decoder_start_token_id is not None
        ):
            return self.config.decoder.decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "bos_token_id")
            and self.config.decoder.bos_token_id is not None
        ):
            return self.config.decoder.bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

    @staticmethod
    def _init_sequence_length_for_generation(
        input_ids: torch.LongTensor, max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        sequence_lengths = input_ids.new(input_ids.shape[0]).fill_(max_length)

        cur_len = input_ids.shape[-1]
        return sequence_lengths, unfinished_sequences, cur_len

    @staticmethod
    def _update_seq_length_for_generation(
        sequence_lengths: torch.LongTensor,
        unfinished_sequences: torch.LongTensor,
        cur_len: int,
        is_eos_in_next_token: torch.BoolTensor,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        # check if sentence is not finished yet
        is_sent_unfinished = unfinished_sequences.mul(is_eos_in_next_token.long()).bool()

        # update sentence length
        sequence_lengths = sequence_lengths.masked_fill(is_sent_unfinished, cur_len)
        unfinished_sequences = unfinished_sequences.mul((~is_eos_in_next_token).long())
        return sequence_lengths, unfinished_sequences

    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return model_kwargs

    def _reorder_cache(self, past, beam_idx):
        raise NotImplementedError(
            f"Make sure that a `_reorder_cache` function is correctly implemented in {self.__class__.__module__} to enable beam search for {self.__class__}"
        )

    def _get_logits_warper(
        self, top_k: int = None, top_p: float = None, temperature: float = None, num_beams: int = None
    ) -> LogitsProcessorList:
        """
        This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.LogitsWarper` instances used for multinomial sampling.
        """

        # init warp parameters
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        temperature = temperature if temperature is not None else self.config.temperature
        # instantiate warpers list
        warpers = LogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        return warpers

    def _get_logits_processor(
        self,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        encoder_no_repeat_ngram_size: int,
        encoder_input_ids: torch.LongTensor,
        bad_words_ids: List[List[int]],
        min_length: int,
        eos_token_id: int,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
    ) -> LogitsProcessorList:
        """
        This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
        """

        # init warp parameters
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        encoder_no_repeat_ngram_size = (
            encoder_no_repeat_ngram_size
            if encoder_no_repeat_ngram_size is not None
            else self.config.encoder_no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        min_length = min_length if min_length is not None else self.config.min_length
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        diversity_penalty = diversity_penalty if diversity_penalty is not None else self.config.diversity_penalty
        # instantiate processors list
        processors = LogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if diversity_penalty is not None and diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
                )
            )
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if encoder_no_repeat_ngram_size is not None and encoder_no_repeat_ngram_size > 0:
            if self.config.is_encoder_decoder:
                processors.append(EncoderNoRepeatNGramLogitsProcessor(encoder_no_repeat_ngram_size, encoder_input_ids))
            else:
                raise ValueError(
                    "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture"
                )
        if bad_words_ids is not None:
            processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        if prefix_allowed_tokens_fn is not None:
            processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams))
        return processors

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        multinomial sampling, beam-search decoding, and beam-search multinomial sampling.

        Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
        attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
        indicated are the default values of those config.

        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (:obj:`float`, `optional`, defaults tp 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or
                higher are kept for generation.
            repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
                model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
                sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            encoder_no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size that occur in the ``encoder_input_ids`` cannot occur in the
                ``decoder_input_ids``.
            bad_words_ids(:obj:`List[List[int]]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer(bad_word,
                add_prefix_space=True).input_ids`.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same
                shape as :obj:`input_ids` that masks the pad token. `What are attention masks?
                <../glossary.html#attention-mask>`__
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            num_beam_groups (:obj:`int`, `optional`, defaults to 1):
                Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
                beams. `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
            diversity_penalty (:obj:`float`, `optional`, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that :obj:`diversity_penalty` is only effective if ``group beam search`` is
                enabled.
            prefix_allowed_tokens_fn: (:obj:`Callable[[int, torch.Tensor], List[int]]`, `optional`):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments :obj:`inputs_ids` and the batch ID
                :obj:`batch_id`. It has to return a list with the allowed tokens for the next generation step
                conditioned on the previously generated tokens :obj:`inputs_ids` and the batch ID :obj:`batch_id`. This
                argument is useful for constrained generation conditioned on the prefix, as described in
                `Autoregressive Entity Retrieval <https://arxiv.org/abs/2010.00904>`__.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return trhe hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.

            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If the
                model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific
                kwargs should be prefixed with `decoder_`.

        Return:
            :class:`~transformers.file_utils.ModelOutput` or :obj:`torch.LongTensor`: A
            :class:`~transformers.file_utils.ModelOutput` (if ``return_dict_in_generate=True`` or when
            ``config.return_dict_in_generate=True``) or a :obj:`torch.FloatTensor`.

                If the model is `not` an encoder-decoder model (``model.config.is_encoder_decoder=False``), the
                possible :class:`~transformers.file_utils.ModelOutput` types are:

                    - :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.SampleDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.BeamSampleDecoderOnlyOutput`

                If the model is an encoder-decoder model (``model.config.is_encoder_decoder=True``), the possible
                :class:`~transformers.file_utils.ModelOutput` types are:

                    - :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.SampleEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.BeamSampleEncoderDecoderOutput`

        Examples::
            >>> from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

            >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            >>> # do greedy decoding without providing a prompt
            >>> outputs = model.generate(max_length=40)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
            >>> document = (
            ... "at least two people were killed in a suspected bomb attack on a passenger bus "
            ... "in the strife-torn southern philippines on monday , the military said."
            ... )
            >>> # encode input contex
            >>> input_ids = tokenizer(document, return_tensors="pt").input_ids
            >>> # generate 3 independent sequences using beam search decoding (5 beams)
            >>> # with T5 encoder-decoder model conditioned on short news article.
            >>> outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            >>> input_context = "The dog"
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> # generate 3 candidates using sampling
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, num_return_sequences=3, do_sample=True)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("ctrl")
            >>> model = AutoModelForCausalLM.from_pretrained("ctrl")
            >>> # "Legal" is one of the control codes for ctrl
            >>> input_context = "Legal My neighbor is"
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, repetition_penalty=1.2)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> input_context = "My cute dog"
            >>> # get tokens of words that should not be generated
            >>> bad_words_ids = [tokenizer(bad_word, add_prefix_space=True).input_ids for bad_word in ["idiot", "stupid", "shut up"]]
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> # generate sequences without allowing bad_words to be generated
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, do_sample=True, bad_words_ids=bad_words_ids)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
        """

        # set init values
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states

        if input_ids is None:
            # init `input_ids` with bos_token_id
            input_ids = self._prepare_input_ids_for_generation(bos_token_id)

        if model_kwargs.get("attention_mask", None) is None:
            # init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )

        # special case if pad_token_id is not defined
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        # Storing encoder_input_ids for logits_processor that could use them
        encoder_input_ids = input_ids if self.config.is_encoder_decoder else None

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self._prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id
                )

            if "encoder_outputs" not in model_kwargs or not isinstance(model_kwargs["encoder_outputs"], ModelOutput):
                raise ValueError("Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")

        if input_ids.shape[-1] >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # determine generation mode
        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
        is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # set model_kwargs
        model_kwargs["use_cache"] = use_cache

        # get distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=encoder_input_ids,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            eos_token_id=eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
        )

        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # get probability distribution warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            # expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # interleave with `num_beams`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            batch_size = input_ids.shape[0] * num_return_sequences

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # interleave with `num_beams * num_return_sequences`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams * num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if num_beams % num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            diverse_beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            # interleave with `num_beams`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            return self.group_beam_search(
                input_ids,
                diverse_beam_scorer,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using greedy decoding.



        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return trhe hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.

            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the :obj:`forward` function of the
                model. If model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.

        Examples::

            >>> from transformers import (
            ... AutoTokenizer,
            ... AutoModelForCausalLM,
            ... LogitsProcessorList,
            ... MinLengthLogitsProcessor,
            ... )

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

            >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
            >>> model.config.pad_token_id = model.config.eos_token_id

            >>> input_prompt = "Today is a beautiful day, and"
            >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
            ... ])

            >>> outputs = model.greedy_search(input_ids, logits_processor=logits_processor)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # init sequence length tensors
        sequence_lengths, unfinished_sequences, cur_len = self._init_sequence_length_for_generation(
            input_ids, max_length
        )

        while cur_len < max_length:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits = outputs.logits[:, -1, :]

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self._update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            # update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # increase cur_len
            cur_len = cur_len + 1

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using multinomial sampling.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            logits_warper (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsWarper` used to warp the prediction score distribution of the language
                modeling head applied before multinomial sampling at each generation step.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return trhe hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If
                model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utils.SampleDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.SampleEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.SampleDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.SampleEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.

        Examples::

            >>> from transformers import (
            ...    AutoTokenizer,
            ...    AutoModelForCausalLM,
            ...    LogitsProcessorList,
            ...    MinLengthLogitsProcessor,
            ...    TopKLogitsWarper,
            ...    TemperatureLogitsWarper,
            ... )

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

            >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
            >>> model.config.pad_token_id = model.config.eos_token_id

            >>> input_prompt = "Today is a beautiful day, and"
            >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
            ... ])
            >>> # instantiate logits processors
            >>> logits_warper = LogitsProcessorList([
            ...     TopKLogitsWarper(50),
            ...     TemperatureLogitsWarper(0.7),
            ... ])

            >>> outputs = model.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # init sequence length tensors
        sequence_lengths, unfinished_sequences, cur_len = self._init_sequence_length_for_generation(
            input_ids, max_length
        )

        # auto-regressive generation
        while cur_len < max_length:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            cur_len = cur_len + 1

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self._update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search decoding.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            beam_scorer (:obj:`BeamScorer`):
                An derived instance of :class:`~transformers.BeamScorer` that defines how beam hypotheses are
                constructed, stored and sorted during generation. For more information, the documentation of
                :class:`~transformers.BeamScorer` should be read.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return trhe hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If
                model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utilsBeamSearchDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.


        Examples::

            >>> from transformers import (
            ...    AutoTokenizer,
            ...    AutoModelForSeq2SeqLM,
            ...    LogitsProcessorList,
            ...    MinLengthLogitsProcessor,
            ...    BeamSearchScorer,
            ... )
            >>> import torch

            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

            >>> encoder_input_str = "translate English to German: How old are you?"
            >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


            >>> # lets run beam search using 3 beams
            >>> num_beams = 3
            >>> # define decoder start token ids
            >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
            >>> input_ids = input_ids * model.config.decoder_start_token_id

            >>> # add encoder_outputs to model keyword arguments
            >>> model_kwargs = {
            ...     "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
            ... }

            >>> # instantiate beam scorer
            >>> beam_scorer = BeamSearchScorer(
            ...     batch_size=1,
            ...     max_length=model.config.max_length,
            ...     num_beams=num_beams,
            ...     device=model.device,
            ... )

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
            ... ])

            >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits = outputs.logits[:, -1, :]

            # adjust tokens for Bart, *e.g.*
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            cur_len = cur_len + 1

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[BeamSampleOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search with multinomial sampling.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            beam_scorer (:obj:`BeamScorer`):
                A derived instance of :class:`~transformers.BeamScorer` that defines how beam hypotheses are
                constructed, stored and sorted during generation. For more information, the documentation of
                :class:`~transformers.BeamScorer` should be read.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            logits_warper (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsWarper` used to warp the prediction score distribution of the language
                modeling head applied before multinomial sampling at each generation step.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return trhe hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If
                model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utils.BeamSampleDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.BeamSampleEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.BeamSampleDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.BeamSampleEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.

        Examples::

            >>> from transformers import (
            ...     AutoTokenizer,
            ...     AutoModelForSeq2SeqLM,
            ...     LogitsProcessorList,
            ...     MinLengthLogitsProcessor,
            ...     TopKLogitsWarper,
            ...     TemperatureLogitsWarper,
            ...     BeamSearchScorer,
            ... )
            >>> import torch

            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

            >>> encoder_input_str = "translate English to German: How old are you?"
            >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

            >>> # lets run beam search using 3 beams
            >>> num_beams = 3
            >>> # define decoder start token ids
            >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
            >>> input_ids = input_ids * model.config.decoder_start_token_id

            >>> # add encoder_outputs to model keyword arguments
            >>> model_kwargs = {
            ...     "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
            ... }

            >>> # instantiate beam scorer
            >>> beam_scorer = BeamSearchScorer(
            ...     batch_size=1,
            ...     max_length=model.config.max_length,
            ...     num_beams=num_beams,
            ...     device=model.device,
            ... )

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)
            ... ])
            >>> # instantiate logits processors
            >>> logits_warper = LogitsProcessorList([
            ...     TopKLogitsWarper(50),
            ...     TemperatureLogitsWarper(0.7),
            ... ])

            >>> outputs = model.beam_sample(
            ...     input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs
            ... )

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits = outputs.logits[:, -1, :]

            # adjust token scores (a no-op by default)
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

    def group_beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ):
        r"""
        Generates sequences for models with a language modeling head using beam search decoding.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            beam_scorer (:obj:`BeamScorer`):
                An derived instance of :class:`~transformers.BeamScorer` that defines how beam hypotheses are
                constructed, stored and sorted during generation. For more information, the documentation of
                :class:`~transformers.BeamScorer` should be read.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return trhe hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            model_kwargs:
                Additional model specific kwargs that will be forwarded to the :obj:`forward` function of the model. If
                model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput` if
            :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.

        Examples::

            >>> from transformers import (
            ...    AutoTokenizer,
            ...    AutoModelForSeq2SeqLM,
            ...    LogitsProcessorList,
            ...    MinLengthLogitsProcessor,
            ...    HammingDiversityLogitsProcessor,
            ...    BeamSearchScorer,
            ... )
            >>> import torch

            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

            >>> encoder_input_str = "translate English to German: How old are you?"
            >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


            >>> # lets run diverse beam search using 6 beams
            >>> num_beams = 6
            >>> # define decoder start token ids
            >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
            >>> input_ids = input_ids * model.config.decoder_start_token_id

            >>> # add encoder_outputs to model keyword arguments
            >>> model_kwargs = {
            ...     "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
            ... }

            >>> # instantiate beam scorer
            >>> beam_scorer = BeamSearchScorer(
            ...     batch_size=1,
            ...     max_length=model.config.max_length,
            ...     num_beams=num_beams,
            ...     device=model.device,
            ...     num_beam_groups=3
            ... )

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     HammingDiversityLogitsProcessor(5.5, num_beams=6, num_beam_groups=3),
            ...     MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
            ... ])

            >>> outputs = model.group_beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        device = input_ids.device

        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == batch_beam_size
        ), f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while cur_len < max_length:
            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            # do one decoder step on all beams of all sentences in batch
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                if output_scores:
                    processed_score = torch.zeros_like(outputs.logits[:, -1, :])

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of current group only
                next_token_logits = outputs.logits[batch_group_indices, -1, :]

                # adjust tokens for Bart, *e.g.*
                next_token_logits = self.adjust_logits_during_generation(
                    next_token_logits, cur_len=cur_len, max_length=max_length
                )

                next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * group_size, vocab_size)
                vocab_size = next_token_scores.shape[-1]

                next_token_scores = logits_processor(
                    group_input_ids, next_token_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores + beam_scores[batch_group_indices].unsqueeze(-1).expand_as(
                    next_token_scores
                )

                if output_scores:
                    processed_score[batch_group_indices] = next_token_scores

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = next_tokens // vocab_size
                next_tokens = next_tokens % vocab_size

                # stateless
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * (beam_idx // group_size) + group_start_idx + (beam_idx % group_size)
                )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (processed_score,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], reordering_indices)

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1
            if beam_scorer.is_done:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"]
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]


def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)(None, logits)

    return logits