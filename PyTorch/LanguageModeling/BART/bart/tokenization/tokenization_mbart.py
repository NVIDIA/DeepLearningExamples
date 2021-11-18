# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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

from typing import List, Optional

from utils.file_utils import add_start_docstrings
from bart.tokenization.tokenization_utils import BatchEncoding
from bart.tokenization.tokenization_utils_base import PREPARE_SEQ2SEQ_BATCH_DOCSTRING
from bart.tokenization.tokenization_xlm_roberta import XLMRobertaTokenizer
from utils import logging


logger = logging.get_logger(__name__)

_all_mbart_models = ["facebook/mbart-large-en-ro", "facebook/mbart-large-cc25"]
SPM_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/mbart-large-en-ro/sentence.bpe.model"

FAIRSEQ_LANGUAGE_CODES = [
    "ar_AR",
    "cs_CZ",
    "de_DE",
    "en_XX",
    "es_XX",
    "et_EE",
    "fi_FI",
    "fr_XX",
    "gu_IN",
    "hi_IN",
    "it_IT",
    "ja_XX",
    "kk_KZ",
    "ko_KR",
    "lt_LT",
    "lv_LV",
    "my_MM",
    "ne_NP",
    "nl_XX",
    "ro_RO",
    "ru_RU",
    "si_LK",
    "tr_TR",
    "vi_VN",
    "zh_CN",
]


class MBartTokenizer(XLMRobertaTokenizer):
    """
    This inherits from XLMRobertaTokenizer. ``prepare_seq2seq_batch`` should be used to encode inputs.
    Other tokenizer methods like ``encode`` do not work properly.
    The tokenization method is ``<tokens> <eos> <language code>`` for source language documents, and
    ``<language code> <tokens> <eos>``` for target language documents.

    Examples::

        >>> from transformers import MBartTokenizer
        >>> tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-en-ro')
        >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
        >>> expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"
        >>> batch: dict = tokenizer.prepare_seq2seq_batch(
        ...     example_english_phrase, src_lang="en_XX", tgt_lang="ro_RO", tgt_texts=expected_translation_romanian
        ... )

    """

    vocab_files_names = {"vocab_file": "sentencepiece.bpe.model"}
    max_model_input_sizes = {m: 1024 for m in _all_mbart_models}
    pretrained_vocab_files_map = {"vocab_file": {m: SPM_URL for m in _all_mbart_models}}

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sp_model_size = len(self.sp_model)
        self.lang_code_to_id = {
            code: self.sp_model_size + i + self.fairseq_offset for i, code in enumerate(FAIRSEQ_LANGUAGE_CODES)
        }
        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}
        self.cur_lang_code = self.lang_code_to_id["en_XX"]
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset

        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}
        self._additional_special_tokens = list(self.lang_code_to_id.keys())
        self.set_src_lang_special_tokens(kwargs.get("src_lang", "en_XX"))

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` methods.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))
        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens. The special tokens depend on calling set_lang.
        An MBART sequence has the following format, where ``X`` represents the sequence:
        - ``input_ids`` (for encoder) ``X [eos, src_lang_code]``
        - ``decoder_input_ids``: (for decoder) ``[tgt_lang_code] X [eos]``
        BOS is never used.
        Pairs of sequences are not the expected use case, but they will be handled without a separator.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    @add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "ro_RO",
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        truncation: bool = True,
        padding: str = "longest",
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchEncoding:
        """Prepare a batch that can be passed directly to an instance of MBartModel.

        Arguments:
            src_texts: (:obj:`list`):
                list of documents to summarize or source language texts
            src_lang: (:obj:`str`, `optional`, default='en_XX'):
                default en_XX (english), the language we are translating from
            tgt_texts: (:obj:`list`, `optional`):
                list of tgt language texts or summaries.
            tgt_lang: (:obj:`str`, `optional`, default='ro_RO'):
                default ro_RO (romanian), the language we are translating to
            max_length (:obj:`int`, `optional`):
                Controls the maximum length for encoder inputs (documents to summarize or source language texts)
                If left unset or set to :obj:`None`, this will use the predefined model maximum length if a maximum
                length is required by one of the truncation/padding parameters. If the model has no specific maximum
                input length (like XLNet) truncation/padding to a maximum length will be deactivated.
            max_target_length (:obj:`int`, `optional`):
                Controls the maximum length of decoder inputs (target language texts or summaries)
                If left unset or set to :obj:`None`, this will use the max_length value.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls padding. Accepts the following values:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            return_tensors (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`, defaults to "pt"):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            truncation (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.TruncationStrategy`, `optional`, defaults to :obj:`True`):
                Activates and controls truncation. Accepts the following values:

                * :obj:`True` or :obj:`'longest_first'`: Truncate to a maximum length specified with the argument
                  :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not
                  provided. This will truncate token by token, removing a token from the longest sequence in the pair
                  if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_first'`: Truncate to a maximum length specified with the argument :obj:`max_length` or to
                  the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_second'`: Truncate to a maximum length specified with the argument :obj:`max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`False` or :obj:`'do_not_truncate'` (default): No truncation (i.e., can output batch with
                  sequence lengths greater than the model maximum admissible input size).

        Return:
            :class:`~transformers.BatchEncoding`: A :class:`~transformers.BatchEncoding` with the following fields:

            - **input_ids** -- List of token ids to be fed to the encoder.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
            - **labels** -- List of token ids for tgt_texts

            The full set of keys ``[input_ids, attention_mask, decoder_input_ids,  labels]``,
            will only be returned if tgt_texts is passed. Otherwise, input_ids, attention_mask will be the only keys.

        """
        if max_length is None:
            max_length = self.max_len
        self.set_src_lang_special_tokens(src_lang)
        model_inputs: BatchEncoding = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        if tgt_texts is None:
            return model_inputs
        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        self.set_tgt_lang_special_tokens(tgt_lang)

        labels = self(
            tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_target_length,
            truncation=True,
            **kwargs,
        )["input_ids"]
        model_inputs["labels"] = labels
        self.set_src_lang_special_tokens(src_lang)  # sets to src_lang
        return model_inputs

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, cur_lang_code]."""
        self.cur_lang_code = self.lang_code_to_id[src_lang]
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. Prefix [tgt_lang_code], suffix =[eos]."""
        self.cur_lang_code = self.lang_code_to_id[lang]
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]