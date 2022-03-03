# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
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
"""Tokenization classes for python and fast tokenizers. Fast tokenizers are provided by HuggingFace's tokenizers library."""

import copy
import functools
import itertools
import json
import logging
import operator
import os
import re
import warnings
import unicodedata
import collections

from collections import UserDict, defaultdict
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, List, MutableMapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from tokenizers import AddedToken as AddedTokenFast
from tokenizers import Encoding as EncodingFast
from tokenizers import BertWordPieceTokenizer
from tokenizers.decoders import Decoder as DecoderFast
from tokenizers.implementations import BaseTokenizer as BaseTokenizerFast

from file_utils import cached_path, hf_bucket_url, is_remote_url, is_tf_available, is_torch_available, torch_required

if is_tf_available():
    import tensorflow as tf
if is_torch_available():
    import torch

logger = logging.getLogger(__name__)

NO_PAD_TOKEN_FOR_BATCH_MSG = (
    "No padding token is set for this model, therefore no batch can be made with uneven "
    "sequences. Set a padding token or adjust the lengths of the sequences building the "
    "batch so that every sequence is of the same length."
)

UNEVEN_SEQUENCES_FOR_BATCH_MSG = (
    "The sequences building the batch are not of the same size, no tensor "
    "can be built. Set `pad_to_max_length=True` to pad the smaller sequences"
    "up to the larger sequence's length."
)

SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

VERY_LARGE_INTEGER = int(1e30)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(1e20)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]


class TensorType(Enum):
    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"


class CharSpan(NamedTuple):
    """ Character span in the original string

        Args:
            start: index of the first character in the original string
            end: index of the character following the last character in the original string
    """

    start: int
    end: int


class TokenSpan(NamedTuple):
    """ Token span in an encoded string (list of tokens)

        Args:
            start: index of the first token in the span
            end: index of the token following the last token in the span
    """

    start: int
    end: int


def flatten(x: Sequence):
    """
    Flatten the provided (potentially nested) sequence

    Args:
        x (Sequence): Potentially nested sequence to flatten

    Returns:
        list: Flattened sequence
    """

    return functools.reduce(operator.iconcat, x, [])


@contextmanager
def truncate_and_pad(
    tokenizer: BaseTokenizerFast,
    max_length: int,
    stride: int,
    strategy: str,
    pad_to_max_length: bool,
    padding_side: str,
    pad_token_id: int,
    pad_token_type_id: int,
    pad_token: str,
):
    """ This contextmanager is in charge of defining the truncation and the padding strategies for fast tokenizers
        (provided by HuggingFace tokenizers library) and restore the tokenizer settings afterwards.

        This contextmanager assumes the provider tokenizer has no padding / truncation strategy
        before the managed section. If your tokenizer set a padding / truncation strategy before,
        then it will be reset to no padding/truncation when exiting the managed section.

        Args:
            tokenizer (BaseTokenizerFast): The tokenizer which will be used
            max_length (int): The maximum size of the sequence
            stride (int): The stride to use when handling overflow
            strategy (str): Overflowing logic to use
            pad_to_max_length (bool): Boolean indicating if the output needs to be padded up to max_length
            padding_side (str): "left" or "right" indicating the direction the output sequence will be padded
            pad_token_id (int): The integer representation of the padding token to use
            pad_token_type_id (int): The integer representation of the padding token type to use
            pad_token (str): The string representation of the padding token to use

    """

    # Handle all the truncation and padding stuff
    if max_length is not None:
        tokenizer.enable_truncation(max_length, stride=stride, strategy=strategy)

    if pad_to_max_length and (pad_token and pad_token_id >= 0):
        tokenizer.enable_padding(
            max_length=max_length,
            direction=padding_side,
            pad_id=pad_token_id,
            pad_type_id=pad_token_type_id,
            pad_token=pad_token,
        )
    elif pad_to_max_length:
        logger.warning(
            "Disabled padding because no padding token set (pad_token: {}, pad_token_id: {}).\n"
            "To remove this error, you can add a new pad token and then resize model embedding:\n"
            "\ttokenizer.pad_token = '<PAD>'\n\tmodel.resize_token_embeddings(len(tokenizer))".format(
                pad_token, pad_token_id
            )
        )

    yield

    # TODO(morgan, anthony): once we have a simple way to serialize tokenizers maybe store and restore the state afterward
    # to avoid destructing the padding / truncation strategy as we do now.

    if max_length is not None:
        tokenizer.no_truncation()

    if pad_to_max_length and (pad_token and pad_token_id >= 0):
        tokenizer.no_padding()


def convert_to_tensors(
    batch_outputs: MutableMapping, return_tensors: Union[str, TensorType], prepend_batch_axis: bool = False
) -> MutableMapping:
    # Convert to TensorType
    if not isinstance(return_tensors, TensorType):
        return_tensors = TensorType(return_tensors)

    # Get a function reference for the correct framework
    if return_tensors == TensorType.TENSORFLOW and is_tf_available():
        as_tensor = tf.constant
    elif return_tensors == TensorType.PYTORCH and is_torch_available():
        as_tensor = torch.tensor
    elif return_tensors == TensorType.NUMPY:
        as_tensor = np.asarray
    else:
        raise ImportError(
            "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                return_tensors
            )
        )

    # Do the tensor conversion in batch
    for key, value in batch_outputs.items():
        try:
            if prepend_batch_axis:
                value = [value]

            tensor = as_tensor(value)

            # at-least2d
            if tensor.ndim > 2:
                tensor = tensor.squeeze(0)
            elif tensor.ndim < 2:
                tensor = tensor[None, :]

            batch_outputs[key] = tensor
        except ValueError:
            if None in [item for sequence in value for item in sequence]:
                raise ValueError(NO_PAD_TOKEN_FOR_BATCH_MSG)
            else:
                raise ValueError(UNEVEN_SEQUENCES_FOR_BATCH_MSG)

    return batch_outputs


class BatchEncoding(UserDict):
    """ BatchEncoding hold the output of the encode and batch_encode methods (tokens, attention_masks, etc).
        This class is derived from a python Dictionary and can be used as a dictionnary.
        In addition, this class expose utility methods to map from word/char space to token space.

        Args:
            data (:obj:`dict`): Dictionary of lists/arrays returned by the encode/batch_encode methods ('input_ids', 'attention_mask'...)
            encoding (:obj:`EncodingFast`, :obj:`list(EncodingFast)`, `optional`, defaults to :obj:`None`):
                If the tokenizer is a fast tokenizer which outputs additional informations like mapping from word/char space to token space
                the `EncodingFast` instance or list of instance (for batches) hold these informations.

    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        encoding: Optional[Union[EncodingFast, Sequence[EncodingFast]]] = None,
    ):
        super().__init__(data)

        if isinstance(encoding, EncodingFast):
            encoding = [encoding]

        self._encodings = encoding

    def __getitem__(self, item: Union[int, str]) -> EncodingFast:
        """ If the key is a string, get the value of the dict associated to `key` ('input_ids', 'attention_mask'...)
            If the key is an integer, get the EncodingFast for batch item with index `key`
        """
        if isinstance(item, str):
            return self.data[item]
        elif self._encodings is not None:
            return self._encodings[item]
        else:
            raise KeyError(
                "Indexing with integers (to access backend Encoding for a given batch index) "
                "is not available when using Python based tokenizers"
            )

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    # After this point:
    # Extended properties and methods only available for fast (Rust-based) tokenizers
    # provided by HuggingFace tokenizers library.

    @property
    def encodings(self) -> Optional[List[EncodingFast]]:
        """
        Return the list all encoding from the tokenization process

        Returns: List[EncodingFast] or None if input was tokenized through Python (i.e. not fast) tokenizer
        """
        return self._encodings

    def tokens(self, batch_index: int = 0) -> List[int]:
        if not self._encodings:
            raise ValueError("tokens() is not available when using Python based tokenizers")
        return self._encodings[batch_index].tokens

    def words(self, batch_index: int = 0) -> List[Optional[int]]:
        if not self._encodings:
            raise ValueError("words() is not available when using Python based tokenizers")
        return self._encodings[batch_index].words

    def token_to_word(self, batch_or_token_index: int, token_index: Optional[int] = None) -> int:
        """ Get the index of the word corresponding (i.e. comprising) to an encoded token
            in a sequence of the batch.

            Can be called as:
                - self.token_to_word(token_index) if batch size is 1
                - self.token_to_word(batch_index, token_index) if batch size is greater than 1

            This method is particularly suited when the input sequences are provided as
            pre-tokenized sequences (i.e. words are defined by the user). In this case it allows
            to easily associate encoded tokens with provided tokenized words.

        Args:
            batch_or_token_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence,
                this can be the index of the token in the sequence
            token_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index
                of the token in the sequence.

        Returns:
            word_index (:obj:`int`):
                index of the word in the input sequence.

        """

        if not self._encodings:
            raise ValueError("token_to_word() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_word(token_index)

    def word_to_tokens(self, batch_or_word_index: int, word_index: Optional[int] = None) -> TokenSpan:
        """ Get the encoded token span corresponding to a word in the sequence of the batch.

            Token spans are returned as a TokenSpan NamedTuple with:
                start: index of the first token
                end: index of the token following the last token

            Can be called as:
                - self.word_to_tokens(word_index) if batch size is 1
                - self.word_to_tokens(batch_index, word_index) if batch size is greater or equal to 1

            This method is particularly suited when the input sequences are provided as
            pre-tokenized sequences (i.e. words are defined by the user). In this case it allows
            to easily associate encoded tokens with provided tokenized words.

        Args:
            batch_or_word_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence,
                this can be the index of the word in the sequence
            word_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index
                of the word in the sequence.

        Returns:
            token_span (:obj:`TokenSpan`):
                Span of tokens in the encoded sequence.

                TokenSpan are NamedTuple with:
                    start: index of the first token
                    end: index of the token following the last token
        """

        if not self._encodings:
            raise ValueError("word_to_tokens() is not available when using Python based tokenizers")
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if word_index < 0:
            word_index = self._seq_len + word_index
        return TokenSpan(*(self._encodings[batch_index].word_to_tokens(word_index)))

    def token_to_chars(self, batch_or_token_index: int, token_index: Optional[int] = None) -> CharSpan:
        """ Get the character span corresponding to an encoded token in a sequence of the batch.

            Character spans are returned as a CharSpan NamedTuple with:
                start: index of the first character in the original string associated to the token
                end: index of the character following the last character in the original string associated to the token

            Can be called as:
                - self.token_to_chars(token_index) if batch size is 1
                - self.token_to_chars(batch_index, token_index) if batch size is greater or equal to 1

        Args:
            batch_or_token_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence,
                this can be the index of the token in the sequence
            token_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index
                of the token or tokens in the sequence.

        Returns:
            char_span (:obj:`CharSpan`):
                Span of characters in the original string.

                CharSpan are NamedTuple with:
                    start: index of the first character in the original string
                    end: index of the character following the last character in the original string
        """

        if not self._encodings:
            raise ValueError("token_to_chars() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        return CharSpan(*(self._encodings[batch_index].token_to_chars(token_index)))

    def char_to_token(self, batch_or_char_index: int, char_index: Optional[int] = None) -> int:
        """ Get the index of the token in the encoded output comprising a character
            in the original string for a sequence of the batch.

            Can be called as:
                - self.char_to_token(char_index) if batch size is 1
                - self.char_to_token(batch_index, char_index) if batch size is greater or equal to 1

            This method is particularly suited when the input sequences are provided as
            pre-tokenized sequences (i.e. words are defined by the user). In this case it allows
            to easily associate encoded tokens with provided tokenized words.

        Args:
            batch_or_char_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence,
                this can be the index of the word in the sequence
            char_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index
                of the word in the sequence.


        Returns:
            token_index (:obj:`int`):
                Index of the token.
        """

        if not self._encodings:
            raise ValueError("char_to_token() is not available when using Python based tokenizers")
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_token(char_index)

    def word_to_chars(self, batch_or_word_index: int, word_index: Optional[int] = None) -> CharSpan:
        """ Get the character span in the original string corresponding to given word in a sequence
            of the batch.

            Character spans are returned as a CharSpan NamedTuple with:
                start: index of the first character in the original string
                end: index of the character following the last character in the original string

            Can be called as:
                - self.word_to_chars(word_index) if batch size is 1
                - self.word_to_chars(batch_index, word_index) if batch size is greater or equal to 1

        Args:
            batch_or_word_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence,
                this can be the index of the word in the sequence
            word_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index
                of the word in the sequence.

        Returns:
            char_span (:obj:`CharSpan` or :obj:`List[CharSpan]`):
                Span(s) of the associated character or characters in the string.
                CharSpan are NamedTuple with:
                    start: index of the first character associated to the token in the original string
                    end: index of the character following the last character associated to the token in the original string
        """

        if not self._encodings:
            raise ValueError("word_to_chars() is not available when using Python based tokenizers")
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        return CharSpan(*(self._encodings[batch_index].word_to_chars(word_index)))

    def char_to_word(self, batch_or_char_index: int, char_index: Optional[int] = None) -> int:
        """ Get the word in the original string corresponding to a character in the original string of
            a sequence of the batch.

            Can be called as:
                - self.char_to_word(char_index) if batch size is 1
                - self.char_to_word(batch_index, char_index) if batch size is greater than 1

            This method is particularly suited when the input sequences are provided as
            pre-tokenized sequences (i.e. words are defined by the user). In this case it allows
            to easily associate encoded tokens with provided tokenized words.

        Args:
            batch_or_char_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence,
                this can be the index of the character in the orginal string.
            char_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index
                of the character in the orginal string.


        Returns:
            token_index (:obj:`int` or :obj:`List[int]`):
                Index or indices of the associated encoded token(s).
        """

        if not self._encodings:
            raise ValueError("char_to_word() is not available when using Python based tokenizers")
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_word(char_index)

    @torch_required
    def to(self, device: str):
        """Send all values to device by calling v.to(device)"""
        self.data = {k: v.to(device) for k, v in self.data.items()}
        return self


class SpecialTokensMixin:
    """ SpecialTokensMixin is derived by ``PreTrainedTokenizer`` and ``PreTrainedTokenizerFast`` and
        handles specific behaviors related to special tokens. In particular, this class hold the
        attributes which can be used to directly access to these special tokens in a
        model-independant manner and allow to set and update the special tokens.
    """

    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]

    def __init__(self, **kwargs):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0
        self._additional_special_tokens = []

        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)) and all(isinstance(t, str) for t in value)
                    setattr(self, key, value)
                elif isinstance(value, AddedTokenFast):
                    setattr(self, key, str(value))
                elif isinstance(value, str):
                    setattr(self, key, value)
                else:
                    raise TypeError(
                        "special token {} has to be either str or AddedTokenFast but got: {}".format(key, type(value))
                    )

    @property
    def bos_token(self):
        """ Beginning of sentence token (string). Log an error if used while not having been set. """
        if self._bos_token is None:
            logger.error("Using bos_token, but it is not set yet.")
        return self._bos_token

    @property
    def eos_token(self):
        """ End of sentence token (string). Log an error if used while not having been set. """
        if self._eos_token is None:
            logger.error("Using eos_token, but it is not set yet.")
        return self._eos_token

    @property
    def unk_token(self):
        """ Unknown token (string). Log an error if used while not having been set. """
        if self._unk_token is None:
            logger.error("Using unk_token, but it is not set yet.")
        return self._unk_token

    @property
    def sep_token(self):
        """ Separation token (string). E.g. separate context and query in an input sequence. Log an error if used while not having been set. """
        if self._sep_token is None:
            logger.error("Using sep_token, but it is not set yet.")
        return self._sep_token

    @property
    def pad_token(self):
        """ Padding token (string). Log an error if used while not having been set. """
        if self._pad_token is None:
            logger.error("Using pad_token, but it is not set yet.")
        return self._pad_token

    @property
    def cls_token(self):
        """ Classification token (string). E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model. Log an error if used while not having been set. """
        if self._cls_token is None:
            logger.error("Using cls_token, but it is not set yet.")
        return self._cls_token

    @property
    def mask_token(self):
        """ Mask token (string). E.g. when training a model with masked-language modeling. Log an error if used while not having been set. """
        if self._mask_token is None:
            logger.error("Using mask_token, but it is not set yet.")
        return self._mask_token

    @property
    def additional_special_tokens(self):
        """ All the additional special tokens you may want to use (list of strings). Log an error if used while not having been set. """
        if self._additional_special_tokens is None:
            logger.error("Using additional_special_tokens, but it is not set yet.")
        return self._additional_special_tokens

    def _maybe_update_backend(self, value):
        """ To be overriden by derived class if a backend tokenizer has to be updated. """
        pass

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value
        self._maybe_update_backend([value])

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value
        self._maybe_update_backend([value])

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value
        self._maybe_update_backend([value])

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value
        self._maybe_update_backend([value])

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        self._maybe_update_backend([value])

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value
        self._maybe_update_backend([value])

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value
        self._maybe_update_backend([value])

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value
        self._maybe_update_backend(value)

    @property
    def bos_token_id(self):
        """ Id of the beginning of sentence token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self):
        """ Id of the end of sentence token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self):
        """ Id of the unknown token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self):
        """ Id of the separation token in the vocabulary. E.g. separate context and query in an input sequence. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self):
        """ Id of the padding token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self):
        """ Id of the padding token type in the vocabulary."""
        return self._pad_token_type_id

    @property
    def cls_token_id(self):
        """ Id of the classification token in the vocabulary. E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self):
        """ Id of the mask token in the vocabulary. E.g. when training a model with masked-language modeling. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def additional_special_tokens_ids(self):
        """ Ids of all the additional special tokens in the vocabulary (list of integers). Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    @property
    def special_tokens_map(self):
        """ A dictionary mapping special token class attribute (cls_token, unk_token...) to their
            values ('<unk>', '<cls>'...)
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self):
        """ List all the special tokens ('<unk>', '<cls>'...) mapped to class attributes
            (cls_token, unk_token...).
        """
        all_toks = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(set(all_toks))
        return all_toks

    @property
    def all_special_ids(self):
        """ List the vocabulary indices of the special tokens ('<unk>', '<cls>'...) mapped to
            class attributes (cls_token, unk_token...).
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids


class PreTrainedTokenizer(SpecialTokensMixin):
    """ Base class for all tokenizers.

    Handle all the shared methods for tokenization and special tokens as well as methods
    downloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary.

    This class also contain the added tokens in a unified way on top of all tokenizers so we don't
    have to handle the specific vocabulary augmentation methods of the various underlying
    dictionary structures (BPE, sentencepiece...).

    Class attributes (overridden by derived classes):

        - ``vocab_files_names``: a python ``dict`` with, as keys, the ``__init__`` keyword name of each vocabulary file
            required by the model, and as associated values, the filename for saving the associated file (string).
        - ``pretrained_vocab_files_map``: a python ``dict of dict`` the high-level keys
            being the ``__init__`` keyword name of each vocabulary file required by the model, the low-level being the
            `short-cut-names` (string) of the pretrained models with, as associated values, the `url` (string) to the
            associated pretrained vocabulary file.
        - ``max_model_input_sizes``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the pretrained
            models, and as associated values, the maximum length of the sequence inputs of this model, or None if the
            model has no maximum input size.
        - ``pretrained_init_configuration``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the
            pretrained models, and as associated values, a dictionnary of specific arguments to pass to the
            ``__init__``method of the tokenizer class for this pretrained model when loading the tokenizer with the
            ``from_pretrained()`` method.

    Args:
        - ``model_max_length``: (`Optional`) int: the maximum length in number of tokens for the inputs to the transformer model.
            When the tokenizer is loaded with `from_pretrained`, this will be set to the value stored for the associated
            model in ``max_model_input_sizes`` (see above). If no value is provided, will default to VERY_LARGE_INTEGER (`int(1e30)`).
            no associated max_length can be found in ``max_model_input_sizes``.
        - ``padding_side``: (`Optional`) string: the side on which the model should have padding applied.
            Should be selected between ['right', 'left']
        - ``model_input_names``: (`Optional`) List[string]: the list of the forward pass inputs accepted by the
            model ("token_type_ids", "attention_mask"...).
        - ``bos_token``: (`Optional`) string: a beginning of sentence token.
            Will be associated to ``self.bos_token`` and ``self.bos_token_id``
        - ``eos_token``: (`Optional`) string: an end of sentence token.
            Will be associated to ``self.eos_token`` and ``self.eos_token_id``
        - ``unk_token``: (`Optional`) string: an unknown token.
            Will be associated to ``self.unk_token`` and ``self.unk_token_id``
        - ``sep_token``: (`Optional`) string: a separation token (e.g. to separate context and query in an input sequence).
            Will be associated to ``self.sep_token`` and ``self.sep_token_id``
        - ``pad_token``: (`Optional`) string: a padding token.
            Will be associated to ``self.pad_token`` and ``self.pad_token_id``
        - ``cls_token``: (`Optional`) string: a classification token (e.g. to extract a summary of an input sequence
            leveraging self-attention along the full depth of the model).
            Will be associated to ``self.cls_token`` and ``self.cls_token_id``
        - ``mask_token``: (`Optional`) string: a masking token (e.g. when training a model with masked-language
            modeling). Will be associated to ``self.mask_token`` and ``self.mask_token_id``
        - ``additional_special_tokens``: (`Optional`) list: a list of additional special tokens.
            Adding all special tokens here ensure they won't be split by the tokenization process.
            Will be associated to ``self.additional_special_tokens`` and ``self.additional_special_tokens_ids``
    """

    vocab_files_names: Dict[str, str] = {}
    pretrained_vocab_files_map: Dict[str, Dict[str, str]] = {}
    pretrained_init_configuration: Dict[str, Dict[str, Any]] = {}
    max_model_input_sizes: Dict[str, int] = {}
    model_input_names: List[str] = ["token_type_ids", "attention_mask"]

    padding_side: str = "right"

    @property
    def vocab_size(self) -> int:
        """ Size of the base vocabulary (without the added tokens) """
        raise NotImplementedError

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def max_len(self) -> int:
        """ Kept here for backward compatibility.
            Now renamed to `model_max_length` to avoid ambiguity.
        """
        return self.model_max_length

    @property
    def max_len_single_sentence(self) -> int:
        return self.model_max_length - self.num_special_tokens_to_add(pair=False)

    @property
    def max_len_sentences_pair(self) -> int:
        return self.model_max_length - self.num_special_tokens_to_add(pair=True)

    @max_len_single_sentence.setter
    def max_len_single_sentence(self, value) -> int:
        """ For backward compatibility, allow to try to setup 'max_len_single_sentence' """
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=False):
            logger.warning(
                "Setting 'max_len_single_sentence' is now deprecated. " "This value is automatically set up."
            )
        else:
            raise ValueError(
                "Setting 'max_len_single_sentence' is now deprecated. " "This value is automatically set up."
            )

    @max_len_sentences_pair.setter
    def max_len_sentences_pair(self, value) -> int:
        """ For backward compatibility, allow to try to setup 'max_len_sentences_pair' """
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=True):
            logger.warning(
                "Setting 'max_len_sentences_pair' is now deprecated. " "This value is automatically set up."
            )
        else:
            raise ValueError(
                "Setting 'max_len_sentences_pair' is now deprecated. " "This value is automatically set up."
            )

    def get_vocab(self):
        """ Returns the vocabulary as a dict of {token: index} pairs. `tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the vocab. """
        raise NotImplementedError()

    def __init__(self, model_max_length=None, **kwargs):

        super().__init__(**kwargs)

        # For backward compatibility we fallback to set model_max_length from max_len if provided
        if "max_len" in kwargs:
            warnings.warn(
                "Parameter max_len is deprecated and will be removed in a future release. "
                "Use model_max_length instead.",
                category=FutureWarning,
            )

            model_max_length = kwargs.pop("max_len")
        self.model_max_length = model_max_length if model_max_length is not None else VERY_LARGE_INTEGER

        # Padding side is right by default and overridden in subclasses. If specified in the kwargs, it is changed.
        self.padding_side = kwargs.pop("padding_side", self.padding_side)
        assert self.padding_side in [
            "right",
            "left",
        ], f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
        self.model_input_names = kwargs.pop("model_input_names", self.model_input_names)

        # Added tokens
        self.added_tokens_encoder = {}
        self.unique_added_tokens_encoder = set()
        self.added_tokens_decoder = {}

        # inputs and kwargs for saving and re-loading (see ``from_pretrained`` and ``save_pretrained``)
        self.init_inputs = ()
        self.init_kwargs = {}

    def __len__(self):
        """ Size of the full vocabulary with the added tokens """
        return self.vocab_size + len(self.added_tokens_encoder)

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        r"""
        Instantiate a :class:`~transformers.PreTrainedTokenizer` (or a derived class) from a predefined tokenizer.

        Args:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a predefined tokenizer to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a predefined tokenizer that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing vocabulary files required by the tokenizer, for instance saved using the :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g.: ``./my_model_directory/``.
                - (not applicable to all derived classes, deprecated) a path or url to a single saved vocabulary file if and only if the tokenizer only requires a single vocabulary file (e.g. Bert, XLNet), e.g.: ``./my_model_directory/vocab.txt``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the vocabulary files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            inputs: (`optional`) positional arguments: will be passed to the Tokenizer ``__init__`` method.

            kwargs: (`optional`) keyword arguments: will be passed to the Tokenizer ``__init__`` method. Can be used to set special tokens like ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``, ``additional_special_tokens``. See parameters in the doc string of :class:`~transformers.PreTrainedTokenizer` for details.

        Examples::

            # We can't instantiate directly the base class `PreTrainedTokenizer` so let's show our examples on a derived class: BertTokenizer

            # Download vocabulary from S3 and cache.
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            # Download vocabulary from S3 (user-uploaded) and cache.
            tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-german-cased')

            # If vocabulary files are in a directory (e.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`)
            tokenizer = BertTokenizer.from_pretrained('./test/saved_model/')

            # If the tokenizer uses a single vocabulary file, you can point directly to this file
            tokenizer = BertTokenizer.from_pretrained('./test/saved_model/my_vocab.txt')

            # You can link tokens to special vocabulary when instantiating
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', unk_token='<unk>')
            # You should be sure '<unk>' is in the vocabulary when doing that.
            # Otherwise use tokenizer.add_special_tokens({'unk_token': '<unk>'}) instead)
            assert tokenizer.unk_token == '<unk>'

        """
        return cls._from_pretrained(*inputs, **kwargs)

    @classmethod
    def _from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)

        s3_models = list(cls.max_model_input_sizes.keys())
        vocab_files = {}
        init_configuration = {}
        if pretrained_model_name_or_path in s3_models:
            # Get the vocabulary from AWS S3 bucket
            for file_id, map_list in cls.pretrained_vocab_files_map.items():
                vocab_files[file_id] = map_list[pretrained_model_name_or_path]
            if (
                cls.pretrained_init_configuration
                and pretrained_model_name_or_path in cls.pretrained_init_configuration
            ):
                init_configuration = cls.pretrained_init_configuration[pretrained_model_name_or_path].copy()
        else:
            # Get the vocabulary from local files
            logger.info(
                "Model name '{}' not found in model shortcut name list ({}). "
                "Assuming '{}' is a path, a model identifier, or url to a directory containing tokenizer files.".format(
                    pretrained_model_name_or_path, ", ".join(s3_models), pretrained_model_name_or_path
                )
            )

            if os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                if len(cls.vocab_files_names) > 1:
                    raise ValueError(
                        f"Calling {cls.__name__}.from_pretrained() with the path to a single file or url is not supported."
                        "Use a model identifier or the path to a directory instead."
                    )
                logger.warning(
                    f"Calling {cls.__name__}.from_pretrained() with the path to a single file or url is deprecated"
                )
                file_id = list(cls.vocab_files_names.keys())[0]
                vocab_files[file_id] = pretrained_model_name_or_path
            else:
                # At this point pretrained_model_name_or_path is either a directory or a model identifier name
                additional_files_names = {
                    "added_tokens_file": ADDED_TOKENS_FILE,
                    "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,
                    "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
                }
                # Look for the tokenizer main vocabulary files + the additional tokens files
                for file_id, file_name in {**cls.vocab_files_names, **additional_files_names}.items():
                    if os.path.isdir(pretrained_model_name_or_path):
                        full_file_name = os.path.join(pretrained_model_name_or_path, file_name)
                        if not os.path.exists(full_file_name):
                            logger.info("Didn't find file {}. We won't load it.".format(full_file_name))
                            full_file_name = None
                    else:
                        full_file_name = hf_bucket_url(
                            pretrained_model_name_or_path, filename=file_name, use_cdn=False
                        )

                    vocab_files[file_id] = full_file_name

        # Get files from url, cache, or disk depending on the case
        try:
            resolved_vocab_files = {}
            for file_id, file_path in vocab_files.items():
                if file_path is None:
                    resolved_vocab_files[file_id] = None
                else:
                    resolved_vocab_files[file_id] = cached_path(
                        file_path,
                        cache_dir=cache_dir
                    )
        except EnvironmentError:
            if pretrained_model_name_or_path in s3_models:
                msg = "Couldn't reach server at '{}' to download vocabulary files."
            else:
                msg = (
                    "Model name '{}' was not found in tokenizers model name list ({}). "
                    "We assumed '{}' was a path or url to a directory containing vocabulary files "
                    "named {}, but couldn't find such vocabulary files at this path or url.".format(
                        pretrained_model_name_or_path,
                        ", ".join(s3_models),
                        pretrained_model_name_or_path,
                        list(cls.vocab_files_names.values()),
                    )
                )

            raise EnvironmentError(msg)

        if all(full_file_name is None for full_file_name in resolved_vocab_files.values()):
            raise EnvironmentError(
                "Model name '{}' was not found in tokenizers model name list ({}). "
                "We assumed '{}' was a path, a model identifier, or url to a directory containing vocabulary files "
                "named {} but couldn't find such vocabulary files at this path or url.".format(
                    pretrained_model_name_or_path,
                    ", ".join(s3_models),
                    pretrained_model_name_or_path,
                    list(cls.vocab_files_names.values()),
                )
            )

        for file_id, file_path in vocab_files.items():
            if file_path == resolved_vocab_files[file_id]:
                logger.info("loading file {}".format(file_path))
            else:
                logger.info("loading file {} from cache at {}".format(file_path, resolved_vocab_files[file_id]))

        # Prepare tokenizer initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        tokenizer_config_file = resolved_vocab_files.pop("tokenizer_config_file", None)
        if tokenizer_config_file is not None:
            with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
                init_kwargs = json.load(tokenizer_config_handle)
            saved_init_inputs = init_kwargs.pop("init_inputs", ())
            if not init_inputs:
                init_inputs = saved_init_inputs
        else:
            init_kwargs = init_configuration

        # Update with newly provided kwargs
        init_kwargs.update(kwargs)

        # Set max length if needed
        if pretrained_model_name_or_path in cls.max_model_input_sizes:
            # if we're using a pretrained model, ensure the tokenizer
            # wont index sequences longer than the number of positional embeddings
            model_max_length = cls.max_model_input_sizes[pretrained_model_name_or_path]
            if model_max_length is not None and isinstance(model_max_length, (int, float)):
                init_kwargs["model_max_length"] = min(init_kwargs.get("model_max_length", int(1e30)), model_max_length)

        # Merge resolved_vocab_files arguments in init_kwargs.
        added_tokens_file = resolved_vocab_files.pop("added_tokens_file", None)
        special_tokens_map_file = resolved_vocab_files.pop("special_tokens_map_file", None)
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path
        if special_tokens_map_file is not None:
            with open(special_tokens_map_file, encoding="utf-8") as special_tokens_map_handle:
                special_tokens_map = json.load(special_tokens_map_handle)
            for key, value in special_tokens_map.items():
                if key not in init_kwargs:
                    init_kwargs[key] = value

        # Instantiate tokenizer.
        try:
            tokenizer = cls(*init_inputs, **init_kwargs)
        except OSError:
            raise OSError(
                "Unable to load vocabulary from file. "
                "Please check that the provided vocabulary is accessible and not corrupted."
            )

        # Save inputs and kwargs for saving and re-loading with ``save_pretrained``
        tokenizer.init_inputs = init_inputs
        tokenizer.init_kwargs = init_kwargs

        # update unique_added_tokens_encoder with special tokens for correct tokenization
        tokenizer.unique_added_tokens_encoder.update(set(tokenizer.all_special_tokens))

        # Add supplementary tokens.
        if added_tokens_file is not None:
            with open(added_tokens_file, encoding="utf-8") as added_tokens_handle:
                added_tok_encoder = json.load(added_tokens_handle)
            added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
            tokenizer.added_tokens_encoder.update(added_tok_encoder)
            tokenizer.added_tokens_decoder.update(added_tok_decoder)
            tokenizer.unique_added_tokens_encoder.update(set(tokenizer.added_tokens_encoder.keys()))

        return tokenizer

    def save_pretrained(self, save_directory):
        """ Save the tokenizer vocabulary files together with:
                - added tokens,
                - special-tokens-to-class-attributes-mapping,
                - tokenizer instantiation positional and keywords inputs (e.g. do_lower_case for Bert).

            Warning: This won't save modifications you may have applied to the tokenizer after the instantiation
            (e.g. modifying tokenizer.do_lower_case after creation).

            This method make sure the full tokenizer can then be re-loaded using the
            :func:`~transformers.PreTrainedTokenizer.from_pretrained` class method.
        """
        if not os.path.isdir(save_directory):
            logger.error("Saving directory ({}) should be a directory".format(save_directory))
            return

        special_tokens_map_file = os.path.join(save_directory, SPECIAL_TOKENS_MAP_FILE)
        added_tokens_file = os.path.join(save_directory, ADDED_TOKENS_FILE)
        tokenizer_config_file = os.path.join(save_directory, TOKENIZER_CONFIG_FILE)

        tokenizer_config = copy.deepcopy(self.init_kwargs)
        if len(self.init_inputs) > 0:
            tokenizer_config["init_inputs"] = copy.deepcopy(self.init_inputs)
        for file_id in self.vocab_files_names.keys():
            tokenizer_config.pop(file_id, None)

        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(tokenizer_config, ensure_ascii=False))

        with open(special_tokens_map_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.special_tokens_map, ensure_ascii=False))

        if len(self.added_tokens_encoder) > 0:
            with open(added_tokens_file, "w", encoding="utf-8") as f:
                out_str = json.dumps(self.added_tokens_encoder, ensure_ascii=False)
                f.write(out_str)

        vocab_files = self.save_vocabulary(save_directory)

        return vocab_files + (special_tokens_map_file, added_tokens_file)

    def save_vocabulary(self, save_directory) -> Tuple[str]:
        """ Save the tokenizer vocabulary to a directory. This method does *NOT* save added tokens
            and special token mappings.

            Please use :func:`~transformers.PreTrainedTokenizer.save_pretrained` `()` to save the full
            Tokenizer state if you want to reload it using the :func:`~transformers.PreTrainedTokenizer.from_pretrained`
            class method.
        """
        raise NotImplementedError

    def add_tokens(self, new_tokens: Union[str, List[str]]) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the
        vocabulary, they are added to it with indices starting from length of the current vocabulary.

        Args:
            new_tokens: string or list of string. Each string is a token to add. Tokens are only added if they are not
            already in the vocabulary (tested by checking if the tokenizer assign the index of the ``unk_token`` to them).

        Returns:
            Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to increase the vocabulary of Bert model and tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
            print('We have added', num_added_toks, 'tokens')
            model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        """
        if not new_tokens:
            return 0

        if not isinstance(new_tokens, list):
            new_tokens = [new_tokens]

        tokens_to_add = []
        for token in new_tokens:
            assert isinstance(token, str)
            if self.init_kwargs.get("do_lower_case", False) and token not in self.all_special_tokens:
                token = token.lower()
            if (
                token != self.unk_token
                and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
                and token not in tokens_to_add
            ):
                tokens_to_add.append(token)
                logger.info("Adding %s to the vocabulary", token)

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(tokens_to_add))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.unique_added_tokens_encoder = set(self.added_tokens_encoder.keys()).union(set(self.all_special_tokens))
        self.added_tokens_decoder.update(added_tok_decoder)

        return len(tokens_to_add)

    def num_special_tokens_to_add(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. Do not put this
            inside your training loop.

        Args:
            pair: Returns the number of added tokens in the case of a sequence pair if set to True, returns the
                number of added tokens in the case of a single sequence if set to False.

        Returns:
            Number of tokens added to sequences
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def add_special_tokens(self, special_tokens_dict):
        """
        Add a dictionary of special tokens (eos, pad, cls...) to the encoder and link them
        to class attributes. If special tokens are NOT in the vocabulary, they are added
        to it (indexed starting from the last index of the current vocabulary).

        Using `add_special_tokens` will ensure your special tokens can be used in several ways:

        - special tokens are carefully handled by the tokenizer (they are never split)
        - you can easily refer to special tokens using tokenizer class attributes like `tokenizer.cls_token`. This makes it easy to develop model-agnostic training and fine-tuning scripts.

        When possible, special tokens are already registered for provided pretrained models (ex: BertTokenizer cls_token is already registered to be '[CLS]' and XLM's one is also registered to be '</s>')

        Args:
            special_tokens_dict: dict of string. Keys should be in the list of predefined special attributes:
                [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
                ``additional_special_tokens``].

                Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the ``unk_token`` to them).

        Returns:
            Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to add a new classification token to GPT-2
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2Model.from_pretrained('gpt2')

            special_tokens_dict = {'cls_token': '<CLS>'}

            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print('We have added', num_added_toks, 'tokens')
            model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.

            assert tokenizer.cls_token == '<CLS>'
        """
        if not special_tokens_dict:
            return 0

        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES
            if key == "additional_special_tokens":
                assert isinstance(value, (list, tuple)) and all(isinstance(t, str) for t in value)
                added_tokens += self.add_tokens(value)
            else:
                assert isinstance(value, str)
                added_tokens += self.add_tokens([value])
            logger.info("Assigning %s to the %s key of the tokenizer", value, key)
            setattr(self, key, value)

        return added_tokens

    def tokenize(self, text: TextInput, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.

            Args:
                text (:obj:`string`): The sequence to be encoded.
                **kwargs (:obj: `dict`): Arguments passed to the model-specific `prepare_for_tokenization` preprocessing method.
        """
        all_special_tokens = self.all_special_tokens
        text = self.prepare_for_tokenization(text, **kwargs)

        # TODO: should this be in the base class?
        def lowercase_text(t):
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            return re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), t)

        if self.init_kwargs.get("do_lower_case", False):
            text = lowercase_text(text)

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.rstrip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_added_tokens_encoder:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token) if token not in self.unique_added_tokens_encoder else [token]
                        for token in tokenized_text
                    )
                )
            )

        added_tokens = self.unique_added_tokens_encoder
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text

    def _tokenize(self, text, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        """ Converts a token string (or a sequence of tokens) in a single integer id
            (or a sequence of ids), using the vocabulary.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ):
        """
        Converts a string in a sequence of ids (integer), using the tokenizer and vocabulary. Adds the model-specific
        special tokens (such as beginning of sequence, end of sequence, sequence separator).

        If specifying ``add_special_tokens=False``, same as doing ``self.convert_tokens_to_ids(self.tokenize(text))``.

        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
            add_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length (:obj:`int`, `optional`, defaults to :obj:`None`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary.
                You can set it to the maximal input size of the model with `max_length = tokenizer.model_max_length`.
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy (:obj:`str`, `optional`, defaults to `longest_first`):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                  starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the
                model's max length. The tokenizer padding sides are handled by the class attribute `padding_side`
                which can be set to the following strings:

                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            return_tensors (:obj:`str`, `optional`, defaults to :obj:`None`):
                Can be set to 'tf' or 'pt' to return respectively TensorFlow :obj:`tf.constant`
                or PyTorch :obj:`torch.Tensor` instead of a list of python integers.
            **kwargs: passed to the `self.tokenize()` method
        """
        encoded_inputs = self.encode_plus(
            text,
            text_pair=text_pair,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            stride=stride,
            truncation_strategy=truncation_strategy,
            pad_to_max_length=pad_to_max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

        return encoded_inputs["input_ids"]

    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        is_pretokenized: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        **kwargs
    ) -> BatchEncoding:
        """
        Returns a dictionary containing the encoded sequence or sequence pair and additional information:
        the mask for sequence classification and the overflowing elements if a ``max_length`` is specified.

        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]` (the later only for not-fast tokenizers)):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
            add_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length (:obj:`int`, `optional`, defaults to :obj:`None`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
                You can set it to the maximal input size of the model with `max_length = tokenizer.model_max_length`.
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy (:obj:`str`, `optional`, defaults to `longest_first`):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                  starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the
                model's max length. The tokenizer padding sides are handled by the class attribute `padding_side`
                which can be set to the following strings:

                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Set to True to indicate the input is already tokenized
            return_tensors (:obj:`str`, `optional`, defaults to :obj:`None`):
                Can be set to 'tf' or 'pt' to return respectively TensorFlow :obj:`tf.constant`
                or PyTorch :obj:`torch.Tensor` instead of a list of python integers.
            return_token_type_ids (:obj:`bool`, `optional`, defaults to :obj:`None`):
                Whether to return token type IDs. If left to the default, will return the token type IDs according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are token type IDs? <../glossary.html#token-type-ids>`_
            return_attention_mask (:obj:`bool`, `optional`, defaults to :obj:`none`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are attention masks? <../glossary.html#attention-mask>`__
            return_overflowing_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return overflowing token information (default False).
            return_special_tokens_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return special tokens mask information (default False).
            return_offsets_mapping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return (char_start, char_end) for each token (default False).
                If using Python's tokenizer, this method will raise NotImplementedError.
                This one is only available on fast tokenizers inheriting from PreTrainedTokenizerFast.
            **kwargs: passed to the `self.tokenize()` method

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[int],
                    token_type_ids: list[int] if return_token_type_ids is True (default)
                    attention_mask: list[int] if return_attention_mask is True (default)
                    overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True``
                    and return_special_tokens_mask is True
                }

            With the fields:

            - ``input_ids``: list of token ids to be fed to a model
            - ``token_type_ids``: list of token type ids to be fed to a model
            - ``attention_mask``: list of indices specifying which tokens should be attended to by the model
            - ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
            - ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
            - ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
              tokens and 1 specifying sequence tokens.
        """

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, add_special_tokens=add_special_tokens, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # Throw an error if we can pad because there is no padding token
        if pad_to_max_length and self.pad_token_id is None:
            raise ValueError(
                "Unable to set proper padding strategy as the tokenizer does not have a padding token. "
                "In this case please set the `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via the function add_special_tokens if you want to use a padding strategy"
            )

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            add_special_tokens=add_special_tokens,
            stride=stride,
            truncation_strategy=truncation_strategy,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            prepend_batch_axis=return_tensors is not None,
        )

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        is_pretokenized: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_masks: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_masks: bool = False,
        return_offsets_mapping: bool = False,
        return_lengths: bool = False,
        **kwargs
    ) -> BatchEncoding:
        """
        Returns a dictionary containing the encoded sequence or sequence pair and additional information:
        the mask for sequence classification and the overflowing elements if a ``max_length`` is specified.

        Args:
            batch_text_or_text_pairs (:obj:`List[str]`,  :obj:`List[Tuple[str, str]]`,
                                      :obj:`List[List[str]]`,  :obj:`List[Tuple[List[str], List[str]]]`,
                                      and for not-fast tokenizers, also:
                                      :obj:`List[List[int]]`,  :obj:`List[Tuple[List[int], List[int]]]`):
                Batch of sequences or pair of sequences to be encoded.
                This can be a list of string/string-sequences/int-sequences or a list of pair of
                string/string-sequences/int-sequence (see details in encode_plus)
            add_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length (:obj:`int`, `optional`, defaults to :obj:`None`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy (:obj:`str`, `optional`, defaults to `longest_first`):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                  starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the
                model's max length. The tokenizer padding sides are handled by the class attribute `padding_side`
                which can be set to the following strings:

                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Set to True to indicate the input is already tokenized
            return_tensors (:obj:`str`, `optional`, defaults to :obj:`None`):
                Can be set to 'tf' or 'pt' to return respectively TensorFlow :obj:`tf.constant`
                or PyTorch :obj:`torch.Tensor` instead of a list of python integers.
            return_token_type_ids (:obj:`bool`, `optional`, defaults to :obj:`None`):
                Whether to return token type IDs. If left to the default, will return the token type IDs according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are token type IDs? <../glossary.html#token-type-ids>`_
            return_attention_masks (:obj:`bool`, `optional`, defaults to :obj:`none`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are attention masks? <../glossary.html#attention-mask>`__
            return_overflowing_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return overflowing token information (default False).
            return_special_tokens_masks (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return special tokens mask information (default False).
            return_offsets_mapping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return (char_start, char_end) for each token (default False).
                If using Python's tokenizer, this method will raise NotImplementedError. This one is only available on
                Rust-based tokenizers inheriting from PreTrainedTokenizerFast.
            return_lengths (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set the resulting dictionary will include the length of each encoded inputs
            **kwargs: passed to the `self.tokenize()` method

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[List[int]],
                    token_type_ids: list[List[int]] if return_token_type_ids is True (default)
                    attention_mask: list[List[int]] if return_attention_mask is True (default)
                    overflowing_tokens: list[List[int]] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: List[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[List[int]] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
                }

            With the fields:

            - ``input_ids``: list of token ids to be fed to a model
            - ``token_type_ids``: list of token type ids to be fed to a model
            - ``attention_mask``: list of indices specifying which tokens should be attended to by the model
            - ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
            - ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
            - ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
              tokens and 1 specifying sequence tokens.
        """

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, add_special_tokens=add_special_tokens, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        # Throw an error if we can pad because there is no padding token
        if pad_to_max_length and self.pad_token_id is None:
            raise ValueError(
                "Unable to set proper padding strategy as the tokenizer does not have a padding token. In this case please set the `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via the function add_special_tokens if you want to use a padding strategy"
            )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if isinstance(ids_or_pair_ids, (list, tuple)) and len(ids_or_pair_ids) == 2 and not is_pretokenized:
                ids, pair_ids = ids_or_pair_ids
            else:
                ids, pair_ids = ids_or_pair_ids, None

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        if max_length is None and pad_to_max_length:

            def total_sequence_length(input_pairs):
                first_ids, second_ids = input_pairs
                return len(first_ids) + (
                    self.num_special_tokens_to_add()
                    if second_ids is None
                    else (len(second_ids) + self.num_special_tokens_to_add(pair=True))
                )

            max_length = max([total_sequence_length(ids) for ids in input_ids])

        batch_outputs = {}
        for first_ids, second_ids in input_ids:
            # Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by
            # the model. It adds special tokens, truncates sequences if overflowing while taking into account
            # the special tokens and manages a window stride for overflowing tokens
            outputs = self.prepare_for_model(
                first_ids,
                pair_ids=second_ids,
                max_length=max_length,
                pad_to_max_length=pad_to_max_length,
                add_special_tokens=add_special_tokens,
                stride=stride,
                truncation_strategy=truncation_strategy,
                return_attention_mask=return_attention_masks,
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_masks,
                return_lengths=return_lengths,
                return_tensors=None,  # We will convert the whole batch to tensors at the end
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        if return_tensors is not None:
            convert_to_tensors(batch_outputs, return_tensors)

        return BatchEncoding(batch_outputs)

    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_lengths: bool = False,
        prepend_batch_axis: bool = False,
    ) -> BatchEncoding:
        """ Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model.
        It adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            max_length: maximum length of the returned list. Will truncate by taking into account the special tokens.
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            stride: window stride for overflowing tokens. Can be useful to remove edge effect when using sequential
                list of inputs. The overflowing token will contains a part of the previous window of tokens.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length: if set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the model's max length.
                The tokenizer padding sides are handled by the following strings:
                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            return_token_type_ids: (optional) Set to False to avoid returning token_type_ids (default: set to model specifics).
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
            return_overflowing_tokens: (optional) Set to True to return overflowing token information (default False).
            return_special_tokens_mask: (optional) Set to True to return special tokens mask information (default False).
            return_lengths (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set the resulting dictionary will include the length of each encoded inputs
            prepend_batch_axis (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set the resulting object will feature an extra dim at position 0.
                This can be seen as an unsqueezing operator.

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[int],
                    token_type_ids: list[int] if return_token_type_ids is True (default)
                    overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
                    length: int if return_lengths is True
                }

            With the fields:
                - ``input_ids``: list of token ids to be fed to a model
                - ``token_type_ids``: list of token type ids to be fed to a model

                - ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
                - ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
                - ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
                    tokens and 1 specifying sequence tokens.
                - ``length``: this is the length of ``input_ids``
        """
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Truncation: Handle max sequence length
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)
        if max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )
            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else [])

        # Build output dictionnary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        assert max_length is None or len(encoded_inputs["input_ids"]) <= max_length
        if max_length is None and len(encoded_inputs["input_ids"]) > self.model_max_length:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum sequence length "
                "for this model ({} > {}). Running this sequence through the model will result in "
                "indexing errors".format(len(ids), self.model_max_length)
            )

        # Padding
        needs_to_be_padded = pad_to_max_length and (
            max_length
            and len(encoded_inputs["input_ids"]) < max_length
            or max_length is None
            and len(encoded_inputs["input_ids"]) < self.model_max_length
            and self.model_max_length <= LARGE_INTEGER
        )

        if pad_to_max_length and max_length is None and self.model_max_length > LARGE_INTEGER:
            logger.warning(
                "Sequence can't be padded as no maximum length is specified and the model maximum length is too high."
            )

        if needs_to_be_padded:
            difference = (max_length if max_length is not None else self.model_max_length) - len(
                encoded_inputs["input_ids"]
            )
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"]) + [0] * difference
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [1] * len(encoded_inputs["input_ids"])
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [self.pad_token_id] * difference + encoded_inputs["input_ids"]
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])

        if return_lengths:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        # Prepare model inputs as tensors if asked
        if return_tensors is not None:
            convert_to_tensors(encoded_inputs, return_tensors, prepend_batch_axis)

        return BatchEncoding(encoded_inputs)

    def prepare_for_tokenization(self, text: str, **kwargs) -> str:
        """ Performs any necessary transformations before tokenization """
        return text

    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: str = "longest_first",
        stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:
        """ Truncates a sequence pair in place to the maximum length.

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            num_tokens_to_remove (:obj:`int`, `optional`, defaults to ``0``):
                number of tokens to remove using the truncation strategy
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences).
                    Overflowing tokens only contains overflow from the first sequence.
                - 'only_first': Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
        """
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if truncation_strategy == "longest_first":
            overflowing_tokens = []
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    overflowing_tokens = [ids[-1]] + overflowing_tokens
                    ids = ids[:-1]
                else:
                    pair_ids = pair_ids[:-1]
            window_len = min(len(ids), stride)
            if window_len > 0:
                overflowing_tokens = ids[-window_len:] + overflowing_tokens
        elif truncation_strategy == "only_first":
            assert len(ids) > num_tokens_to_remove
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            ids = ids[:-num_tokens_to_remove]
        elif truncation_strategy == "only_second":
            assert pair_ids is not None and len(pair_ids) > num_tokens_to_remove
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]
        elif truncation_strategy == "do_not_truncate":
            raise ValueError("Input sequence are too long for max_length. Please select a truncation strategy.")
        else:
            raise ValueError(
                "Truncation_strategy should be selected in ['longest_first', 'only_first', 'only_second', 'do_not_truncate']"
            )
        return (ids, pair_ids, overflowing_tokens)

    def create_token_type_ids_from_sequences(self, token_ids_0: List, token_ids_1: Optional[List] = None) -> List[int]:
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def build_inputs_with_special_tokens(self, token_ids_0: List, token_ids_1: Optional[List] = None) -> List:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens. This implementation does not add special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[int, List[int]]:
        """ Converts a single index or a sequence of indices (integers) in a token "
            (resp.) a sequence of tokens (str), using the vocabulary and added tokens.

            Args:
                skip_special_tokens: Don't decode special tokens (self.all_special_tokens). Default: False
        """
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """ Converts a sequence of tokens (string) in a single string.
            The most simple way to do it is ' '.join(self.convert_ids_to_tokens(token_ids))
            but we often want to remove sub-word tokenization artifacts at the same time.
        """
        return " ".join(self.convert_ids_to_tokens(tokens))

    def decode(
        self, token_ids: List[int], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True
    ) -> str:
        """
        Converts a sequence of ids (integer) in a string, using the tokenizer and vocabulary
        with options to remove special tokens and clean up tokenization spaces.
        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.

        Args:
            token_ids: list of tokenized input ids. Can be obtained using the `encode` or `encode_plus` methods.
            skip_special_tokens: if set to True, will replace special tokens.
            clean_up_tokenization_spaces: if set to True, will clean up the tokenization spaces.
        """
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separatly for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))
        text = " ".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def batch_decode(self, sequences: List[List[int]], **kwargs) -> List[str]:
        return [self.decode(seq, **kwargs) for seq in sequences]

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """ Clean up a list of simple English tokenization artifacts like spaces before punctuations and abreviated forms.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string


class PreTrainedTokenizerFast(PreTrainedTokenizer):
    """ Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Inherit from PreTrainedTokenizer.

    Handle all the shared methods for tokenization and special tokens as well as methods
    downloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary.

    This class also contain the added tokens in a unified way on top of all tokenizers so we don't
    have to handle the specific vocabulary augmentation methods of the various underlying
    dictionary structures (BPE, sentencepiece...).

    Class attributes (overridden by derived classes):

        - ``vocab_files_names``: a python ``dict`` with, as keys, the ``__init__`` keyword name of each vocabulary file
            required by the model, and as associated values, the filename for saving the associated file (string).
        - ``pretrained_vocab_files_map``: a python ``dict of dict`` the high-level keys
            being the ``__init__`` keyword name of each vocabulary file required by the model, the low-level being the
            `short-cut-names` (string) of the pretrained models with, as associated values, the `url` (string) to the
            associated pretrained vocabulary file.
        - ``max_model_input_sizes``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the pretrained
            models, and as associated values, the maximum length of the sequence inputs of this model, or None if the
            model has no maximum input size.
        - ``pretrained_init_configuration``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the
            pretrained models, and as associated values, a dictionnary of specific arguments to pass to the
            ``__init__``method of the tokenizer class for this pretrained model when loading the tokenizer with the
            ``from_pretrained()`` method.

    Args:
        - ``tokenizer`` (`BaseTokenizerFast`): A Fast tokenizer from the HuggingFace tokenizer library (in low level Rust language)
        - ``model_max_length``: (`Optional`) int: the maximum length in number of tokens for the inputs to the transformer model.
            When the tokenizer is loaded with `from_pretrained`, this will be set to the value stored for the associated
            model in ``max_model_input_sizes`` (see above). If no value is provided, will default to VERY_LARGE_INTEGER (`int(1e30)`).
            no associated max_length can be found in ``max_model_input_sizes``.
        - ``padding_side``: (`Optional`) string: the side on which the model should have padding applied.
            Should be selected between ['right', 'left']
        - ``model_input_names``: (`Optional`) List[string]: the list of the forward pass inputs accepted by the
            model ("token_type_ids", "attention_mask"...).
        - ``bos_token``: (`Optional`) string: a beginning of sentence token.
            Will be associated to ``self.bos_token`` and ``self.bos_token_id``
        - ``eos_token``: (`Optional`) string: an end of sentence token.
            Will be associated to ``self.eos_token`` and ``self.eos_token_id``
        - ``unk_token``: (`Optional`) string: an unknown token.
            Will be associated to ``self.unk_token`` and ``self.unk_token_id``
        - ``sep_token``: (`Optional`) string: a separation token (e.g. to separate context and query in an input sequence).
            Will be associated to ``self.sep_token`` and ``self.sep_token_id``
        - ``pad_token``: (`Optional`) string: a padding token.
            Will be associated to ``self.pad_token`` and ``self.pad_token_id``
        - ``cls_token``: (`Optional`) string: a classification token (e.g. to extract a summary of an input sequence
            leveraging self-attention along the full depth of the model).
            Will be associated to ``self.cls_token`` and ``self.cls_token_id``
        - ``mask_token``: (`Optional`) string: a masking token (e.g. when training a model with masked-language
            modeling). Will be associated to ``self.mask_token`` and ``self.mask_token_id``
        - ``additional_special_tokens``: (`Optional`) list: a list of additional special tokens.
            Adding all special tokens here ensure they won't be split by the tokenization process.
            Will be associated to ``self.additional_special_tokens`` and ``self.additional_special_tokens_ids``
    """

    def __init__(self, tokenizer: BaseTokenizerFast, **kwargs):
        if not isinstance(tokenizer, BaseTokenizerFast):
            raise ValueError(
                "Tokenizer should be an instance of a Tokenizer " "provided by HuggingFace tokenizers library."
            )
        self._tokenizer: BaseTokenizerFast = tokenizer

        # Initialize all the rest of the kwargs
        super().__init__(**kwargs)

    @property
    def backend_tokenizer(self) -> BaseTokenizerFast:
        return self._tokenizer

    @property
    def decoder(self) -> DecoderFast:
        return self._tokenizer._tokenizer.decoder

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def _maybe_update_backend(self, value):
        """ Update the backend fast tokenizer.
            Override method from base class SpecialTokensMixin """
        self._tokenizer.add_special_tokens(value)

    def _convert_encoding(
        self,
        encoding: EncodingFast,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
    ) -> Dict[str, Any]:
        """ Convert the encoding representation (from low-level HuggingFace tokenizer output) to a python Dict.

            Overflowing tokens are converted to additional examples (like batches) so the output values of
            the dict are lists (overflows) of lists (tokens).

            If return_tensors is not None, these lists of lists are converted to 2-D tensors
            for input_ids, token_type_ids and attention_mask.
            Output shape: (overflows, sequence length)
        """
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if return_overflowing_tokens and encoding.overflowing is not None:
            encodings = [encoding] + encoding.overflowing
        else:
            encodings = [encoding]

        encoding_dict = defaultdict(list)
        for e in encodings:
            encoding_dict["input_ids"].append(e.ids)

            if return_token_type_ids:
                encoding_dict["token_type_ids"].append(e.type_ids)
            if return_attention_mask:
                encoding_dict["attention_mask"].append(e.attention_mask)
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
            if return_offsets_mapping:
                encoding_dict["offset_mapping"].append(e.offsets)

        if return_tensors is not None:
            encoding_dict = convert_to_tensors(encoding_dict, return_tensors)

        return encoding_dict

    def _convert_token_to_id_with_added_voc(self, token: int) -> str:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        return self._tokenizer.id_to_token(int(index))

    def get_vocab(self):
        return self._tokenizer.get_vocab(True)

    def convert_tokens_to_string(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens)

    def add_tokens(self, new_tokens: List[Union[str, AddedTokenFast]]) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the
        vocabulary, they are added to it with indices starting from length of the current vocabulary.

        Args:
            new_tokens: string or list of string or AddedTokenFast. Each string is a token to add.
            Tokens are only added if they are not already in the vocabulary. AddedTokenFast wrap a string token to let you personnalize it's behavior (Whether this token should only match against single word, whether this token should strip all potential whitespaces on the left side, Whether this token should strip all potential whitespaces on the right side...).
            See details for AddedToken in HuggingFace tokenizers library.

        Returns:
            Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to increase the vocabulary of Bert model and tokenizer
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
            print('We have added', num_added_toks, 'tokens')
            model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        """
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
        return self._tokenizer.add_tokens(new_tokens)

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        # Map special tokens to class attributes (self.pad_token...)
        super().add_special_tokens(special_tokens_dict)

        # If the backend tokenizer the only specificities of special tokens are that
        #    - they will never be processed by the model, and
        #    - they will be removed while decoding.
        # But they are not mapped to special attributes in the backend so we can just
        # send a list.
        tokens = []
        for token in special_tokens_dict.values():
            if isinstance(token, list):
                tokens += token
            else:
                tokens += [token]
        num_added_tokens = self._tokenizer.add_special_tokens(tokens)

        return num_added_tokens

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return self._tokenizer.num_special_tokens_to_add(pair)

    def tokenize(
        self, text: TextInput, pair: Optional[TextInput] = None, add_special_tokens: bool = False
    ) -> List[str]:
        return self._tokenizer.encode(text, pair, add_special_tokens).tokens

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]
        ],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        is_pretokenized: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_lengths: bool = False,
        **kwargs
    ) -> BatchEncoding:

        if not isinstance(batch_text_or_text_pairs, list):
            raise ValueError(
                "batch_text_or_text_pairs has to be a list (got {})".format(type(batch_text_or_text_pairs))
            )

        # Needed if we have to return a tensor
        pad_to_max_length = pad_to_max_length or (return_tensors is not None and len(batch_text_or_text_pairs) > 1)

        # Throw an error if we can pad because there is no padding token
        if pad_to_max_length and self.pad_token_id is None:
            raise ValueError("Unable to set proper padding strategy as the tokenizer does not have a padding token")

        # Set the truncation and padding strategy and restore the initial configuration
        with truncate_and_pad(
            tokenizer=self._tokenizer,
            max_length=max_length,
            stride=stride,
            strategy=truncation_strategy,
            pad_to_max_length=pad_to_max_length,
            padding_side=self.padding_side,
            pad_token_id=self.pad_token_id if self._pad_token is not None else None,
            pad_token_type_id=self.pad_token_type_id,
            pad_token=self._pad_token,
        ):

            # Check for the pretokenized path
            if is_pretokenized:
                encodings = []

                # Iterate over each sample (we don't know yet if they are pairs or simple input
                for i, sample in enumerate(batch_text_or_text_pairs):

                    if not isinstance(sample, (list, tuple)):
                        raise TypeError(
                            "batch_encode_plus(..., is_pretokenized=True) requires batch_text_or_text_pairs "
                            "to be either List[List[str]] or List[Tuple[List[str], List[str]]] but sample at "
                            "index {} is of type {}".format(i, type(sample))
                        )

                    # Test if we have a pair of sentences by checking the depth of nesting
                    is_pair = bool(len(sample) > 0 and isinstance(sample[0], (list, tuple)))

                    # Take care of the first sequence - we multi-thread over the words
                    encodings_text = EncodingFast.merge(
                        self._tokenizer.encode_batch(sample[0] if is_pair else sample, add_special_tokens=False),
                        growing_offsets=True,
                    )

                    # Take care of the second sequence if we have a pair
                    if is_pair:
                        encodings_pair = EncodingFast.merge(
                            self._tokenizer.encode_batch([("", s) for s in sample[1]], add_special_tokens=False),
                            growing_offsets=True,
                        )
                    else:
                        encodings_pair = None

                    # Post-process - truncate/pad and add special tokens
                    encoding = self._tokenizer.post_process(encodings_text, encodings_pair, add_special_tokens)
                    encodings.append(encoding)

            # Classical path with strings input
            else:
                # Avoid thread overhead if only one example.
                if len(batch_text_or_text_pairs) == 1:
                    if isinstance(batch_text_or_text_pairs[0], (tuple, list)):
                        encodings = self._tokenizer.encode(
                            *batch_text_or_text_pairs[0], add_special_tokens=add_special_tokens
                        )
                    else:
                        encodings = self._tokenizer.encode(
                            batch_text_or_text_pairs[0], add_special_tokens=add_special_tokens
                        )
                    encodings = [encodings]
                else:
                    encodings = self._tokenizer.encode_batch(
                        batch_text_or_text_pairs, add_special_tokens=add_special_tokens
                    )

        # Convert encoding to dict
        # `Tokens` has type: List[Dict[str, List[List[int]]]] or List[Dict[str, 2D-Tensor]]
        # with nested dimensions corresponding to batch, overflows, sequence length
        tokens = [
            self._convert_encoding(
                encoding=encoding,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
            )
            for encoding in encodings
        ]

        # Sanitize the output to have dict[list] from list[dict]
        sanitized = {}
        for key in tokens[0].keys():
            # To List[List[List[int]]] of shape (batch, overflows, sequence length)
            stack = [e for item in tokens for e in item[key]]
            if return_tensors == "tf":
                stack = tf.stack(stack, axis=0)
            elif return_tensors == "pt":
                stack = torch.stack(stack, dim=0)
            # elif not return_tensors and len(stack) == 1:
            #     stack = stack[0]

            sanitized[key] = stack

        # If returning overflowing tokens, we need to return a mapping
        # from the batch idx to the original sample
        if return_overflowing_tokens:
            overflow_to_sample_mapping = flatten([[i] * len(enc["input_ids"]) for i, enc in enumerate(tokens)])
            sanitized["overflow_to_sample_mapping"] = overflow_to_sample_mapping

        return BatchEncoding(sanitized, encodings)

    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        pad_to_max_length: bool = False,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        is_pretokenized: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        **kwargs
    ) -> BatchEncoding:

        # Check for pretokenized path (ie [token1, token2, ..., tokenN] -> [id1, id2, ..., idN]
        if is_pretokenized:
            if isinstance(text, list) and len(text) > 0:

                # Encode through encode_batch with sequence of only one word which will be merged after hand
                encoding = self._tokenizer.encode_batch(text, add_special_tokens=False)
                encoding = EncodingFast.merge(encoding, growing_offsets=True)

                # Let's do the same for pairs if provided
                if isinstance(text_pair, list):
                    # We prepend empty string before each word so that encoding is aware content is a pair
                    encoding_pair = self._tokenizer.encode_batch(
                        [("", p) for p in text_pair], add_special_tokens=False
                    )
                    encoding_pair = EncodingFast.merge(encoding_pair, growing_offsets=True)
                elif text_pair is None:
                    encoding_pair = None
                else:
                    raise TypeError(
                        "encode_plus(..., is_pretokenized=True) requires text and text_pair to be List[str] "
                        "but got (text={}, text_pair={})".format(type(text), type(text_pair))
                    )

                # Post process and if asked to do so, insert special tokens where needed
                encoding = self._tokenizer.post_process(encoding, encoding_pair, add_special_tokens)

                batched_output = BatchEncoding(
                    self._convert_encoding(
                        encoding,
                        return_tensors=return_tensors,
                        return_token_type_ids=return_token_type_ids,
                        return_attention_mask=return_attention_mask,
                        return_overflowing_tokens=return_overflowing_tokens,
                        return_special_tokens_mask=return_special_tokens_mask,
                        return_offsets_mapping=return_offsets_mapping,
                    ),
                    encoding,
                )
            else:
                raise TypeError(
                    "encode_plus(..., is_pretokenized=True) requires text to be List[str] "
                    "but got (text={}, text_pair={})".format(type(text), type(text_pair))
                )
        else:
            batched_input = [(text, text_pair)] if text_pair else [text]
            batched_output = self.batch_encode_plus(
                batched_input,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                stride=stride,
                truncation_strategy=truncation_strategy,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                pad_to_max_length=pad_to_max_length,
                **kwargs,
            )

        # Return tensor is None, then we can remove the leading batch axis
        if not return_tensors:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        return batched_output

    def decode(
        self, token_ids: List[int], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True
    ) -> str:
        text = self._tokenizer.decode(token_ids, skip_special_tokens)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def save_vocabulary(self, save_directory: str) -> Tuple[str]:
        if os.path.isdir(save_directory):
            files = self._tokenizer.save(save_directory)
        else:
            folder, file = os.path.split(os.path.abspath(save_directory))
            files = self._tokenizer.save(folder, name=file)

        return tuple(files)


def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])




VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
        "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
        "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
        "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
        "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
        "bert-base-german-cased": "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt",
        "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-vocab.txt",
        "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-vocab.txt",
        "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt",
        "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt",
        "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-vocab.txt",
        "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-vocab.txt",
        "bert-base-german-dbmdz-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-vocab.txt",
        "TurkuNLP/bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/vocab.txt",
        "TurkuNLP/bert-base-finnish-uncased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/vocab.txt",
        "wietsedv/bert-base-dutch-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bert-base-uncased": 512,
    "bert-large-uncased": 512,
    "bert-base-cased": 512,
    "bert-large-cased": 512,
    "bert-base-multilingual-uncased": 512,
    "bert-base-multilingual-cased": 512,
    "bert-base-chinese": 512,
    "bert-base-german-cased": 512,
    "bert-large-uncased-whole-word-masking": 512,
    "bert-large-cased-whole-word-masking": 512,
    "bert-large-uncased-whole-word-masking-finetuned-squad": 512,
    "bert-large-cased-whole-word-masking-finetuned-squad": 512,
    "bert-base-cased-finetuned-mrpc": 512,
    "bert-base-german-dbmdz-cased": 512,
    "bert-base-german-dbmdz-uncased": 512,
    "TurkuNLP/bert-base-finnish-cased-v1": 512,
    "TurkuNLP/bert-base-finnish-uncased-v1": 512,
    "wietsedv/bert-base-dutch-cased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "bert-base-uncased": {"do_lower_case": True},
    "bert-large-uncased": {"do_lower_case": True},
    "bert-base-cased": {"do_lower_case": False},
    "bert-large-cased": {"do_lower_case": False},
    "bert-base-multilingual-uncased": {"do_lower_case": True},
    "bert-base-multilingual-cased": {"do_lower_case": False},
    "bert-base-chinese": {"do_lower_case": False},
    "bert-base-german-cased": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking-finetuned-squad": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking-finetuned-squad": {"do_lower_case": False},
    "bert-base-cased-finetuned-mrpc": {"do_lower_case": False},
    "bert-base-german-dbmdz-cased": {"do_lower_case": False},
    "bert-base-german-dbmdz-uncased": {"do_lower_case": True},
    "TurkuNLP/bert-base-finnish-cased-v1": {"do_lower_case": False},
    "TurkuNLP/bert-base-finnish-uncased-v1": {"do_lower_case": True},
    "wietsedv/bert-base-dutch-cased": {"do_lower_case": False},
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a BERT tokenizer. Based on WordPiece.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.

    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to lowercase the input when tokenizing.
        do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to do basic tokenization before WordPiece.
        never_split (:obj:`Iterable`, `optional`, defaults to :obj:`None`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        unk_token (:obj:`string`, `optional`, defaults to "[UNK]"):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`string`, `optional`, defaults to "[SEP]"):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        pad_token (:obj:`string`, `optional`, defaults to "[PAD]"):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`string`, `optional`, defaults to "[CLS]"):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.
        mask_token (:obj:`string`, `optional`, defaults to "[MASK]"):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to tokenize Chinese characters.
            This should likely be deactivated for Japanese:
            see: https://github.com/huggingface/transformers/issues/328
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case, never_split=never_split, tokenize_chinese_chars=tokenize_chinese_chars
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:

        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
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

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        if token_ids_1 is None, only returns the first portion of the mask (0's).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, vocab_path):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.

        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.

        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True):
        """ Constructs a BasicTokenizer.

        Args:
            **do_lower_case**: Whether to lower case the input.
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be deactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text, never_split=None):
        """ Basic Tokenization of a piece of text.
            Split on "white spaces" only, for sub-word tokenization, see WordPieceTokenizer.

        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class BertTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Constructs a "Fast" BERT tokenizer (backed by HuggingFace's `tokenizers` library).

    Bert tokenization is Based on WordPiece.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.

    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to lowercase the input when tokenizing.
        unk_token (:obj:`string`, `optional`, defaults to "[UNK]"):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`string`, `optional`, defaults to "[SEP]"):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        pad_token (:obj:`string`, `optional`, defaults to "[PAD]"):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`string`, `optional`, defaults to "[CLS]"):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.
        mask_token (:obj:`string`, `optional`, defaults to "[MASK]"):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to tokenize Chinese characters.
            This should likely be deactivated for Japanese:
            see: https://github.com/huggingface/transformers/issues/328
        clean_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to clean the text before tokenization by removing any control characters and
            replacing all whitespaces by the classic one.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to tokenize Chinese characters.
            This should likely be deactivated for Japanese:
            see: https://github.com/huggingface/transformers/issues/328
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        clean_text=True,
        tokenize_chinese_chars=True,
        strip_accents=True,
        wordpieces_prefix="##",
        **kwargs
    ):
        super().__init__(
            BertWordPieceTokenizer(
                vocab_file=vocab_file,
                unk_token=unk_token,
                sep_token=sep_token,
                cls_token=cls_token,
                clean_text=clean_text,
                handle_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
                lowercase=do_lower_case,
                wordpieces_prefix=wordpieces_prefix,
            ),
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        self.do_lower_case = do_lower_case

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1:
            output += token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        if token_ids_1 is None, only returns the first portion of the mask (0's).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

