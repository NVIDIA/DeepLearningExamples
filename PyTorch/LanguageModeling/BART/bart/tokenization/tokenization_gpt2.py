# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for OpenAI GPT."""


import json
import logging
import os
from functools import lru_cache

import regex as re
from tokenizers import ByteLevelBPETokenizer

from bart.tokenization.tokenization_utils import AddedToken, PreTrainedTokenizer
from bart.tokenization.tokenization_utils_base import BatchEncoding
from bart.tokenization.tokenization_utils_fast import PreTrainedTokenizerFast


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
        "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-vocab.json",
        "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-vocab.json",
        "gpt2-xl": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-vocab.json",
        "distilgpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-vocab.json",
    },
    "merges_file": {
        "gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
        "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-merges.txt",
        "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-merges.txt",
        "gpt2-xl": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-merges.txt",
        "distilgpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "gpt2": 1024,
    "gpt2-medium": 1024,
    "gpt2-large": 1024,
    "gpt2-xl": 1024,
    "distilgpt2": 1024,
}


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    We specifically avoids mapping to whitespace/control characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class GPT2Tokenizer(PreTrainedTokenizer):
    """
    GPT-2 BPE tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ::

        >>> from transformers import GPT2Tokenizer
        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> tokenizer("Hello world")['input_ids']
        [15496, 995]
        >>> tokenizer(" Hello world")['input_ids']
        [18435, 995]

    You can get around that behavior by passing ``add_prefix_space=True`` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    .. note::

        When used with ``is_pretokenized=True``, this tokenizer will add a space before each word (even the first one).

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        merges_file (:obj:`str`):
            Path to the merges file.
        errors (:obj:`str`, `optional`, defaults to "replace"):
            Paradigm to follow when decoding bytes to UTF-8. See `bytes.decode
            <https://docs.python.org/3/library/stdtypes.html#bytes.decode>`__ for more information.
        unk_token (:obj:`string`, `optional`, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to `<|endoftext|>`):
            The beginning of sequence token.
        eos_token (:obj:`string`, `optional`, defaults to `<|endoftext|>`):
            The end of sequence token.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """ Tokenize a string. """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def save_vocabulary(self, save_directory):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (:obj:`str`):
                The directory in which to save the vocabulary.

        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])
        merge_file = os.path.join(save_directory, VOCAB_FILES_NAMES["merges_file"])

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!".format(merge_file)
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    def prepare_for_tokenization(self, text, is_pretokenized=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if is_pretokenized or add_prefix_space:
            text = " " + text
        return (text, kwargs)


class GPT2TokenizerFast(PreTrainedTokenizerFast):
    """
    Constructs a "Fast" GPT-2 BPE tokenizer (backed by HuggingFace's `tokenizers` library), using byte-level
    Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ::

        >>> from transformers import GPT2TokenizerFast
        >>> tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        >>> tokenizer("Hello world")['input_ids']
        [15496, 995]
        >>> tokenizer(" Hello world")['input_ids']
        [18435, 995]

    You can get around that behavior by passing ``add_prefix_space=True`` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    .. note::

        When used with ``is_pretokenized=True``, this tokenizer needs to be instantiated with
        ``add_prefix_space=True``.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        merges_file (:obj:`str`):
            Path to the merges file.
        errors (:obj:`str`, `optional`, defaults to "replace"):
            Paradigm to follow when decoding bytes to UTF-8. See `bytes.decode
            <https://docs.python.org/3/library/stdtypes.html#bytes.decode>`__ for more information.
        unk_token (:obj:`string`, `optional`, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to `<|endoftext|>`):
            The beginning of sequence token.
        eos_token (:obj:`string`, `optional`, defaults to `<|endoftext|>`):
            The end of sequence token.
        add_prefix_space (:obj:`bool`, `optional`, defaults to `False`):
            Whether to add a leading space to the first word.
            This allows to treat the leading word just as any other word.
            (GPT2 tokenizer detect beginning of words by the preceeding space)
        trim_offsets (:obj:`bool`, `optional`, defaults to `True`):
            Whether the post processing step should trim offsets to avoid including whitespaces.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        trim_offsets=True,
        **kwargs
    ):
        super().__init__(
            ByteLevelBPETokenizer(
                vocab_file=vocab_file,
                merges_file=merges_file,
                add_prefix_space=add_prefix_space,
                trim_offsets=trim_offsets,
            ),
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs,
        )
        self.add_prefix_space = add_prefix_space

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:

        is_pretokenized = kwargs.get("is_pretokenized", False)
        assert self.add_prefix_space or not is_pretokenized, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._batch_encode_plus(*args, **kwargs)

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:

        is_pretokenized = kwargs.get("is_pretokenized", False)
        assert self.add_prefix_space or not is_pretokenized, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._encode_plus(*args, **kwargs)