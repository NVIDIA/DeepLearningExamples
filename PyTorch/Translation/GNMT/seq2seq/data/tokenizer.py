# Copyright (c) 2017 Elad Hoffer
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
from collections import defaultdict
from functools import partial

import torch
import subword_nmt.apply_bpe
import sacremoses
import seq2seq.data.config as config


class Tokenizer:
    """
    Tokenizer class.
    """
    def __init__(self, vocab_fname=None, bpe_fname=None, lang=None, pad=1,
                 separator='@@'):
        """
        Constructor for the Tokenizer class.

        :param vocab_fname: path to the file with vocabulary
        :param bpe_fname: path to the file with bpe codes
        :param pad: pads vocabulary to a multiple of 'pad' tokens
        :param separator: tokenization separator
        """
        self.separator = separator
        self.lang = lang

        if bpe_fname:
            with open(bpe_fname, 'r') as bpe_codes:
                self.bpe = subword_nmt.apply_bpe.BPE(bpe_codes)

        if vocab_fname:
            self.build_vocabulary(vocab_fname, pad)

        if lang:
            self.init_moses(lang)

    def init_moses(self, lang):
        self.moses_tokenizer = sacremoses.MosesTokenizer(lang['src'])
        self.moses_detokenizer = sacremoses.MosesDetokenizer(lang['tgt'])

    def build_vocabulary(self, vocab_fname, pad):
        logging.info(f'Building vocabulary from {vocab_fname}')
        vocab = [config.PAD_TOKEN, config.UNK_TOKEN,
                 config.BOS_TOKEN, config.EOS_TOKEN]
        with open(vocab_fname) as vfile:
            for line in vfile:
                vocab.append(line.strip())

        self.pad_vocabulary(vocab, pad)

        self.vocab_size = len(vocab)
        logging.info(f'Size of vocabulary: {self.vocab_size}')

        self.tok2idx = defaultdict(partial(int, config.UNK))
        for idx, token in enumerate(vocab):
            self.tok2idx[token] = idx

        self.idx2tok = {}
        for key, value in self.tok2idx.items():
            self.idx2tok[value] = key

    def pad_vocabulary(self, vocab, pad):
        """
        Pads vocabulary to a multiple of 'pad' tokens.

        :param vocab: list with vocabulary
        :param pad: integer
        """
        vocab_size = len(vocab)
        padded_vocab_size = (vocab_size + pad - 1) // pad * pad
        for i in range(0, padded_vocab_size - vocab_size):
            token = f'madeupword{i:04d}'
            vocab.append(token)
        assert len(vocab) % pad == 0

    def get_state(self):
        logging.info(f'Saving state of the tokenizer')
        state = {
            'lang': self.lang,
            'separator': self.separator,
            'vocab_size': self.vocab_size,
            'bpe': self.bpe,
            'tok2idx': self.tok2idx,
            'idx2tok': self.idx2tok,
        }
        return state

    def set_state(self, state):
        logging.info(f'Restoring state of the tokenizer')
        self.lang = state['lang']
        self.separator = state['separator']
        self.vocab_size = state['vocab_size']
        self.bpe = state['bpe']
        self.tok2idx = state['tok2idx']
        self.idx2tok = state['idx2tok']

        self.init_moses(self.lang)

    def segment(self, line):
        """
        Tokenizes single sentence and adds special BOS and EOS tokens.

        :param line: sentence

        returns: list representing tokenized sentence
        """
        line = line.strip().split()
        entry = [self.tok2idx[i] for i in line]
        entry = [config.BOS] + entry + [config.EOS]
        return entry

    def tokenize(self, line):
        tokenized = self.moses_tokenizer.tokenize(line, return_str=True)
        bpe = self.bpe.process_line(tokenized)
        segmented = self.segment(bpe)
        tensor = torch.tensor(segmented)
        return tensor

    def detokenize_bpe(self, inp, delim=' '):
        """
        Detokenizes single sentence and removes token separator characters.

        :param inputs: sequence of tokens
        :param delim: tokenization delimiter

        returns: string representing detokenized sentence
        """
        detok = delim.join([self.idx2tok[idx] for idx in inp])
        detok = detok.replace(self.separator + ' ', '')
        detok = detok.replace(self.separator, '')

        detok = detok.replace(config.BOS_TOKEN, '')
        detok = detok.replace(config.EOS_TOKEN, '')
        detok = detok.replace(config.PAD_TOKEN, '')
        detok = detok.strip()
        return detok

    def detokenize_moses(self, inp):
        output = self.moses_detokenizer.detokenize(inp.split())
        return output

    def detokenize(self, inp):
        detok_bpe = self.detokenize_bpe(inp)
        output = self.detokenize_moses(detok_bpe)
        return output
