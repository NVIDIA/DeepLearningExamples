import logging
from collections import defaultdict
from functools import partial

import seq2seq.data.config as config


class Tokenizer:
    """
    Tokenizer class.
    """
    def __init__(self, vocab_fname=None, pad=1, separator='@@'):
        """
        Constructor for the Tokenizer class.

        :param vocab_fname: path to the file with vocabulary
        :param pad: pads vocabulary to a multiple of 'pad' tokens
        :param separator: tokenization separator
        """
        if vocab_fname:
            self.separator = separator

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
            'separator': self.separator,
            'vocab_size': self.vocab_size,
            'tok2idx': self.tok2idx,
            'idx2tok': self.idx2tok,
        }
        return state

    def set_state(self, state):
        logging.info(f'Restoring state of the tokenizer')
        self.separator = state['separator']
        self.vocab_size = state['vocab_size']
        self.tok2idx = state['tok2idx']
        self.idx2tok = state['idx2tok']

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

    def detokenize(self, inputs, delim=' '):
        """
        Detokenizes single sentence and removes token separator characters.

        :param inputs: sequence of tokens
        :param delim: tokenization delimiter

        returns: string representing detokenized sentence
        """
        detok = delim.join([self.idx2tok[idx] for idx in inputs])
        detok = detok.replace(self.separator + ' ', '')
        detok = detok.replace(self.separator, '')

        detok = detok.replace(config.BOS_TOKEN, '')
        detok = detok.replace(config.EOS_TOKEN, '')
        detok = detok.replace(config.PAD_TOKEN, '')
        detok = detok.strip()
        return detok
