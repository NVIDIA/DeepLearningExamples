import logging
from collections import defaultdict

import seq2seq.data.config as config

def default():
    return config.UNK

class Tokenizer:
    """
    Tokenizer class.
    """
    def __init__(self, vocab_fname, separator='@@'):
        """
        Constructor for the Tokenizer class.

        :param vocab_fname: path to the file with vocabulary
        :param separator: tokenization separator
        """
        self.separator = separator

        logging.info(f'Building vocabulary from {vocab_fname}')
        vocab = [config.PAD_TOKEN, config.UNK_TOKEN,
                 config.BOS_TOKEN, config.EOS_TOKEN]

        with open(vocab_fname) as vfile:
            for line in vfile:
                vocab.append(line.strip())

        logging.info(f'Size of vocabulary: {len(vocab)}')
        self.vocab_size = len(vocab)


        self.tok2idx = defaultdict(default)
        for idx, token in enumerate(vocab):
            self.tok2idx[token] = idx

        self.idx2tok = {}
        for key, value in self.tok2idx.items():
            self.idx2tok[value] = key

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
        detok = detok.replace(
            self.separator+ ' ', '').replace(self.separator, '')
        return detok
