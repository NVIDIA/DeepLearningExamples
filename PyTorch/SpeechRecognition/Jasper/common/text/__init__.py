# Copyright (c) 2017 Keith Ito
""" from https://github.com/keithito/tacotron """
import re
import string
from . import cleaners

def _clean_text(text, cleaner_names, *args):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text, *args)
    return text


def punctuation_map(labels):
    # Punctuation to remove
    punctuation = string.punctuation
    punctuation = punctuation.replace("+", "")
    punctuation = punctuation.replace("&", "")
    # TODO We might also want to consider:
    # @ -> at
    # # -> number, pound, hashtag
    # ~ -> tilde
    # _ -> underscore
    # % -> percent
    # If a punctuation symbol is inside our vocab, we do not remove from text
    for l in labels:
        punctuation = punctuation.replace(l, "")
    # Turn all punctuation to whitespace
    table = str.maketrans(punctuation, " " * len(punctuation))
    return table
