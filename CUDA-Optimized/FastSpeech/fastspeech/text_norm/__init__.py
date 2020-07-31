# Copyright (c) 2017 Keith Ito

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

""" from https://github.com/keithito/tacotron """
import re
from . import cleaners
from .symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    # print(m)
    if not m:
      sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)

  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'
