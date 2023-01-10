import re
import numpy as np
from .chinese import split_text, is_chinese, chinese_text_to_symbols
from ..text_processing import TextProcessing


class MandarinTextProcessing(TextProcessing):
    def __init__(self, symbol_set, cleaner_names, p_arpabet=0.0,
                 handle_arpabet='word', handle_arpabet_ambiguous='ignore',
                 expand_currency=True):
        
        super().__init__(symbol_set, cleaner_names, p_arpabet, handle_arpabet, 
                       handle_arpabet_ambiguous, expand_currency)


    def sequence_to_text(self, sequence):
        result = ''
        
        tmp = ''
        for symbol_id in sequence:
            if symbol_id in self.id_to_symbol:
                s = self.id_to_symbol[symbol_id]
                # Enclose ARPAbet and mandarin phonemes back in curly braces:
                if len(s) > 1 and s[0] == '@':
                    s = '{%s}' % s[1:]
                    result += s
                elif len(s) > 1 and s[0] == '#' and s[1].isdigit(): # mandarin tone
                    tmp += s[1] + '} '
                    result += tmp
                    tmp = ''
                elif len(s) > 1 and s[0] == '#' and (s[1].isalpha() or s[1] == '^'): # mandarin phoneme
                    if tmp == '':
                        tmp += ' {' + s[1:] + ' '
                    else:
                        tmp += s[1:] + ' '
                elif len(s) > 1 and s[0] == '#': # chinese punctuation
                    s = s[1]
                    result += s
                else:
                    result += s
                    
        return result.replace('}{', ' ').replace('  ', ' ')

    
    def chinese_symbols_to_sequence(self, symbols):
        return self.symbols_to_sequence(['#' + s for s in symbols])


    def encode_text(self, text, return_all=False):
        # split the text into English and Chinese segments
        segments = [segment for segment in split_text(text) if segment != ""]
        
        text_encoded = []
        text_clean = ""
        text_arpabet = ""
        
        for segment in segments:
            if is_chinese(segment[0]): # process the Chinese segment
                chinese_symbols, segment_arpabet = chinese_text_to_symbols(segment)
                segment_encoded = self.chinese_symbols_to_sequence(chinese_symbols)
                segment_clean = segment
                segment_encoded = segment_encoded
            else: # process the English segment
                segment_encoded, segment_clean, segment_arpabet = \
                    super().encode_text(segment, return_all=True)
            
            text_encoded += segment_encoded
            text_clean += segment_clean
            text_arpabet += segment_arpabet

        if return_all:
            return text_encoded, text_clean, text_arpabet

        return text_encoded