# *****************************************************************************
#  Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import re

from pypinyin import lazy_pinyin, Style


valid_symbols = ['^', 'A', 'AI', 'AN', 'ANG', 'AO', 'B', 'C', 'CH', 'D', 
                 'E', 'EI', 'EN', 'ENG', 'ER', 'F', 'G', 'H', 'I', 'IE', 
                 'IN', 'ING', 'IU', 'J', 'K', 'L', 'M', 'N', 'O', 'ONG', 
                 'OU', 'P', 'Q', 'R', 'S', 'SH', 'T', 'U', 'UI', 'UN', 
                 'V', 'VE', 'VN', 'W', 'X', 'Y', 'Z', 'ZH']
tones = ['1', '2', '3', '4', '5']
chinese_punctuations = "，。？！；：、‘’“”（）【】「」《》"
valid_symbols += tones


def load_pinyin_dict(path="common/text/zh/pinyin_dict.txt"):
    with open(path) as f:
        return {l.split()[0]: l.split()[1:] for l in f}

pinyin_dict = load_pinyin_dict()


def is_chinese(text):
    return u'\u4e00' <= text[0] <= u'\u9fff' or text[0] in chinese_punctuations


def split_text(text):
    regex = r'([\u4e00-\u9fff' + chinese_punctuations + ']+)'
    return re.split(regex, text)


def chinese_text_to_symbols(text):
    symbols = []
    phonemes_and_tones = ""
    
    # convert text to mandarin pinyin sequence
    # ignore polyphonic words as it has little effect on training
    pinyin_seq = lazy_pinyin(text, style=Style.TONE3)
    
    for item in pinyin_seq:
        if item in chinese_punctuations:
            symbols += [item]
            phonemes_and_tones += ' ' + item
            continue
        if not item[-1].isdigit():
           item += '5'
        item, tone = item[:-1], item[-1]
        phonemes = pinyin_dict[item.upper()]
        symbols += phonemes
        symbols += [tone]
        
        phonemes_and_tones += '{' + ' '.join(phonemes + [tone]) + '}'
    
    return symbols, phonemes_and_tones
