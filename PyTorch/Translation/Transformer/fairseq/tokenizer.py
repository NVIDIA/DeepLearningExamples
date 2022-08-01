# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
#-------------------------------------------------------------------------
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import os
from collections import Counter
import re

import torch


SPACE_NORMALIZER = re.compile("\s+")

path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'prefixes/nonbreaking_prefix.en')
prefixes ={}

with open(path, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line[0] == '#':
            match = re.search(r'(.*)[\s]+(\#NUMERIC_ONLY\#)', line)
            if match:
                prefixes[match.group(1)] = 2
            else:
                prefixes[line] = 1

def get_unicode_categories():
    import sys
    from collections import defaultdict
    import unicodedata
    cats = defaultdict(list)
    for c in map(chr, range(sys.maxunicode + 1)):
        cats[unicodedata.category(c)].append(c)
    return cats

NUMERICS = ''.join(get_unicode_categories()['No'])

def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line

def tokenize_en(line):
    line = line.strip()
    line = ' ' + line + ' '
    # remove ASCII junk
    line = re.sub(r'\s+', ' ', line)
    line = re.sub(r'[\x00-\x1F]', '', line)
    #fix whitespaces
    line = re.sub('\ +', ' ', line)
    line = re.sub('^ ', '', line)
    line = re.sub(' $', '', line)
    #separate other special characters
    line = re.sub(r'([^\s\.\'\`\,\-\w]|[_'+NUMERICS+'])', r' \g<1> ', line)
    line = re.sub(r'(\w)\-(?=\w)', r'\g<1> @-@ ', line)

    #multidots stay together
    line = re.sub(r'\.([\.]+)', r' DOTMULTI\g<1>', line)
    while re.search(r'DOTMULTI\.', line):
        line = re.sub(r'DOTMULTI\.([^\.])', r'DOTDOTMULTI \g<1>', line)
        line = re.sub(r'DOTMULTI\.', r'DOTDOTMULTI', line)

    # separate out "," except if within numbers (5,300)
    line = re.sub(r'([\D])[,]', r'\g<1> , ', line)
    line = re.sub(r'[,]([\D])', r' , \g<1>', line)

    # separate "," after a number if it's the end of sentence
    line = re.sub(r'(\d)[,]$', r'\g<1> ,', line)

    # split contractions right
    line = re.sub(r'([\W\d])[\']([\W\d])', '\g<1> \' \g<2>', line)
    line = re.sub(r'(\W)[\']([\w\D])', '\g<1> \' \g<2>', line)
    line = re.sub(r'([\w\D])[\']([\W\d])', '\g<1> \' \g<2>', line)
    line = re.sub(r'([\w\D])[\']([\w\D])', '\g<1> \'\g<2>', line)
    # special case for "1990's"
    line = re.sub(r'([\W\d])[\']([s])', '\g<1> \'\g<2>', line)

    # apply nonbreaking prefixes
    words = line.split()
    line = ''
    for i in range(len(words)):
        word = words[i]
        match =  re.search(r'^(\S+)\.$', word)
        if match:
            pre = match.group(1)
            if i==len(words)-1:
                # split last words independently as they are unlikely to be non-breaking prefixes
                word = pre+' .'
            elif ((re.search(r'\.', pre) and re.search(r'[^\.\W\d]', pre))
                    or (pre in prefixes and prefixes[pre]==1)
                    or re.search(r'^[a-z]', words[i+1])
                    or (pre in prefixes and prefixes[pre]==2 and re.search(r'^[0-9]+', words[i+1]))):
                pass
            else:
                word = pre+' .'

        word +=' '
        line += word

    # clean up extraneous spaces
    line = re.sub(' +', ' ', line)
    line = re.sub('^ ', '', line)
    line = re.sub(' $', '', line)

    # .' at end of sentence is missed
    line = re.sub(r'\.\' ?$', ' . \' ', line)

    #restore multi-dots
    while re.search('DOTDOTMULTI', line):
        line = re.sub('DOTDOTMULTI', 'DOTMULTI.', line)

    line = re.sub('DOTMULTI', '.', line)

    # escape special characters
    line = re.sub(r'\&', r'&amp;', line)
    line = re.sub(r'\|', r'&#124;', line)
    line = re.sub(r'\<', r'&lt;', line)
    line = re.sub(r'\>', r'&gt;', line)
    line = re.sub(r'\'', r'&apos;', line)
    line = re.sub(r'\"', r'&quot;', line)
    line = re.sub(r'\[', r'&#91;', line)
    line = re.sub(r'\]', r'&#93;', line)

    #ensure final line breaks
    if line[-1] != '\n':
        line += '\n'

    return line

def deescape(line):
    line = re.sub(r'&#124;', r'|', line)
    line = re.sub(r'&lt;', r'<', line)
    line = re.sub(r'&gt;', r'>', line)
    line = re.sub(r'&quot;', '\"', line)
    line = re.sub(r'&apos;', '\'', line)
    line = re.sub(r'&#91;', r'[', line)
    line = re.sub(r'&#93;', r']', line)
    line = re.sub(r'&amp;', r'&', line)
    return line


class Tokenizer:

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize):
        with open(filename, 'r') as f:
            for line in f:
                for word in tokenize(line).split():
                    dict.add_symbol(word)
                dict.add_symbol(dict.eos_word)

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line,
                 append_eos=True, reverse_order=False):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, 'r') as f:
            for line in f:
                ids = Tokenizer.tokenize(
                    line=line,
                    dictionary=dict,
                    tokenize=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1

                consumer(ids)
                ntok += len(ids)
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': len(replaced)}

    @staticmethod
    def tokenize(line, dictionary, tokenize=tokenize_line, add_if_not_exist=True,
                 consumer=None, append_eos=True, reverse_order=False, bpe=None):
        line = tokenize(line)
        if bpe:
            line = bpe.process_line(line)
        words = line.split()
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dictionary.add_symbol(word)
            else:
                idx = dictionary.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = dictionary.eos_index
        return ids
    
    @staticmethod
    def detokenize(line, lang):
        #don't try to detokenize XML/HTML tag lines
        if re.search(r'^<.+>$', line) or re.search(r'^\s*$', line):
            return line

        line = line.strip()
        line = ' '+line+' '
        line = re.sub(r' @-@ ', '-', line)
        line = deescape(line)
        words = line.split()
        line = ''
        quote_count = {'\'':0, '\"':0}
        prepend_space = ' '
        for i in range(len(words)):
            #perform rught shift of currency and some punctuation
            if re.search(r'^[\u20ac\x24\(\[\{]+$', words[i]):
                line += prepend_space + words[i]
                prepend_space = ''
            elif re.search(r'^[\,\.\?\!\:\;\\\%\}\]\)]+$', words[i]):
                if lang=='fr' and re.search(r'^[\?\!\:\;\\\%]$', words[i]):
                    line += ' '
                line += words[i]
                prepend_space = ' '
            elif lang=='en' and i>0 and re.search(r'^[\'][\w\D]', words[i]) and re.search(r'\w$', words[i-1]):
                line += words[i]
                prepend_space = ' '
            elif lang=='cs' and i>1 and re.search(r'^\d+$', words[i-2]) and re.search(r'^[.,]$', words[i-1]) and re.search(r'^\w+$', words[i]):
                line += words[i]
                prepend_space = ' '
            elif (lang=='fr' or lang=='it') and i<len(words)-1 and re.search(r'[\w\D][\']$', words[i]) and re.search(r'^[\w\D]', words[i+1]):
                line += prepend_space + words[i]
                prepend_space = ''
            elif lang=='cs' and i<len(words)-3 and \
                    re.search(r'[\w\D]$', words[i]) and \
                    re.search(r'^-$', words[i+1]) and \
                    re.search(r'^li$|^mail.*', words[i+2], re.I):
                #line += ' '+words[i]+words[i+1]
                pass #TODO: skip one word
            elif re.search(r'^[\'\"\x60\u201c\u201d]+$', words[i]):
                normalized_quo = '\"' if re.search(r'^[\u201c\u201d]+$', words[i]) else words[i]
                quote_count[normalized_quo] = 0 if normalized_quo not in quote_count.keys() else quote_count[normalized_quo]
                if lang=='cs' and words[i] == '\u201c':
                    quote_count[normalized_quo] = 0
                if lang=='cs' and words[i] == '\u201d':
                    quote_count[normalized_quo] = 1
                if quote_count[normalized_quo] % 2 == 0:
                    if lang=='en' and words[i]=='\'' and i > 0 and re.search(r'[s]$', words[i-1]):
                        #single quote for posessives ending in s... "The Jones' house"
                        #left shift
                        line += words[i]
                        prepend_space = ' '
                    else:
                        #right shift
                        line += prepend_space + words[i]
                        prepend_space = ''
                        quote_count[normalized_quo] += 1
                else:
                    #left shift
                    line += words[i]
                    prepend_space = ' '
                    quote_count[normalized_quo] += 1
            elif lang=='fi' and re.search(r':$', words[i-1]) and re.search(r'^(N|n|A|a|Ä|ä|ssa|Ssa|ssä|Ssä|sta|stä|Sta|Stä|hun|Hun|hyn|Hyn|han|Han|hän|Hän|hön|Hön|un|Un|yn|Yn|an|An|än|Än|ön|Ön|seen|Seen|lla|Lla|llä|Llä|lta|Lta|ltä|Ltä|lle|Lle|ksi|Ksi|kse|Kse|tta|Tta|ine|Ine)(ni|si|mme|nne|nsa)?(ko|kö|han|hän|pa|pä|kaan|kään|kin)?$', words[i]):
                line += words[i].lower()
                prepend_space = ' '
            else:
                line += prepend_space + words[i]
                prepend_space = ' '

        #clean up spaces at head and tail of each line as well as any double-spacing
        line = re.sub(r' +', ' ', line)
        line = re.sub(r'\n ', '\n', line)
        line = re.sub(r' \n', '\n', line)
        line = re.sub(r'^ ', '', line)
        line = re.sub(r' $', '', line)

        #add trailing break
        line += '\n' if line[-1] != '\n' else ''

        return line
