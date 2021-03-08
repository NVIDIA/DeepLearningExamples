# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import print_function

import argparse
import codecs
from onmt.translate import GNMTGlobalScorer
from utils.translation_model import load_test_model
from utils.translator import Translator


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--beam_size", type=int, default=4, help="beam size")
parser.add_argument("--model_type", type=str, help="ori, decoder_ext, decoding_ext, torch_decoding, torch_decoding_with_decoder_ext")
parser.add_argument("--data_type", type=str, help="fp32, fp16")
parser.add_argument('--model_path', type=str, default='./pytorch/translation/models/averaged-10-epoch.pt',
                    help='path for model checkpoint')
parser.add_argument('--module_path', type=str, default='./',
                    help='path containing the th_fastertransformer dynamic lib')
parser.add_argument('--ths', action='store_true', help='use custom TorchScript class (only for extensions)')
parser.add_argument('--ths_path', type=str, default='./lib/libths_fastertransformer.so',
                    help='path of the ths_fastertransformer dynamic lib file')
parser.add_argument('--input_file', type=str, default='./pytorch/translation/data/test.en',
                    help='input file path')
parser.add_argument('--output_file', type=str, default='',
                    help='output file path')                    
args = parser.parse_args()

opt = argparse.Namespace(models=[args.model_path],
                         fp32=False, data_type='text', output='/dev/null', report_align=False, report_time=True,
                         random_sampling_topk=1, random_sampling_temp=1.0, seed=829,
                         beam_size=args.beam_size, min_length=0, max_length=100,
                         stepwise_penalty=False, length_penalty='none', ratio=-0.0, coverage_penalty='none', alpha=0.0, beta=-0.0,
                         block_ngram_repeat=0, ignore_when_blocking=[], replace_unk=False, phrase_table='',
                         verbose=True, dump_beam='', n_best=1, batch_type='sents', gpu=0)


fields, model, model_opt = load_test_model(opt, args)
scorer = GNMTGlobalScorer.from_opt(opt)
out_file = codecs.open(opt.output, 'w+', 'utf-8')
translator = Translator.from_opt(
    model,
    fields,
    opt,
    model_opt,
    args,
    global_scorer=scorer,
    out_file=out_file,
    report_align=opt.report_align,
    report_score=False,
    logger=None
)


res = []
n = 1
with open(args.input_file, 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    translated = translator.translate(lines, batch_size=args.batch_size)
    for i in range(len(translated[1])):
        res.append(translated[1][i][0])

if args.output_file:
    with open(args.output_file, 'w') as f:
        for line in res:
            f.write(line + '\n')
