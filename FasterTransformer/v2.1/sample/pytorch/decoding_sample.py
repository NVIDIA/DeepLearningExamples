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

import os
import argparse
import timeit
import torch
# import torch.cuda.nvtx as nvtx

from onmt.utils.misc import sequence_mask
from utils.decoding import DecodingWeights, CustomDecoding, TorchDecoding, ArgHelper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int,
                        help='batch size')
    parser.add_argument('layer_num', type=int,
                        help='number of layers')
    parser.add_argument('seq_len', type=int,
                        help='sequence length')
    parser.add_argument('head_num', type=int,
                        help='head number')
    parser.add_argument('head_size', type=int,
                        help='size per head')
    parser.add_argument('beam_size', type=int,
                        help='beam size')
    parser.add_argument('vocab_size', type=int,
                        help='vocab size')
    parser.add_argument('--fp16', action='store_true',
                        help='is fp16')
    parser.add_argument('--time', action='store_true',
                        help='test the time or not.')
    parser.add_argument('--use_pretrained', action='store_true',
                        help='use pretrained weights or not.')
    parser.add_argument('--module_path', type=str, default='./',
                        help='directory containing the th_fastertransformer dynamic lib')
    parser.add_argument('--ths', action='store_true',
                        help='use TorchScript mode')
    parser.add_argument('--ths_path', type=str, default='./lib/libths_fastertransformer.so',
                        help='path of the ths_fastertransformer dynamic lib file')

    args = parser.parse_args()

    if args.use_pretrained:
        layer_num = 6
        head_num = 8
        head_size = 64
        vocab_size = 31538
    else:
        layer_num = args.layer_num
        head_num = args.head_num
        head_size = args.head_size
        vocab_size = args.vocab_size
    hidden_dim = head_num * head_size

    print("\n=============== Argument ===============")
    print('batch_size: ' + str(args.batch_size))
    print('layer_num: ' + str(layer_num))
    print('seq_len: ' + str(args.seq_len))
    print('head_num: ' + str(head_num))
    print('head_size: ' + str(head_size))
    print('hidden_dim: ' + str(hidden_dim))
    print('beam_size: ' + str(args.beam_size))
    print('vocab_size: ' + str(vocab_size))
    print('use_pretrained: ' + str(args.use_pretrained))
    print('use_fp16: ' + str(args.fp16))
    print('TorchScript mode: ' + str(args.ths))
    print('test_time: ' + str(args.time))
    print("========================================\n")

    decodingargs1 = ArgHelper('torch_decoding', 'fp16' if args.fp16 else 'fp32',
                              os.path.abspath(args.module_path), args.ths, os.path.abspath(args.ths_path))
    decodingargs2 = ArgHelper('torch_decoding_with_decoder_ext', 'fp16' if args.fp16 else 'fp32',
                              os.path.abspath(args.module_path), args.ths, os.path.abspath(args.ths_path))

    mem = torch.empty(args.batch_size, args.seq_len, hidden_dim).cuda()
    torch.nn.init.uniform_(mem, -1, 1)
    if args.fp16:
        mem = mem.half()
    mem_seq_lens = torch.randint(1, args.seq_len+1, (args.batch_size,), dtype=torch.int32).cuda()

    if args.use_pretrained:
        ckpt = torch.load('./pytorch/translation/models/averaged-10-epoch.pt')
        import re
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                        r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                        r'\1.layer_norm\2.weight', s)
            return s
        ckpt['model'] = {fix_key(k): v for k, v in ckpt['model'].items()}
        weights = DecodingWeights(layer_num, hidden_dim, vocab_size, ckpt)
    else:
        weights = DecodingWeights(layer_num, hidden_dim, vocab_size)
    torch_decoding = TorchDecoding(layer_num, head_num, head_size, vocab_size, 2, 3, weights, args=decodingargs1)
    torch_decoding_with_decoder_ext = TorchDecoding(args.layer_num, head_num, head_size, args.vocab_size, 2, 3, weights, args=decodingargs2)
    torch_decoding.cuda()
    torch_decoding_with_decoder_ext.cuda()
    if args.fp16:
        torch_decoding.half()
        torch_decoding_with_decoder_ext.half()
    torch_decoding.eval()
    torch_decoding_with_decoder_ext.eval()
    weights.to_cuda()
    if args.fp16:
        weights.to_half()
    custom_decoding = CustomDecoding(layer_num, head_num, head_size, vocab_size, 2, 3, weights,
                                     args=decodingargs1)

    with torch.no_grad():
        output0, lens0 = torch_decoding(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
        print(output0)
        print(lens0)
        output1, lens1 = torch_decoding_with_decoder_ext(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
        print(output1)
        print(lens1)
        output2, lens2 = custom_decoding(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
        print(output2)
        print(lens2)
        # diff = torch.abs((output1 - output2) / output1)
        # print('step: {}     Mean relative diff: {}     Max relative diff: {}     Min relative diff: {}'.format(
        #     i, torch.mean(diff), torch.max(diff), torch.min(diff)))

        if args.time:
            iterations = 10

            for i in range(iterations):
                output, lens = torch_decoding(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
            t00 = timeit.default_timer()
            for i in range(iterations):
                output, lens = torch_decoding(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
            t0 = timeit.default_timer() - t00

            for i in range(iterations):
                output, lens = torch_decoding_with_decoder_ext(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
            t10 = timeit.default_timer()
            for i in range(iterations):
                output, lens = torch_decoding_with_decoder_ext(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
            t1 = timeit.default_timer() - t10

            for i in range(iterations):
                output, lens = custom_decoding(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
            t20 = timeit.default_timer()
            for i in range(iterations):
                output, lens = custom_decoding(args.batch_size, args.beam_size, args.seq_len, mem, mem_seq_lens)
            t2 = timeit.default_timer() - t20
            print("[INFO] TorchDecoding time costs: {:.2f} ms".format(t0*1000/iterations))
            print("[INFO] TorchDecoding (with FTDecoder) time costs: {:.2f} ms".format(t1*1000/iterations))
            print("[INFO] FTDecoding time costs: {:.2f} ms".format(t2*1000/iterations))


if __name__ == '__main__':
    main()
