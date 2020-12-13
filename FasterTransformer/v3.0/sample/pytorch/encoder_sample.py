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
import torch.cuda.nvtx as nvtx

from utils.encoder import EncoderWeights, CustomEncoder, CustomEncoder2, HuggingFaceEncoder


def sequence_mask(lengths, max_len=None, is_2d=True):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    mask = (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    if is_2d:
        return mask
    else:
        mask = mask.view(-1, 1, 1, max_len)
        m2 = mask.transpose(2, 3)
        return mask * m2


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
    parser.add_argument('--fp16', action='store_true',
                        help='is fp16')
    parser.add_argument('--time', action='store_true',
                        help='test the time or not.')
    parser.add_argument('--use_pretrained', action='store_true',
                        help='use pretrained weights or not.')
    parser.add_argument('--remove_padding', action='store_true',
                        help='Remove the padding of sentences of encoder.')
    parser.add_argument('--weight_path', type=str,
                        default='./pytorch/bert_squad/models/bert-large-uncased-whole-word-masking-finetuned-squad/pytorch_model.bin',
                        help='path containing the pretrained weights')
    parser.add_argument('--module_path', type=str, default='./',
                        help='directory containing the th_fastertransformer dynamic lib')
    parser.add_argument('--ths', action='store_true',
                        help='use TorchScript mode')
    parser.add_argument('--ths_type', type=int, default=0,
                        help='custom TorchScript type')
    parser.add_argument('--ths_path', type=str, default='./lib/libths_fastertransformer.so',
                        help='path of the ths_fastertransformer dynamic lib file')
    parser.add_argument('--ths_path_2', type=str, default='./lib/libths_fastertransformer_op.so',
                        help='path of the ths_fastertransformer op dynamic lib file')

    args = parser.parse_args()

    batch_size = args.batch_size
    seq_len = args.seq_len
    if args.use_pretrained:
        if 'large' in args.weight_path:
            layer_num = 24
            head_num = 16
            head_size = 64
        elif 'base' in args.weight_path:
            layer_num = 12
            head_num = 12
            head_size = 64
        else:
            layer_num = args.layer_num
            head_num = args.head_num
            head_size = args.head_size
    else:
        layer_num = args.layer_num
        head_num = args.head_num
        head_size = args.head_size
    hidden_dim = head_num * head_size

    print("\n=============== Argument ===============")
    print('batch_size: ' + str(batch_size))
    print('layer_num: ' + str(layer_num))
    print('seq_len: ' + str(seq_len))
    print('head_num: ' + str(head_num))
    print('head_size: ' + str(head_size))
    print('hidden_dim: ' + str(hidden_dim))
    print('use_pretrained: ' + str(args.use_pretrained))
    print('use_fp16: ' + str(args.fp16))
    print('TorchScript mode: ' + str(args.ths))
    print('TorchScript type: ' + str(args.ths_type))
    print('test_time: ' + str(args.time))
    print('remove_padding: ' + str(args.remove_padding))
    print("========================================\n")

    inp = torch.empty(batch_size, seq_len, hidden_dim).cuda()
    torch.nn.init.uniform_(inp, -1, 1)
    mem_seq_lens = torch.randint(1, seq_len+1, (batch_size,), dtype=torch.int32).cuda()
    mask = sequence_mask(mem_seq_lens, args.seq_len, False).to(torch.float)
    # mask = torch.randint(0, 2, (batch_size, seq_len, seq_len), dtype=torch.float32).cuda()
    if args.fp16:
        inp = inp.half()
        mask = mask.half()

    if args.use_pretrained:
        pretrained_weights = torch.load(args.weight_path)
        weights = EncoderWeights(layer_num, hidden_dim, pretrained_weights)
    else:
        weights = EncoderWeights(layer_num, hidden_dim)

    if args.use_pretrained:
        hf_encoder = HuggingFaceEncoder(layer_num, head_num, head_size, pretrained_weights)
    else:
        hf_encoder = HuggingFaceEncoder(layer_num, head_num, head_size, weights)
    hf_encoder.cuda()
    if args.fp16:
        hf_encoder.half()
    hf_encoder.eval()
    if args.ths:
        hf_encoder = torch.jit.trace(hf_encoder, (inp, mask))

    weights.to_cuda()
    if args.fp16:
        weights.to_half()
    if args.ths:
        if args.ths_type == 0:
            custom_encoder = CustomEncoder(layer_num, head_num, head_size, weights,
                                           os.path.abspath(args.ths_path), args.ths, remove_padding=args.remove_padding)
        else:
            custom_encoder = CustomEncoder2(layer_num, head_num, head_size, weights,
                                            os.path.abspath(args.ths_path_2), remove_padding=args.remove_padding)
    else:
        custom_encoder = CustomEncoder(layer_num, head_num, head_size, weights,
                                       os.path.abspath(args.module_path), remove_padding=args.remove_padding)
    if args.ths:
        if args.ths_type == 0:
            custom_encoder = torch.jit.script(custom_encoder)
        else:
            custom_encoder = torch.jit.trace(custom_encoder, (inp, mask, mem_seq_lens))

    with torch.no_grad():
        output_mask = sequence_mask(mem_seq_lens, args.seq_len).to(mask.dtype).unsqueeze(-1)
        output1 = hf_encoder(inp, mask)[0] * output_mask
        print(output1)
        print(output1.size())

        output2 = custom_encoder(inp, mask, mem_seq_lens)[0] * output_mask
        print(output2)
        print(output2.size())

        diff = torch.abs(output1 - output2)
        print('Mean diff: {}'.format(torch.mean(diff)))
        print('Max diff: {}'.format(torch.max(diff)))
        print('Min diff: {}'.format(torch.min(diff)))

        if args.time:
            iterations = 100

            for i in range(iterations):
                output = hf_encoder(inp, mask)
            t10 = timeit.default_timer()
            # nvtx.range_push("hf")
            for i in range(iterations):
                # nvtx.range_push("hf"+str(i))
                output = hf_encoder(inp, mask)
                # nvtx.range_pop()
            # nvtx.range_pop()
            t1 = timeit.default_timer() - t10

            for i in range(iterations):
                output = custom_encoder(inp, mask, mem_seq_lens)
            t20 = timeit.default_timer()
            # nvtx.range_push("ext")
            for i in range(iterations):
                # nvtx.range_push("ext"+str(i))
                output = custom_encoder(inp, mask, mem_seq_lens)
                # nvtx.range_pop()
            # nvtx.range_pop()
            t2 = timeit.default_timer() - t20
            print("[INFO] HuggingFaceEnocder time costs: {:.2f} ms".format(t1*1000/iterations))
            print("[INFO] FasterTransformer time costs: {:.2f} ms".format(t2*1000/iterations))


if __name__ == '__main__':
    main()
