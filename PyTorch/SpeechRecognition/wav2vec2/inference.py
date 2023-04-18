# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import io
import math
import os
import random
import time
import warnings
from argparse import ArgumentParser
from heapq import nlargest
from itertools import chain, repeat
from pathlib import Path
from tqdm import tqdm

import dllogger
import numpy as np
import torch
import torch.distributed as distrib
from dllogger import JSONStreamBackend, StdOutBackend, Verbosity

import wav2vec2.arg_parser
import wav2vec2.utils
import common.fairseq.utils as utils
from common.fairseq.data import Dictionary
from common.helpers import (gather_predictions, gather_transcripts,
                            load_wrapped_state, process_evaluation_epoch)
from common.tb_dllogger import stdout_metric_format, unique_log_fpath
from common.utils import print_once
from torch.utils.data import DataLoader, DistributedSampler
from wav2vec2.logging import init_infer_metadata


def durs_to_percentiles(durations, ratios):
    durations = np.asarray(durations) * 1000  # in ms
    latency = durations

    latency = latency[5:]
    mean_latency = np.mean(latency)

    latency_worst = nlargest(math.ceil((1 - min(ratios)) * len(latency)),
                             latency)
    latency_ranges = get_percentile(ratios, latency_worst, len(latency))
    latency_ranges[0.5] = mean_latency
    return latency_ranges


def get_percentile(ratios, arr, nsamples):
    res = {}
    for a in ratios:
        idx = max(int(nsamples * (1 - a)), 0)
        res[a] = arr[idx]
    return res


def fp_convert_batch(batch, precision):

    dt = {'fp32': torch.float32, 'fp16': torch.half,
          'bf16': torch.bfloat16}[precision]

    def maybe_cast(t):
        if t.dtype is torch.float32:
            return t.to(dtype=dt)
        return t

    return utils.apply_to_sample(maybe_cast, batch)


def main():
    parser = ArgumentParser(description='wav2vec2.0 inference')
    wav2vec2.arg_parser.populate_infer(parser)
    args = parser.parse_args()

    ckpt = torch.load(args.w2v_path, map_location=torch.device("cpu"))
    train_args = wav2vec2.utils.get_ckpt_args(ckpt)
    is_nv_ckpt = "mode" in train_args

    if is_nv_ckpt:
        print("Loaded a model trained with NVIDIA DLE")
        args.fp32_pos_conv = train_args.get("fp32_pos_conv",
                                            args.fp16 or args.bf16)
        args.fp32_conv_norms = train_args.get("fp32_conv_norms", args.fp16)
    else:
        args.fp32_pos_conv = args.fp16
        args.fp32_conv_norms = args.fp16

    args.fp32_pos_conv = True
    args.fp32_conv_norms = True

    log_fpath = args.log_file or str(Path(args.output_dir, 'nvlog_infer.json'))
    dllogger.init(backends=[
        JSONStreamBackend(Verbosity.DEFAULT, log_fpath, append=True),
        JSONStreamBackend(Verbosity.DEFAULT, unique_log_fpath(log_fpath)),
        StdOutBackend(Verbosity.VERBOSE, metric_format=stdout_metric_format)
    ])
    [dllogger.log("PARAMETER", {k: v}) for k, v in vars(args).items()]
    init_infer_metadata()

    if ((train_args.get("fp16", False) or train_args.get("amp", False))
            and args.bf16):
        warnings.warn('Using FP16 ckpts in BF16 precision.')
    if train_args.get("bf16", False) and args.fp16:
        warnings.warn('Using BF16 ckpts in FP16 precision.')

    # load output labels - either from a file, or stored inside an nv ckpt
    assert args.labels_path is not None or is_nv_ckpt
    if args.labels_path is None:
        f = io.StringIO(ckpt["output_labels"])
    else:
        f = open(args.labels_path)
    target_dictionary = Dictionary.load(f)
    f.close()

    w2v_path_for_args = args.w2v_path_for_args or args.w2v_path
    wav2vec2.utils.update_args_for_finetuning(args, w2v_path_for_args)

    # "default" GroupNorm might leak padding
    args.masked_feature_extractor = True

    if args.torchscript:
        from common.fairseq.modules import layer_norm
        layer_norm.TORCHSCRIPT = True

    model, *_ = wav2vec2.utils.build_model(args, "infer", target_dictionary)

    load_wrapped_state(model, ckpt["model"])

    model.w2v_encoder.w2v_model.remove_conv_wn()
    model.w2v_encoder.w2v_model.feature_extractor.forward = \
        model.w2v_encoder.w2v_model.feature_extractor.masked_forward
    model.w2v_encoder.forward = model.w2v_encoder.infer
    model.w2v_encoder.w2v_model.forward = model.w2v_encoder.w2v_model.infer

    if args.cpu:
        device = torch.device('cpu')
    else:
        assert torch.cuda.is_available()
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.seed is not None:
        torch.manual_seed(args.seed + args.local_rank)
        np.random.seed(args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

    # set up distributed training
    multi_gpu = not args.cpu and int(os.environ.get('WORLD_SIZE', 1)) > 1
    if multi_gpu:
        torch.cuda.set_device(args.local_rank)
        distrib.init_process_group(backend='nccl', init_method='env://')
        print_once(f'Inference with {distrib.get_world_size()} GPUs')

    measure_perf = args.steps > 0

    # Compliance with fairseq dataloader
    assert args.batch_size is not None
    args.min_sample_size = None
    args.max_sample_size = None

    if args.transcribe_wav or args.transcribe_filelist:
        assert args.max_duration is None and not measure_perf
        assert not (args.transcribe_wav and args.transcribe_filelist)
        assert args.labels is None, "Labels won't be used during trainscribing"
        assert not multi_gpu, (
            "multigpu is currently supported only for WER/perf measurements")

        if args.transcribe_wav:
            dataset = wav2vec2.utils.single_audio_dataset(args.transcribe_wav,
                                                          args)
        else:
            dataset = wav2vec2.utils.load_dataset(args.transcribe_filelist,
                                                  args, target_dictionary)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=dataset.collater,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            drop_last=False,
        )

    else:  # compute WER or measure perf
        assert args.labels is not None or measure_perf

        dataset = wav2vec2.utils.load_dataset(args.valid_subset, args,
                                              target_dictionary,
                                              with_labels=True)
        sampler = DistributedSampler(
            dataset,
            shuffle=False,
            drop_last=False
        ) if multi_gpu else None

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=dataset.collater,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            drop_last=(True if measure_perf else False),
        )

    model.to(device)
    model.eval()

    assert args.amp == args.fp16, 'During inference these are equivalent'
    if args.fp16:
        model = model.half()
    if args.bf16:
        model = model.to(dtype=torch.bfloat16)

    if (args.fp16 or args.bf16) and args.fp32_pos_conv:
        model.w2v_encoder.w2v_model.encoder.pos_conv.to(dtype=torch.float32)

    if args.torchscript:
        print("Attempting TorchScript export...")
        model = torch.jit.script(model)

    agg = {'txts': [], 'preds': [], 'logits': [], 'ids': []}
    dur = {'data': [], 'dnn': [], 'data+dnn': []}

    looped_loader = chain.from_iterable(repeat(data_loader))

    sync = lambda: torch.cuda.synchronize() if device.type == 'cuda' else None
    steps = args.steps + args.warmup_steps or len(data_loader)

    desc = 'warmup' if args.warmup_steps > 0 else 'inference'
    pbar = tqdm(looped_loader, initial=1, total=steps, desc=desc)
    for it, batch in enumerate(pbar):
        if it == args.warmup_steps:
            pbar.set_description('inference')

        batch = utils.move_to_cuda(batch)

        sync()
        t1 = time.time()

        if args.fp16:
            batch = fp_convert_batch(batch, 'fp16')
        if args.bf16:
            batch = fp_convert_batch(batch, 'bf16')

        with torch.no_grad():
            enc_out, padding_mask = model(batch["net_input"]["source"],
                                          batch["net_input"]["padding_mask"])
            logp = model.get_normalized_probs(enc_out,
                                              padding_mask,
                                              log_probs=True).contiguous()
            # greedy decoding
            preds = logp.argmax(dim=-1, keepdim=False).int()

        sync()
        t2 = time.time()

        # burn-in period; wait for a new loader due to num_workers
        if it >= 1 and (args.steps == 0 or it >= args.warmup_steps):
            dur['data'].append(t1 - t0)
            dur['dnn'].append(t2 - t1)
            dur['data+dnn'].append(t2 - t0)

        preds = preds.transpose(0, 1)
        agg['preds'] += gather_predictions([preds],
                                           target_dictionary,
                                           blank_id=0)
        agg['logits'].append(logp)

        if 'target' in batch:
            agg['txts'] += gather_transcripts([batch['target']],
                                              [batch['target_lengths']],
                                              target_dictionary)
        if multi_gpu:
            # ids are needed to remove duplicates in multi_gpu inference
            agg['ids'] += batch['id'].tolist()

        if it + 1 == steps:
            break

        sync()
        t0 = time.time()

    tdict = target_dictionary
    agg['preds'] = [pred.replace(tdict[tdict.nspecial], ' ')
                    for pred in agg['preds']]
    agg['txts'] = [txt.replace(tdict[tdict.nspecial], ' ')
                   for txt in agg['txts']]

    # communicate the results
    if args.transcribe_wav or args.transcribe_filelist:
        for idx, p in enumerate(agg['preds']):
            print_once(f'Prediction {idx + 1: >3}: {p}')

    elif args.valid_subset and not measure_perf:
        wer, _ = process_evaluation_epoch(agg)
        if not multi_gpu or distrib.get_rank() == 0:
            dllogger.log(step=(), data={'eval_wer': 100 * wer})

    if args.save_predictions and (not multi_gpu or distrib.get_rank() == 0):
        with open(args.save_predictions, 'w') as f:
            f.write('\n'.join(agg['preds']))

    if args.save_logits and (not multi_gpu or distrib.get_rank() == 0):
        logits = torch.cat(agg['logits'], dim=0).cpu()
        torch.save(logits, args.save_logits)

    # report timings
    if len(dur['data']) >= 20 and (not multi_gpu or distrib.get_rank() == 0):
        ratios = [0.9, 0.95, 0.99]
        for stage in dur:
            lat = durs_to_percentiles(dur[stage], ratios)
            for k in [0.99, 0.95, 0.9, 0.5]:
                k_ = str(k).replace('.', '_')
                dllogger.log(step=(), data={f'{stage}_latency_{k_}': lat[k]})
    else:
        print_once('Not enough samples to measure latencies.')


if __name__ == "__main__":
    main()
