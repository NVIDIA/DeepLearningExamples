# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import os
import random
import time

import torch
import numpy as np
import torch.distributed as dist
from contextlib import suppress as empty_context

from common import helpers
from common.dali.data_loader import DaliDataLoader
from common.dataset import AudioDataset, get_data_loader
from common.features import BaseFeatures, FilterbankFeatures
from common.helpers import (Checkpointer, greedy_wer, num_weights, print_once,
                            process_evaluation_epoch)
from common.optimizers import AdamW, lr_policy, Novograd
from common.tb_dllogger import flush_log, init_log, log
from common.utils import BenchmarkStats
from jasper import config
from jasper.model import CTCLossNM, GreedyCTCDecoder, Jasper


def parse_args():
    parser = argparse.ArgumentParser(description='Jasper')

    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', default=400, type=int,
                          help='Number of epochs for the entire training; influences the lr schedule')
    training.add_argument("--warmup_epochs", default=0, type=int,
                          help='Initial epochs of increasing learning rate')
    training.add_argument("--hold_epochs", default=0, type=int,
                          help='Constant max learning rate epochs after warmup')
    training.add_argument('--epochs_this_job', default=0, type=int,
                          help=('Run for a number of epochs with no effect on the lr schedule.'
                                'Useful for re-starting the training.'))
    training.add_argument('--cudnn_benchmark', action='store_true', default=True,
                          help='Enable cudnn benchmark')
    training.add_argument('--amp', '--fp16', action='store_true', default=False,
                          help='Use pytorch native mixed precision training')
    training.add_argument('--seed', default=42, type=int, help='Random seed')
    training.add_argument('--local_rank', '--local-rank', default=os.getenv('LOCAL_RANK', 0),
                          type=int, help='GPU id used for distributed training')
    training.add_argument('--pre_allocate_range', default=None, type=int, nargs=2,
                          help='Warmup with batches of length [min, max] before training')

    optim = parser.add_argument_group('optimization setup')
    optim.add_argument('--batch_size', default=32, type=int,
                       help='Global batch size')
    optim.add_argument('--lr', default=1e-3, type=float,
                       help='Peak learning rate')
    optim.add_argument("--min_lr", default=1e-5, type=float,
                       help='minimum learning rate')
    optim.add_argument("--lr_policy", default='exponential', type=str,
                       choices=['exponential', 'legacy'], help='lr scheduler')
    optim.add_argument("--lr_exp_gamma", default=0.99, type=float,
                       help='gamma factor for exponential lr scheduler')
    optim.add_argument('--weight_decay', default=1e-3, type=float,
                       help='Weight decay for the optimizer')
    optim.add_argument('--grad_accumulation_steps', default=1, type=int,
                       help='Number of accumulation steps')
    optim.add_argument('--optimizer', default='novograd', type=str,
                       choices=['novograd', 'adamw'], help='Optimization algorithm')
    optim.add_argument('--ema', type=float, default=0.0,
                       help='Discount factor for exp averaging of model weights')

    io = parser.add_argument_group('feature and checkpointing setup')
    io.add_argument('--dali_device', type=str, choices=['none', 'cpu', 'gpu'],
                    default='gpu', help='Use DALI pipeline for fast data processing')
    io.add_argument('--resume', action='store_true',
                    help='Try to resume from last saved checkpoint.')
    io.add_argument('--ckpt', default=None, type=str,
                    help='Path to a checkpoint for resuming training')
    io.add_argument('--save_frequency', default=10, type=int,
                    help='Checkpoint saving frequency in epochs')
    io.add_argument('--keep_milestones', default=[100, 200, 300], type=int, nargs='+',
                    help='Milestone checkpoints to keep from removing')
    io.add_argument('--save_best_from', default=380, type=int,
                    help='Epoch on which to begin tracking best checkpoint (dev WER)')
    io.add_argument('--eval_frequency', default=200, type=int,
                    help='Number of steps between evaluations on dev set')
    io.add_argument('--log_frequency', default=25, type=int,
                    help='Number of steps between printing training stats')
    io.add_argument('--prediction_frequency', default=100, type=int,
                    help='Number of steps between printing sample decodings')
    io.add_argument('--model_config', type=str, required=True,
                    help='Path of the model configuration file')
    io.add_argument('--train_manifests', type=str, required=True, nargs='+',
                    help='Paths of the training dataset manifest file')
    io.add_argument('--val_manifests', type=str, required=True, nargs='+',
                    help='Paths of the evaluation datasets manifest files')
    io.add_argument('--dataset_dir', required=True, type=str,
                    help='Root dir of dataset')
    io.add_argument('--output_dir', type=str, required=True,
                    help='Directory for logs and checkpoints')
    io.add_argument('--log_file', type=str, default=None,
                    help='Path to save the training logfile.')
    io.add_argument('--benchmark_epochs_num', type=int, default=1,
                    help='Number of epochs accounted in final average throughput.')
    io.add_argument('--override_config', type=str, action='append',
                    help='Overrides a value from a config .yaml.'
                         ' Syntax: `--override_config nested.config.key=val`.')
    return parser.parse_args()


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt.true_divide(num_gpus)


def apply_ema(model, ema_model, decay):
    if not decay:
        return

    sd = getattr(model, 'module', model).state_dict()
    for k, v in ema_model.state_dict().items():
        v.copy_(decay * v + (1 - decay) * sd[k])


@torch.no_grad()
def evaluate(epoch, step, val_loader, val_feat_proc, labels, model,
             ema_model, ctc_loss, greedy_decoder, use_amp, use_dali=False):

    for model, subset in [(model, 'dev'), (ema_model, 'dev_ema')]:
        if model is None:
            continue

        model.eval()
        torch.cuda.synchronize()
        start_time = time.time()
        agg = {'losses': [], 'preds': [], 'txts': []}

        for batch in val_loader:
            if use_dali:
                # with DALI, the data is already on GPU
                feat, feat_lens, txt, txt_lens = batch
                if val_feat_proc is not None:
                    feat, feat_lens = val_feat_proc(feat, feat_lens)
            else:
                batch = [t.cuda(non_blocking=True) for t in batch]
                audio, audio_lens, txt, txt_lens = batch
                feat, feat_lens = val_feat_proc(audio, audio_lens)

            with torch.cuda.amp.autocast(enabled=use_amp):
                log_probs, enc_lens = model(feat, feat_lens)
                loss = ctc_loss(log_probs, txt, enc_lens, txt_lens)
                pred = greedy_decoder(log_probs)

            agg['losses'] += helpers.gather_losses([loss])
            agg['preds'] += helpers.gather_predictions([pred], labels)
            agg['txts'] += helpers.gather_transcripts([txt], [txt_lens], labels)

        wer, loss = process_evaluation_epoch(agg)
        torch.cuda.synchronize()
        log(() if epoch is None else (epoch,),
            step, subset, {'loss': loss, 'wer': 100.0 * wer,
                           'took': time.time() - start_time})
        model.train()
    return wer


def main():
    args = parse_args()

    assert(torch.cuda.is_available())
    assert args.prediction_frequency % args.log_frequency == 0

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # set up distributed training
    multi_gpu = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if multi_gpu:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        world_size = dist.get_world_size()
        print_once(f'Distributed training with {world_size} GPUs\n')
    else:
        world_size = 1

    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    random.seed(args.seed + args.local_rank)

    init_log(args)

    cfg = config.load(args.model_config)
    config.apply_config_overrides(cfg, args)

    symbols = helpers.add_ctc_blank(cfg['labels'])

    assert args.grad_accumulation_steps >= 1
    assert args.batch_size % args.grad_accumulation_steps == 0
    batch_size = args.batch_size // args.grad_accumulation_steps

    print_once('Setting up datasets...')
    train_dataset_kw, train_features_kw = config.input(cfg, 'train')
    val_dataset_kw, val_features_kw = config.input(cfg, 'val')

    use_dali = args.dali_device in ('cpu', 'gpu')
    if use_dali:
        assert train_dataset_kw['ignore_offline_speed_perturbation'], \
            "DALI doesn't support offline speed perturbation"

        # pad_to_max_duration is not supported by DALI - have simple padders
        if train_features_kw['pad_to_max_duration']:
            train_feat_proc = BaseFeatures(
                pad_align=train_features_kw['pad_align'],
                pad_to_max_duration=True,
                max_duration=train_features_kw['max_duration'],
                sample_rate=train_features_kw['sample_rate'],
                window_size=train_features_kw['window_size'],
                window_stride=train_features_kw['window_stride'])
            train_features_kw['pad_to_max_duration'] = False
        else:
            train_feat_proc = None

        if val_features_kw['pad_to_max_duration']:
            val_feat_proc = BaseFeatures(
                pad_align=val_features_kw['pad_align'],
                pad_to_max_duration=True,
                max_duration=val_features_kw['max_duration'],
                sample_rate=val_features_kw['sample_rate'],
                window_size=val_features_kw['window_size'],
                window_stride=val_features_kw['window_stride'])
            val_features_kw['pad_to_max_duration'] = False
        else:
            val_feat_proc = None

        train_loader = DaliDataLoader(gpu_id=args.local_rank,
                                      dataset_path=args.dataset_dir,
                                      config_data=train_dataset_kw,
                                      config_features=train_features_kw,
                                      json_names=args.train_manifests,
                                      batch_size=batch_size,
                                      grad_accumulation_steps=args.grad_accumulation_steps,
                                      pipeline_type="train",
                                      device_type=args.dali_device,
                                      symbols=symbols)

        val_loader = DaliDataLoader(gpu_id=args.local_rank,
                                    dataset_path=args.dataset_dir,
                                    config_data=val_dataset_kw,
                                    config_features=val_features_kw,
                                    json_names=args.val_manifests,
                                    batch_size=batch_size,
                                    pipeline_type="val",
                                    device_type=args.dali_device,
                                    symbols=symbols)
    else:
        train_dataset_kw, train_features_kw = config.input(cfg, 'train')
        train_dataset = AudioDataset(args.dataset_dir,
                                     args.train_manifests,
                                     symbols,
                                     **train_dataset_kw)
        train_loader = get_data_loader(train_dataset,
                                       batch_size,
                                       multi_gpu=multi_gpu,
                                       shuffle=True,
                                       num_workers=4)
        train_feat_proc = FilterbankFeatures(**train_features_kw)

        val_dataset_kw, val_features_kw = config.input(cfg, 'val')
        val_dataset = AudioDataset(args.dataset_dir,
                                   args.val_manifests,
                                   symbols,
                                   **val_dataset_kw)
        val_loader = get_data_loader(val_dataset,
                                     batch_size,
                                     multi_gpu=multi_gpu,
                                     shuffle=False,
                                     num_workers=4,
                                     drop_last=False)
        val_feat_proc = FilterbankFeatures(**val_features_kw)

        dur = train_dataset.duration / 3600
        dur_f = train_dataset.duration_filtered / 3600
        nsampl = len(train_dataset)
        print_once(f'Training samples: {nsampl} ({dur:.1f}h, '
                   f'filtered {dur_f:.1f}h)')

    if train_feat_proc is not None:
        train_feat_proc.cuda()
    if val_feat_proc is not None:
        val_feat_proc.cuda()

    steps_per_epoch = len(train_loader) // args.grad_accumulation_steps

    # set up the model
    model = Jasper(encoder_kw=config.encoder(cfg),
                   decoder_kw=config.decoder(cfg, n_classes=len(symbols)))
    model.cuda()
    ctc_loss = CTCLossNM(n_classes=len(symbols))
    greedy_decoder = GreedyCTCDecoder()

    print_once(f'Model size: {num_weights(model) / 10**6:.1f}M params\n')

    # optimization
    kw = {'lr': args.lr, 'weight_decay': args.weight_decay}
    if args.optimizer == "novograd":
        optimizer = Novograd(model.parameters(), **kw)
    elif args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), **kw)
    else:
        raise ValueError(f'Invalid optimizer "{args.optimizer}"')

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    adjust_lr = lambda step, epoch, optimizer: lr_policy(
        step, epoch, args.lr, optimizer, steps_per_epoch=steps_per_epoch,
        warmup_epochs=args.warmup_epochs, hold_epochs=args.hold_epochs,
        num_epochs=args.epochs, policy=args.lr_policy, min_lr=args.min_lr,
        exp_gamma=args.lr_exp_gamma)

    if args.ema > 0:
        ema_model = copy.deepcopy(model)
    else:
        ema_model = None

    if multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)

    # load checkpoint
    meta = {'best_wer': 10**6, 'start_epoch': 0}
    checkpointer = Checkpointer(args.output_dir, 'Jasper',
                                args.keep_milestones)
    if args.resume:
        args.ckpt = checkpointer.last_checkpoint() or args.ckpt

    if args.ckpt is not None:
        checkpointer.load(args.ckpt, model, ema_model, optimizer, scaler, meta)

    start_epoch = meta['start_epoch']
    best_wer = meta['best_wer']
    epoch = 1
    step = start_epoch * steps_per_epoch + 1

    # training loop
    model.train()

    # pre-allocate
    if args.pre_allocate_range is not None:
        n_feats = train_features_kw['n_filt']
        pad_align = train_features_kw['pad_align']
        a, b = args.pre_allocate_range
        for n_frames in range(a, b + pad_align, pad_align):
            print_once(f'Pre-allocation ({batch_size}x{n_feats}x{n_frames})...')

            feat = torch.randn(batch_size, n_feats, n_frames, device='cuda')
            feat_lens = torch.ones(batch_size, device='cuda').fill_(n_frames)
            txt = torch.randint(high=len(symbols)-1, size=(batch_size, 100),
                                device='cuda')
            txt_lens = torch.ones(batch_size, device='cuda').fill_(100)
            with torch.cuda.amp.autocast(enabled=args.amp):
                log_probs, enc_lens = model(feat, feat_lens)
                del feat
                loss = ctc_loss(log_probs, txt, enc_lens, txt_lens)
            loss.backward()
            model.zero_grad()
    torch.cuda.empty_cache()

    bmark_stats = BenchmarkStats()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if multi_gpu and not use_dali:
            train_loader.sampler.set_epoch(epoch)

        torch.cuda.synchronize()
        epoch_start_time = time.time()
        epoch_utts = 0
        epoch_loss = 0
        accumulated_batches = 0

        for batch in train_loader:

            if accumulated_batches == 0:
                step_loss = 0
                step_utts = 0
                step_start_time = time.time()

            if use_dali:
                # with DALI, the data is already on GPU
                feat, feat_lens, txt, txt_lens = batch
                if train_feat_proc is not None:
                    feat, feat_lens = train_feat_proc(feat, feat_lens)
            else:
                batch = [t.cuda(non_blocking=True) for t in batch]
                audio, audio_lens, txt, txt_lens = batch
                feat, feat_lens = train_feat_proc(audio, audio_lens)

            # Use context manager to prevent redundant accumulation of gradients
            if (multi_gpu and accumulated_batches + 1 < args.grad_accumulation_steps):
                ctx = model.no_sync()
            else:
                ctx = empty_context()

            with ctx:
                with torch.cuda.amp.autocast(enabled=args.amp):
                    log_probs, enc_lens = model(feat, feat_lens)

                    loss = ctc_loss(log_probs, txt, enc_lens, txt_lens)
                    loss /= args.grad_accumulation_steps

                if multi_gpu:
                    reduced_loss = reduce_tensor(loss.data, world_size)
                else:
                    reduced_loss = loss

                if torch.isnan(reduced_loss).any():
                    print_once(f'WARNING: loss is NaN; skipping update')
                    continue
                else:
                    step_loss += reduced_loss.item()
                    step_utts += batch[0].size(0) * world_size
                    epoch_utts += batch[0].size(0) * world_size
                    accumulated_batches += 1

                    scaler.scale(loss).backward()

            if accumulated_batches % args.grad_accumulation_steps == 0:
                epoch_loss += step_loss
                scaler.step(optimizer)
                scaler.update()

                adjust_lr(step, epoch, optimizer)
                optimizer.zero_grad()

                apply_ema(model, ema_model, args.ema)

                if step % args.log_frequency == 0:
                    preds = greedy_decoder(log_probs)
                    wer, pred_utt, ref = greedy_wer(preds, txt, txt_lens, symbols)

                    if step % args.prediction_frequency == 0:
                        print_once(f'  Decoded:   {pred_utt[:90]}')
                        print_once(f'  Reference: {ref[:90]}')

                    step_time = time.time() - step_start_time
                    log((epoch, step % steps_per_epoch or steps_per_epoch, steps_per_epoch),
                        step, 'train',
                        {'loss': step_loss,
                         'wer': 100.0 * wer,
                         'throughput': step_utts / step_time,
                         'took': step_time,
                         'lrate': optimizer.param_groups[0]['lr']})

                step_start_time = time.time()

                if step % args.eval_frequency == 0:
                    wer = evaluate(epoch, step, val_loader, val_feat_proc,
                                   symbols, model, ema_model, ctc_loss,
                                   greedy_decoder, args.amp, use_dali)

                    if wer < best_wer and epoch >= args.save_best_from:
                        checkpointer.save(model, ema_model, optimizer, scaler,
                                          epoch, step, best_wer, is_best=True)
                        best_wer = wer

                step += 1
                accumulated_batches = 0
                # end of step

            # DALI iterator need to be exhausted;
            # if not using DALI, simulate drop_last=True with grad accumulation
            if not use_dali and step > steps_per_epoch * epoch:
                break

        torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start_time
        epoch_loss /= steps_per_epoch
        log((epoch,), None, 'train_avg', {'throughput': epoch_utts / epoch_time,
                                          'took': epoch_time,
                                          'loss': epoch_loss})
        bmark_stats.update(epoch_utts, epoch_time, epoch_loss)

        if epoch % args.save_frequency == 0 or epoch in args.keep_milestones:
            checkpointer.save(model, ema_model, optimizer, scaler, epoch, step,
                              best_wer)

        if 0 < args.epochs_this_job <= epoch - start_epoch:
            print_once(f'Finished after {args.epochs_this_job} epochs.')
            break
        # end of epoch

    log((), None, 'train_avg', bmark_stats.get(args.benchmark_epochs_num))

    evaluate(None, step, val_loader, val_feat_proc, symbols, model,
             ema_model, ctc_loss, greedy_decoder, args.amp, use_dali)

    if epoch == args.epochs:
        checkpointer.save(model, ema_model, optimizer, scaler, epoch, step,
                          best_wer)
    flush_log()


if __name__ == "__main__":
    main()
