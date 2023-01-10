# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import copy
import os
import time
from collections import defaultdict, OrderedDict
from itertools import cycle

import numpy as np
import torch
import torch.distributed as dist
import amp_C
from apex.optimizers import FusedAdam, FusedLAMB
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import common.tb_dllogger as logger
import models
from common.tb_dllogger import log
from common.repeated_dataloader import (RepeatedDataLoader,
                                        RepeatedDistributedSampler)
from common.text import cmudict
from common.utils import (BenchmarkStats, Checkpointer,
                          load_pretrained_weights, prepare_tmp)
from fastpitch.attn_loss_function import AttentionBinarizationLoss
from fastpitch.data_function import batch_to_gpu, ensure_disjoint, TTSCollate, TTSDataset
from fastpitch.loss_function import FastPitchLoss


def parse_args(parser):
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str, default='./',
                        help='Path to dataset')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')

    train = parser.add_argument_group('training setup')
    train.add_argument('--epochs', type=int, required=True,
                       help='Number of total epochs to run')
    train.add_argument('--epochs-per-checkpoint', type=int, default=50,
                       help='Number of epochs per checkpoint')
    train.add_argument('--checkpoint-path', type=str, default=None,
                       help='Checkpoint path to resume training')
    train.add_argument('--keep-milestones', default=list(range(100, 1000, 100)),
                       type=int, nargs='+',
                       help='Milestone checkpoints to keep from removing')
    train.add_argument('--resume', action='store_true',
                       help='Resume training from the last checkpoint')
    train.add_argument('--seed', type=int, default=1234,
                       help='Seed for PyTorch random number generators')
    train.add_argument('--amp', action='store_true',
                       help='Enable AMP')
    train.add_argument('--cuda', action='store_true',
                       help='Run on GPU using CUDA')
    train.add_argument('--cudnn-benchmark', action='store_true',
                       help='Enable cudnn benchmark mode')
    train.add_argument('--ema-decay', type=float, default=0,
                       help='Discounting factor for training weights EMA')
    train.add_argument('--grad-accumulation', type=int, default=1,
                       help='Training steps to accumulate gradients for')
    train.add_argument('--kl-loss-start-epoch', type=int, default=250,
                       help='Start adding the hard attention loss term')
    train.add_argument('--kl-loss-warmup-epochs', type=int, default=100,
                       help='Gradually increase the hard attention loss term')
    train.add_argument('--kl-loss-weight', type=float, default=1.0,
                       help='Gradually increase the hard attention loss term')
    train.add_argument('--benchmark-epochs-num', type=int, default=20,
                        help='Number of epochs for calculating final stats')
    train.add_argument('--validation-freq', type=int, default=1,
                       help='Validate every N epochs to use less compute')
    train.add_argument('--init-from-checkpoint', type=str, default=None,
                       help='Initialize model weights with a pre-trained ckpt')

    opt = parser.add_argument_group('optimization setup')
    opt.add_argument('--optimizer', type=str, default='lamb',
                     help='Optimization algorithm')
    opt.add_argument('-lr', '--learning-rate', type=float, required=True,
                     help='Learing rate')
    opt.add_argument('--weight-decay', default=1e-6, type=float,
                     help='Weight decay')
    opt.add_argument('--grad-clip-thresh', default=1000.0, type=float,
                     help='Clip threshold for gradients')
    opt.add_argument('-bs', '--batch-size', type=int, required=True,
                     help='Batch size per GPU')
    opt.add_argument('--warmup-steps', type=int, default=1000,
                     help='Number of steps for lr warmup')
    opt.add_argument('--dur-predictor-loss-scale', type=float,
                     default=1.0, help='Rescale duration predictor loss')
    opt.add_argument('--pitch-predictor-loss-scale', type=float,
                     default=1.0, help='Rescale pitch predictor loss')
    opt.add_argument('--attn-loss-scale', type=float,
                     default=1.0, help='Rescale alignment loss')

    data = parser.add_argument_group('dataset parameters')
    data.add_argument('--training-files', type=str, nargs='*', required=True,
                      help='Paths to training filelists.')
    data.add_argument('--validation-files', type=str, nargs='*',
                      required=True, help='Paths to validation filelists')
    data.add_argument('--text-cleaners', nargs='*',
                      default=['english_cleaners'], type=str,
                      help='Type of text cleaners for input text')
    data.add_argument('--symbol-set', type=str, default='english_basic',
                      help='Define symbol set for input text')
    data.add_argument('--p-arpabet', type=float, default=0.0,
                      help='Probability of using arpabets instead of graphemes '
                           'for each word; set 0 for pure grapheme training')
    data.add_argument('--heteronyms-path', type=str, default='cmudict/heteronyms',
                      help='Path to the list of heteronyms')
    data.add_argument('--cmudict-path', type=str, default='cmudict/cmudict-0.7b',
                      help='Path to the pronouncing dictionary')
    data.add_argument('--prepend-space-to-text', action='store_true',
                      help='Capture leading silence with a space token')
    data.add_argument('--append-space-to-text', action='store_true',
                      help='Capture trailing silence with a space token')
    data.add_argument('--num-workers', type=int, default=6,
                      help='Subprocesses for train and val DataLoaders')
    data.add_argument('--trainloader-repeats', type=int, default=100,
                      help='Repeats the dataset to prolong epochs')

    cond = parser.add_argument_group('data for conditioning')
    cond.add_argument('--n-speakers', type=int, default=1,
                      help='Number of speakers in the dataset. '
                           'n_speakers > 1 enables speaker embeddings')
    cond.add_argument('--load-pitch-from-disk', action='store_true',
                      help='Use pitch cached on disk with prepare_dataset.py')
    cond.add_argument('--pitch-online-method', default='pyin',
                      choices=['pyin'],
                      help='Calculate pitch on the fly during trainig')
    cond.add_argument('--pitch-online-dir', type=str, default=None,
                      help='A directory for storing pitch calculated on-line')
    cond.add_argument('--pitch-mean', type=float, default=214.72203,
                      help='Normalization value for pitch')
    cond.add_argument('--pitch-std', type=float, default=65.72038,
                      help='Normalization value for pitch')
    cond.add_argument('--load-mel-from-disk', action='store_true',
                      help='Use mel-spectrograms cache on the disk')  # XXX

    audio = parser.add_argument_group('audio parameters')
    audio.add_argument('--max-wav-value', default=32768.0, type=float,
                       help='Maximum audiowave value')
    audio.add_argument('--sampling-rate', default=22050, type=int,
                       help='Sampling rate')
    audio.add_argument('--filter-length', default=1024, type=int,
                       help='Filter length')
    audio.add_argument('--hop-length', default=256, type=int,
                       help='Hop (stride) length')
    audio.add_argument('--win-length', default=1024, type=int,
                       help='Window length')
    audio.add_argument('--mel-fmin', default=0.0, type=float,
                       help='Minimum mel frequency')
    audio.add_argument('--mel-fmax', default=8000.0, type=float,
                       help='Maximum mel frequency')

    dist = parser.add_argument_group('distributed setup')
    dist.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0),
                      help='Rank of the process for multiproc; do not set manually')
    dist.add_argument('--world_size', type=int, default=os.getenv('WORLD_SIZE', 1),
                      help='Number of processes for multiproc; do not set manually')
    return parser


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt.true_divide(num_gpus)


def init_distributed(args, world_size, rank):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing distributed training")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(backend=('nccl' if args.cuda else 'gloo'),
                            init_method='env://')
    print("Done initializing distributed training")


def validate(model, epoch, total_iter, criterion, val_loader, distributed_run,
             batch_to_gpu, ema=False):
    was_training = model.training
    model.eval()

    tik = time.perf_counter()
    with torch.no_grad():
        val_meta = defaultdict(float)
        val_num_frames = 0
        for i, batch in enumerate(val_loader):
            x, y, num_frames = batch_to_gpu(batch)
            y_pred = model(x)
            loss, meta = criterion(y_pred, y, is_training=False, meta_agg='sum')

            if distributed_run:
                for k, v in meta.items():
                    val_meta[k] += reduce_tensor(v, 1)
                val_num_frames += reduce_tensor(num_frames.data, 1).item()
            else:
                for k, v in meta.items():
                    val_meta[k] += v
                val_num_frames += num_frames.item()

        val_meta = {k: v / len(val_loader.dataset) for k, v in val_meta.items()}

    val_meta['took'] = time.perf_counter() - tik

    log((epoch,) if epoch is not None else (), tb_total_steps=total_iter,
        subset='val_ema' if ema else 'val',
        data=OrderedDict([
            ('loss', val_meta['loss'].item()),
            ('mel_loss', val_meta['mel_loss'].item()),
            ('frames/s', val_num_frames / val_meta['took']),
            ('took', val_meta['took'])]),
        )

    if was_training:
        model.train()
    return val_meta


def adjust_learning_rate(total_iter, opt, learning_rate, warmup_iters=None):
    if warmup_iters == 0:
        scale = 1.0
    elif total_iter > warmup_iters:
        scale = 1. / (total_iter ** 0.5)
    else:
        scale = total_iter / (warmup_iters ** 1.5)

    for param_group in opt.param_groups:
        param_group['lr'] = learning_rate * scale


def apply_ema_decay(model, ema_model, decay):
    if not decay:
        return
    st = model.state_dict()
    add_module = hasattr(model, 'module') and not hasattr(ema_model, 'module')
    for k, v in ema_model.state_dict().items():
        if add_module and not k.startswith('module.'):
            k = 'module.' + k
        v.copy_(decay * v + (1 - decay) * st[k])


def init_multi_tensor_ema(model, ema_model):
    model_weights = list(model.state_dict().values())
    ema_model_weights = list(ema_model.state_dict().values())
    ema_overflow_buf = torch.cuda.IntTensor([0])
    return model_weights, ema_model_weights, ema_overflow_buf


def apply_multi_tensor_ema(decay, model_weights, ema_weights, overflow_buf):
    amp_C.multi_tensor_axpby(
        65536, overflow_buf, [ema_weights, model_weights, ema_weights],
        decay, 1-decay, -1)


def main():
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Training',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    if args.p_arpabet > 0.0:
        cmudict.initialize(args.cmudict_path, args.heteronyms_path)

    distributed_run = args.world_size > 1

    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)

    if args.local_rank == 0:
        if not os.path.exists(args.output):
            os.makedirs(args.output)

    log_fpath = args.log_file or os.path.join(args.output, 'nvlog.json')
    tb_subsets = ['train', 'val']
    if args.ema_decay > 0.0:
        tb_subsets.append('val_ema')

    logger.init(log_fpath, args.output, enabled=(args.local_rank == 0),
                tb_subsets=tb_subsets)
    logger.parameters(vars(args), tb_subset='train')

    parser = models.parse_model_args('FastPitch', parser)
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if distributed_run:
        init_distributed(args, args.world_size, args.local_rank)
    else:
        if args.trainloader_repeats > 1:
            print('WARNING: Disabled --trainloader-repeats, supported only for'
                  ' multi-GPU data loading.')
            args.trainloader_repeats = 1

    device = torch.device('cuda' if args.cuda else 'cpu')
    model_config = models.get_model_config('FastPitch', args)
    model = models.get_model('FastPitch', model_config, device)

    if args.init_from_checkpoint is not None:
        load_pretrained_weights(model, args.init_from_checkpoint)

    attention_kl_loss = AttentionBinarizationLoss()

    # Store pitch mean/std as params to translate from Hz during inference
    model.pitch_mean[0] = args.pitch_mean
    model.pitch_std[0] = args.pitch_std

    kw = dict(lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9,
              weight_decay=args.weight_decay)
    if args.optimizer == 'adam':
        optimizer = FusedAdam(model.parameters(), **kw)
    elif args.optimizer == 'lamb':
        optimizer = FusedLAMB(model.parameters(), **kw)
    else:
        raise ValueError

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model)
    else:
        ema_model = None

    if distributed_run:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True)

    train_state = {'epoch': 1, 'total_iter': 1}
    checkpointer = Checkpointer(args.output, args.keep_milestones)

    checkpointer.maybe_load(model, optimizer, scaler, train_state, args,
                            ema_model)

    start_epoch = train_state['epoch']
    total_iter = train_state['total_iter']

    criterion = FastPitchLoss(
        dur_predictor_loss_scale=args.dur_predictor_loss_scale,
        pitch_predictor_loss_scale=args.pitch_predictor_loss_scale,
        attn_loss_scale=args.attn_loss_scale)

    collate_fn = TTSCollate()

    if args.local_rank == 0:
        prepare_tmp(args.pitch_online_dir)

    trainset = TTSDataset(audiopaths_and_text=args.training_files, **vars(args))
    valset = TTSDataset(audiopaths_and_text=args.validation_files, **vars(args))
    ensure_disjoint(trainset, valset)

    if distributed_run:
        train_sampler = RepeatedDistributedSampler(args.trainloader_repeats,
                                                   trainset, drop_last=True)
        val_sampler = DistributedSampler(valset)
        shuffle = False
    else:
        train_sampler, val_sampler, shuffle = None, None, True

    # 4 workers are optimal on DGX-1 (from epoch 2 onwards)
    kw = {'num_workers': args.num_workers, 'batch_size': args.batch_size,
          'collate_fn': collate_fn}
    train_loader = RepeatedDataLoader(args.trainloader_repeats, trainset,
                                      shuffle=shuffle, drop_last=True,
                                      sampler=train_sampler, pin_memory=True,
                                      persistent_workers=True, **kw)
    val_loader = DataLoader(valset, shuffle=False, sampler=val_sampler,
                            pin_memory=False, **kw)
    if args.ema_decay:
        mt_ema_params = init_multi_tensor_ema(model, ema_model)

    model.train()
    bmark_stats = BenchmarkStats()

    torch.cuda.synchronize()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.perf_counter()

        epoch_loss = 0.0
        epoch_mel_loss = 0.0
        epoch_num_frames = 0
        epoch_frames_per_sec = 0.0

        if distributed_run:
            train_loader.sampler.set_epoch(epoch)

        iter_loss = 0
        iter_num_frames = 0
        iter_meta = {}
        iter_start_time = time.perf_counter()

        epoch_iter = 1
        for batch, accum_step in zip(train_loader,
                                     cycle(range(1, args.grad_accumulation + 1))):
            if accum_step == 1:
                adjust_learning_rate(total_iter, optimizer, args.learning_rate,
                                     args.warmup_steps)

                model.zero_grad(set_to_none=True)

            x, y, num_frames = batch_to_gpu(batch)

            with torch.cuda.amp.autocast(enabled=args.amp):
                y_pred = model(x)
                loss, meta = criterion(y_pred, y)

                if (args.kl_loss_start_epoch is not None
                        and epoch >= args.kl_loss_start_epoch):

                    if args.kl_loss_start_epoch == epoch and epoch_iter == 1:
                        print('Begin hard_attn loss')

                    _, _, _, _, _, _, _, _, attn_soft, attn_hard, _, _ = y_pred
                    binarization_loss = attention_kl_loss(attn_hard, attn_soft)
                    kl_weight = min((epoch - args.kl_loss_start_epoch) / args.kl_loss_warmup_epochs, 1.0) * args.kl_loss_weight
                    meta['kl_loss'] = binarization_loss.clone().detach() * kl_weight
                    loss += kl_weight * binarization_loss

                else:
                    meta['kl_loss'] = torch.zeros_like(loss)
                    kl_weight = 0
                    binarization_loss = 0

                loss /= args.grad_accumulation

            meta = {k: v / args.grad_accumulation
                    for k, v in meta.items()}

            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if distributed_run:
                reduced_loss = reduce_tensor(loss.data, args.world_size).item()
                reduced_num_frames = reduce_tensor(num_frames.data, 1).item()
                meta = {k: reduce_tensor(v, args.world_size) for k, v in meta.items()}
            else:
                reduced_loss = loss.item()
                reduced_num_frames = num_frames.item()
            if np.isnan(reduced_loss):
                raise Exception("loss is NaN")

            iter_loss += reduced_loss
            iter_num_frames += reduced_num_frames
            iter_meta = {k: iter_meta.get(k, 0) + meta.get(k, 0) for k in meta}

            if accum_step % args.grad_accumulation == 0:

                logger.log_grads_tb(total_iter, model)
                if args.amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_thresh)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_thresh)
                    optimizer.step()

                if args.ema_decay > 0.0:
                    apply_multi_tensor_ema(args.ema_decay, *mt_ema_params)

                iter_mel_loss = iter_meta['mel_loss'].item()
                iter_kl_loss = iter_meta['kl_loss'].item()
                iter_time = time.perf_counter() - iter_start_time
                epoch_frames_per_sec += iter_num_frames / iter_time
                epoch_loss += iter_loss
                epoch_num_frames += iter_num_frames
                epoch_mel_loss += iter_mel_loss

                num_iters = len(train_loader) // args.grad_accumulation
                log((epoch, epoch_iter, num_iters), tb_total_steps=total_iter,
                    subset='train', data=OrderedDict([
                        ('loss', iter_loss),
                        ('mel_loss', iter_mel_loss),
                        ('kl_loss', iter_kl_loss),
                        ('kl_weight', kl_weight),
                        ('frames/s', iter_num_frames / iter_time),
                        ('took', iter_time),
                        ('lrate', optimizer.param_groups[0]['lr'])]),
                )

                iter_loss = 0
                iter_num_frames = 0
                iter_meta = {}
                iter_start_time = time.perf_counter()

                if epoch_iter == num_iters:
                    break
                epoch_iter += 1
                total_iter += 1

        # Finished epoch
        epoch_loss /= epoch_iter
        epoch_mel_loss /= epoch_iter
        epoch_time = time.perf_counter() - epoch_start_time

        log((epoch,), tb_total_steps=None, subset='train_avg',
            data=OrderedDict([
                ('loss', epoch_loss),
                ('mel_loss', epoch_mel_loss),
                ('frames/s', epoch_num_frames / epoch_time),
                ('took', epoch_time)]),
        )
        bmark_stats.update(epoch_num_frames, epoch_loss, epoch_mel_loss,
                           epoch_time)

        if epoch % args.validation_freq == 0:
            validate(model, epoch, total_iter, criterion, val_loader,
                 distributed_run, batch_to_gpu)

            if args.ema_decay > 0:
                validate(ema_model, epoch, total_iter, criterion, val_loader,
                         distributed_run, batch_to_gpu, ema=True)

        # save before making sched.step() for proper loading of LR
        checkpointer.maybe_save(args, model, ema_model, optimizer, scaler,
                                epoch, total_iter, model_config)
        logger.flush()

    # Finished training
    if len(bmark_stats) > 0:
        log((), tb_total_steps=None, subset='train_avg',
            data=bmark_stats.get(args.benchmark_epochs_num))

    validate(model, None, total_iter, criterion, val_loader, distributed_run,
             batch_to_gpu)


if __name__ == '__main__':
    main()
