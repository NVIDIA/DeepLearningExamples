# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
import itertools
import os
from functools import partial
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from apex.optimizers import FusedAdam, FusedLAMB

import models
from common import tb_dllogger as logger, utils, gpu_affinity
from common.utils import (Checkpointer, freeze, init_distributed, print_once,
                          reduce_tensor, unfreeze, l2_promote)

from hifigan.data_function import get_data_loader, mel_spectrogram
from hifigan.logging import init_logger, Metrics
from hifigan.models import (MultiPeriodDiscriminator, MultiScaleDiscriminator,
                            feature_loss, generator_loss, discriminator_loss)


def parse_args(parser):
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to a DLLogger log file')

    train = parser.add_argument_group('training setup')
    train.add_argument('--epochs', type=int, required=True,
                       help='Number of total epochs to run')
    train.add_argument('--epochs_this_job', type=int, default=None,
                       help='Number of epochs in partial training run')
    train.add_argument('--keep_milestones', type=int, nargs='+',
                       default=[1000, 2000, 3000, 4000, 5000, 6000],
                       help='Milestone checkpoints to keep from removing')
    train.add_argument('--checkpoint_interval', type=int, default=50,
                       help='Saving checkpoints frequency (in epochs)')
    train.add_argument('--step_logs_interval', default=1, type=int,
                       help='Step logs dumping frequency (in steps)')
    train.add_argument('--validation_interval', default=10, type=int,
                       help='Validation frequency (in epochs)')
    train.add_argument('--samples_interval', default=100, type=int,
                       help='Dumping audio samples frequency (in epochs)')
    train.add_argument('--resume', action='store_true',
                       help='Resume training from the last checkpoint')
    train.add_argument('--checkpoint_path_gen', type=str, default=None,
                       help='Resume training from a selected checkpoint')
    train.add_argument('--checkpoint_path_discrim', type=str, default=None,
                       help='Resume training from a selected checkpoint')
    train.add_argument('--seed', type=int, default=1234,
                       help='Seed for PyTorch random number generators')
    train.add_argument('--amp', action='store_true',
                       help='Enable AMP')
    train.add_argument('--autocast_spectrogram', action='store_true',
                       help='Enable autocast while computing spectrograms')
    train.add_argument('--cuda', action='store_true',
                       help='Run on GPU using CUDA')
    train.add_argument('--disable_cudnn_benchmark', action='store_true',
                       help='Disable cudnn benchmark mode')
    train.add_argument('--ema_decay', type=float, default=0,
                       help='Discounting factor for training weights EMA')
    train.add_argument('--grad_accumulation', type=int, default=1,
                       help='Training steps to accumulate gradients for')
    train.add_argument('--num_workers', type=int, default=1,
                       help='Data loader workers number')
    train.add_argument('--fine_tuning', action='store_true',
                       help='Enable fine-tuning')
    train.add_argument('--input_mels_dir', type=str, default=None,
                       help='Directory with mels for fine-tuning')
    train.add_argument('--benchmark_epochs_num', type=int, default=5)
    train.add_argument('--no_amp_grouped_conv', action='store_true',
                       help='Disable AMP on certain convs for better perf')

    opt = parser.add_argument_group('optimization setup')
    opt.add_argument('--optimizer', type=str, default='adamw',
                     help='Optimization algorithm')
    opt.add_argument('--lr_decay', type=float, default=0.9998,
                     help='Learning rate decay')
    opt.add_argument('-lr', '--learning_rate', type=float, required=True,
                     help='Learning rate')
    opt.add_argument('--fine_tune_lr_factor', type=float, default=1.,
                     help='Learning rate multiplier for fine-tuning')
    opt.add_argument('--adam_betas', type=float, nargs=2, default=(0.8, 0.99),
                     help='Adam Beta coefficients')
    opt.add_argument('--grad_clip_thresh', default=1000.0, type=float,
                     help='Clip threshold for gradients')
    opt.add_argument('-bs', '--batch_size', type=int, required=True,
                     help=('Batch size per training iter. '
                           'May be split into grad accumulation steps.'))
    opt.add_argument('--warmup_steps', type=int, default=1000,
                     help='Number of steps for lr warmup')

    data = parser.add_argument_group('dataset parameters')
    data.add_argument('-d', '--dataset_path', default='data/LJSpeech-1.1',
                      help='Path to dataset', type=str)
    data.add_argument('--training_files', type=str, required=True, nargs='+',
                      help='Paths to training filelists.')
    data.add_argument('--validation_files', type=str, required=True, nargs='+',
                      help='Paths to validation filelists.')

    audio = parser.add_argument_group('audio parameters')
    audio.add_argument('--max_wav_value', default=32768.0, type=float,
                       help='Maximum audiowave value')
    audio.add_argument('--sampling_rate', default=22050, type=int,
                       help='Sampling rate')
    audio.add_argument('--filter_length', default=1024, type=int,
                       help='Filter length')
    audio.add_argument('--num_mels', default=80, type=int,
                       help='number of Mel bands')
    audio.add_argument('--hop_length', default=256, type=int,
                       help='Hop (stride) length')
    audio.add_argument('--win_length', default=1024, type=int,
                       help='Window length')
    audio.add_argument('--mel_fmin', default=0.0, type=float,
                       help='Minimum mel frequency')
    audio.add_argument('--mel_fmax', default=8000.0, type=float,
                       help='Maximum mel frequency')
    audio.add_argument('--mel_fmax_loss', default=None, type=float,
                       help='Maximum mel frequency used for computing loss')
    audio.add_argument('--segment_size', default=8192, type=int,
                       help='Training segment size')

    dist = parser.add_argument_group('distributed setup')
    dist.add_argument(
        '--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0),
        help='Rank of the process for multiproc. Do not set manually.')
    dist.add_argument(
        '--world_size', type=int, default=os.getenv('WORLD_SIZE', 1),
        help='Number of processes for multiproc. Do not set manually.')
    dist.add_argument('--affinity', type=str,
                      default='socket_unique_interleaved',
                      choices=['socket', 'single', 'single_unique',
                               'socket_unique_interleaved',
                               'socket_unique_continuous',
                               'disabled'],
                      help='type of CPU affinity')

    return parser


def validate(args, gen, mel_spec, mpd, msd, val_loader, val_metrics):
    gen.eval()
    val_metrics.start_val()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x, y, _, y_mel = batch

            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).unsqueeze(1)
            y_mel = y_mel.cuda(non_blocking=True)

            with autocast(enabled=args.amp):
                y_g_hat = gen(x)
            with autocast(enabled=args.amp and args.autocast_spectrogram):
                y_g_hat_mel = mel_spec(y_g_hat.float().squeeze(1),
                                       fmax=args.mel_fmax_loss)

            with autocast(enabled=args.amp):
                # val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item() * 45
                # NOTE: Scale by 45.0 to match train loss magnitude
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f = discriminator_loss(y_df_hat_r, y_df_hat_g)

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                loss_disc_s = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            val_metrics['loss_discrim'] = reduce_tensor(
                loss_disc_s + loss_disc_f, args.world_size)
            val_metrics['loss_gen'] = reduce_tensor(loss_gen_all,
                                                    args.world_size)
            val_metrics['loss_mel'] = reduce_tensor(loss_mel, args.world_size)
            val_metrics['frames'] = x.size(0) * x.size(1) * args.world_size
            val_metrics.accumulate(scopes=['val'])

        val_metrics.finish_val()
        gen.train()


def main():
    parser = argparse.ArgumentParser(description='PyTorch HiFi-GAN Training',
                                     allow_abbrev=False)
    parser = models.parse_model_args('HiFi-GAN', parse_args(parser))
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    if args.affinity != 'disabled':
        nproc_per_node = torch.cuda.device_count()
        print(nproc_per_node)
        affinity = gpu_affinity.set_affinity(
            args.local_rank,
            nproc_per_node,
            args.affinity
        )
        print(f'{args.local_rank}: thread affinity: {affinity}')

    # seeds, distributed init, logging, cuDNN
    distributed_run = args.world_size > 1

    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)

    if distributed_run:
        init_distributed(args, args.world_size, args.local_rank)

    metrics = Metrics(scopes=['train', 'train_avg'],
                      benchmark_epochs=args.benchmark_epochs_num,
                      cuda=args.cuda)
    val_metrics = Metrics(scopes=['val'], cuda=args.cuda)
    init_logger(args.output, args.log_file, args.ema_decay)
    logger.parameters(vars(args), tb_subset='train')

    l2_promote()
    torch.backends.cudnn.benchmark = not args.disable_cudnn_benchmark

    train_setup = models.get_model_train_setup('HiFi-GAN', args)
    gen_config = models.get_model_config('HiFi-GAN', args)
    gen = models.get_model('HiFi-GAN', gen_config, 'cuda')

    mpd = MultiPeriodDiscriminator(periods=args.mpd_periods,
                                   concat_fwd=args.concat_fwd).cuda()

    assert args.amp or not args.no_amp_grouped_conv, \
        "--no-amp-grouped-conv is applicable only when AMP is enabled"
    msd = MultiScaleDiscriminator(concat_fwd=args.concat_fwd,
                                  no_amp_grouped_conv=args.no_amp_grouped_conv)
    msd = msd.cuda()

    mel_spec = partial(mel_spectrogram, n_fft=args.filter_length,
                       num_mels=args.num_mels,
                       sampling_rate=args.sampling_rate,
                       hop_size=args.hop_length, win_size=args.win_length,
                       fmin=args.mel_fmin)

    kw = {'lr': args.learning_rate, 'betas': args.adam_betas}
    proto = {'adam': FusedAdam, 'lamb': FusedLAMB, 'adamw': AdamW
             }[args.optimizer]
    optim_g = proto(gen.parameters(), **kw)
    optim_d = proto(itertools.chain(msd.parameters(), mpd.parameters()), **kw)

    scaler_g = amp.GradScaler(enabled=args.amp)
    scaler_d = amp.GradScaler(enabled=args.amp)

    # setup EMA
    if args.ema_decay > 0:
        # burried import, requires apex
        from common.ema_utils import (apply_multi_tensor_ema,
                                      init_multi_tensor_ema)

        gen_ema = models.get_model('HiFi-GAN', gen_config, 'cuda').cuda()
        mpd_ema = MultiPeriodDiscriminator(
            periods=args.mpd_periods,
            concat_fwd=args.concat_fwd).cuda()
        msd_ema = MultiScaleDiscriminator(
            concat_fwd=args.concat_fwd,
            no_amp_grouped_conv=args.no_amp_grouped_conv).cuda()
    else:
        gen_ema, mpd_ema, msd_ema = None, None, None

    # setup DDP
    if distributed_run:
        kw = {'device_ids': [args.local_rank],
              'output_device': args.local_rank}
        gen = DDP(gen, **kw)
        msd = DDP(msd, **kw)
        # DDP needs nonempty model
        mpd = DDP(mpd, **kw) if len(args.mpd_periods) else mpd

    # resume from last / load a checkpoint
    train_state = {}
    checkpointer = Checkpointer(args.output, args.keep_milestones)
    checkpointer.maybe_load(
        gen, mpd, msd, optim_g, optim_d, scaler_g, scaler_d, train_state, args,
        gen_ema=None, mpd_ema=None, msd_ema=None)
    iters_all = train_state.get('iters_all', 0)
    last_epoch = train_state['epoch'] + 1 if 'epoch' in train_state else -1

    sched_g = ExponentialLR(optim_g, gamma=args.lr_decay, last_epoch=last_epoch)
    sched_d = ExponentialLR(optim_d, gamma=args.lr_decay, last_epoch=last_epoch)

    if args.fine_tuning:
        print_once('Doing fine-tuning')

    train_loader = get_data_loader(args, distributed_run, train=True)
    val_loader = get_data_loader(args, distributed_run, train=False,
                                 val_kwargs=dict(repeat=5, split=True))
    val_samples_loader = get_data_loader(args, False, train=False,
                                         val_kwargs=dict(split=False),
                                         batch_size=1)
    if args.ema_decay > 0.0:
        gen_ema_params = init_multi_tensor_ema(gen, gen_ema)
        mpd_ema_params = init_multi_tensor_ema(mpd, mpd_ema)
        msd_ema_params = init_multi_tensor_ema(msd, msd_ema)

    epochs_done = 0

    for epoch in range(max(1, last_epoch), args.epochs + 1):

        metrics.start_epoch(epoch)

        if distributed_run:
            train_loader.sampler.set_epoch(epoch)

        gen.train()
        mpd.train()
        msd.train()

        iter_ = 0
        iters_num = len(train_loader) // args.grad_accumulation

        for step, batch in enumerate(train_loader):
            if step // args.grad_accumulation >= iters_num:
                break  # only full effective batches

            is_first_accum_step = step % args.grad_accumulation == 0
            is_last_accum_step = (step + 1) % args.grad_accumulation == 0
            assert (args.grad_accumulation > 1
                    or (is_first_accum_step and is_last_accum_step))

            if is_first_accum_step:
                iter_ += 1
                iters_all += 1
                metrics.start_iter(iter_)
                accum_batches = []
                optim_d.zero_grad(set_to_none=True)
                optim_g.zero_grad(set_to_none=True)

            x, y, _, y_mel = batch
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).unsqueeze(1)
            y_mel = y_mel.cuda(non_blocking=True)
            accum_batches.append((x, y, y_mel))

            with torch.set_grad_enabled(is_last_accum_step), \
                    autocast(enabled=args.amp):
                y_g_hat = gen(x)

            unfreeze(mpd)
            unfreeze(msd)

            with autocast(enabled=args.amp):
                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f = discriminator_loss(y_df_hat_r, y_df_hat_g)

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                loss_disc_s = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_disc_all = loss_disc_s + loss_disc_f

            metrics['loss_discrim'] = reduce_tensor(loss_disc_all, args.world_size)
            metrics['frames'] = x.size(0) * x.size(1) * args.world_size
            metrics.accumulate()

            loss_disc_all /= args.grad_accumulation
            scaler_d.scale(loss_disc_all).backward()

            if not is_last_accum_step:
                continue

            scaler_d.step(optim_d)
            scaler_d.update()

            # generator
            freeze(mpd)
            freeze(msd)

            for _i, (x, y, y_mel) in enumerate(reversed(accum_batches)):
                if _i != 0:  # first `y_g_hat` can be reused
                    with autocast(enabled=args.amp):
                        y_g_hat = gen(x)
                with autocast(enabled=args.amp and args.autocast_spectrogram):
                    y_g_hat_mel = mel_spec(y_g_hat.float().squeeze(1),
                                           fmax=args.mel_fmax_loss)

                # L1 mel-spectrogram Loss
                with autocast(enabled=args.amp):
                    loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                    loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                    loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                    loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                    loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                    loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

                metrics['loss_gen'] = reduce_tensor(loss_gen_all, args.world_size)
                metrics['loss_mel'] = reduce_tensor(loss_mel, args.world_size)
                metrics.accumulate()

                loss_gen_all /= args.grad_accumulation
                scaler_g.scale(loss_gen_all).backward()

            scaler_g.step(optim_g)
            scaler_g.update()

            metrics['lrate_gen'] = optim_g.param_groups[0]['lr']
            metrics['lrate_discrim'] = optim_d.param_groups[0]['lr']
            metrics.accumulate()

            if args.ema_decay > 0.0:
                apply_multi_tensor_ema(args.ema_decay, *gen_ema_params)
                apply_multi_tensor_ema(args.ema_decay, *mpd_ema_params)
                apply_multi_tensor_ema(args.ema_decay, *msd_ema_params)

            metrics.finish_iter()  # done accumulating
            if iters_all % args.step_logs_interval == 0:
                logger.log((epoch, iter_, iters_num), metrics, scope='train',
                           tb_iter=iters_all, flush_log=True)

        assert is_last_accum_step
        metrics.finish_epoch()
        logger.log((epoch,), metrics, scope='train_avg', flush_log=True)

        if epoch % args.validation_interval == 0:
            validate(args, gen, mel_spec, mpd, msd, val_loader, val_metrics)
            logger.log((epoch,), val_metrics, scope='val', tb_iter=iters_all,
                       flush_log=True)

        # validation samples
        if epoch % args.samples_interval == 0 and args.local_rank == 0:

            gen.eval()

            with torch.no_grad():
                for i, batch in enumerate(islice(val_samples_loader, 5)):
                    x, y, _, _ = batch

                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True).unsqueeze(1)

                    with autocast(enabled=args.amp):
                        y_g_hat = gen(x)
                    with autocast(enabled=args.amp and args.autocast_spectrogram):
                        # args.fmax instead of args.max_for_inference
                        y_hat_spec = mel_spec(y_g_hat.float().squeeze(1),
                                              fmax=args.mel_fmax)

                    logger.log_samples_tb(iters_all, i, y_g_hat, y_hat_spec,
                                          args.sampling_rate)

                    if epoch == args.samples_interval:  # ground truth
                        logger.log_samples_tb(0, i, y, x, args.sampling_rate)
            gen.train()

        train_state.update({'epoch': epoch, 'iters_all': iters_all})
        # save before making sched.step() for proper loading of LR
        checkpointer.maybe_save(
            gen, mpd, msd, optim_g, optim_d, scaler_g, scaler_d, epoch,
            train_state, args, gen_config, train_setup,
            gen_ema=gen_ema, mpd_ema=mpd_ema, msd_ema=msd_ema)
        logger.flush()

        sched_g.step()
        sched_d.step()

        epochs_done += 1
        if (args.epochs_this_job is not None
                and epochs_done == args.epochs_this_job):
            break

    # finished training
    if epochs_done > 0:
        logger.log((), metrics, scope='train_benchmark', flush_log=True)
        if epoch % args.validation_interval != 0:  # val metrics are not up-to-date
            validate(args, gen, mel_spec, mpd, msd, val_loader, val_metrics)
        logger.log((), val_metrics, scope='val', flush_log=True)
    else:
        print_once(f'Finished without training after epoch {args.epochs}.')


if __name__ == '__main__':
    main()
