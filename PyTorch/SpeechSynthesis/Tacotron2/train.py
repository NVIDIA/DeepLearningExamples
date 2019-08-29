# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import os
import time
import argparse
import numpy as np
from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from apex.parallel import DistributedDataParallel as DDP

import models
import loss_functions
import data_functions

from dllogger.logger import LOGGER
import dllogger.logger as dllg
from dllogger import tags
from dllogger.autologging import log_hardware, log_args
from scipy.io.wavfile import write as write_wav

from apex import amp
amp.lists.functional_overrides.FP32_FUNCS.remove('softmax')
amp.lists.functional_overrides.FP16_FUNCS.append('softmax')


def parse_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_directory', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    parser.add_argument('-m', '--model-name', type=str, default='', required=True,
                        help='Model to train')
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('--phrase-path', type=str, default=None,
                        help='Path to phrase sequence file used for sample generation')
    parser.add_argument('--waveglow-checkpoint', type=str, default=None,
                        help='Path to pre-trained WaveGlow checkpoint for sample generation')
    parser.add_argument('--tacotron2-checkpoint', type=str, default=None,
                        help='Path to pre-trained Tacotron2 checkpoint for sample generation')
    parser.add_argument('--anneal-steps', nargs='*',
                        help='Epochs after which decrease learning rate')
    parser.add_argument('--anneal-factor', type=float, choices=[0.1, 0.3], default=0.1,
                        help='Factor for annealing learning rate')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, required=True,
                          help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=50,
                          help='Number of epochs per checkpoint')
    training.add_argument('--seed', type=int, default=1234,
                          help='Seed for PyTorch random number generators')
    training.add_argument('--dynamic-loss-scaling', type=bool, default=True,
                          help='Enable dynamic loss scaling')
    training.add_argument('--amp-run', action='store_true',
                          help='Enable AMP')
    training.add_argument('--cudnn-enabled', action='store_true',
                          help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', action='store_true',
                          help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',
                          help='disable uniform initialization of batchnorm layer weight')

    optimization = parser.add_argument_group('optimization setup')
    optimization.add_argument(
        '--use-saved-learning-rate', default=False, type=bool)
    optimization.add_argument('-lr', '--learning-rate', type=float, required=True,
                              help='Learing rate')
    optimization.add_argument('--weight-decay', default=1e-6, type=float,
                              help='Weight decay')
    optimization.add_argument('--grad-clip-thresh', default=1.0, type=float,
                              help='Clip threshold for gradients')
    optimization.add_argument('-bs', '--batch-size', type=int, required=True,
                              help='Batch size per GPU')
    optimization.add_argument('--grad-clip', default=5.0, type=float,
                              help='Enables gradient clipping and sets maximum gradient norm value')

    # dataset parameters
    dataset = parser.add_argument_group('dataset parameters')
    dataset.add_argument('--load-mel-from-disk', action='store_true',
                         help='Loads mel spectrograms from disk instead of computing them on the fly')
    dataset.add_argument('--training-files',
                         default='filelists/ljs_audio_text_train_filelist.txt',
                         type=str, help='Path to training filelist')
    dataset.add_argument('--validation-files',
                         default='filelists/ljs_audio_text_val_filelist.txt',
                         type=str, help='Path to validation filelist')
    dataset.add_argument('--text-cleaners', nargs='*',
                         default=['english_cleaners'], type=str,
                         help='Type of text cleaners for input text')

    # audio parameters
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

    distributed = parser.add_argument_group('distributed setup')
    # distributed.add_argument('--distributed-run', default=True, type=bool,
    #                          help='enable distributed run')
    distributed.add_argument('--rank', default=0, type=int,
                             help='Rank of the process, do not set! Done by multiproc module')
    distributed.add_argument('--world-size', default=1, type=int,
                             help='Number of processes, do not set! Done by multiproc module')
    distributed.add_argument('--dist-url', type=str, default='tcp://localhost:23456',
                             help='Url used to set up distributed training')
    distributed.add_argument('--group-name', type=str, default='group_name',
                             required=False, help='Distributed group name')
    distributed.add_argument('--dist-backend', default='nccl', type=str, choices={'nccl'},
                             help='Distributed run backend')

    return parser


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= num_gpus
    return rt


def init_distributed(args, world_size, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=world_size, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def save_checkpoint(model, epoch, config, filepath):
    print("Saving model and optimizer state at epoch {} to {}".format(
        epoch, filepath))
    torch.save({'epoch': epoch,
                'config': config,
                'state_dict': model.state_dict()}, filepath)


def save_sample(model_name, model, waveglow_path, tacotron2_path, phrase_path, filepath, sampling_rate):
    if phrase_path is None:
        return
    phrase = torch.load(phrase_path, map_location='cpu')
    if model_name == 'Tacotron2':
        if waveglow_path is None:
            raise Exception(
                "WaveGlow checkpoint path is missing, could not generate sample")
        with torch.no_grad():
            checkpoint = torch.load(waveglow_path, map_location='cpu')
            waveglow = models.get_model(
                'WaveGlow', checkpoint['config'], to_cuda=False)
            waveglow.eval()
            model.eval()
            mel = model.infer(phrase.cuda())[0].cpu()
            model.train()
            audio = waveglow.infer(mel, sigma=0.6)
    elif model_name == 'WaveGlow':
        if tacotron2_path is None:
            raise Exception(
                "Tacotron2 checkpoint path is missing, could not generate sample")
        with torch.no_grad():
            checkpoint = torch.load(tacotron2_path, map_location='cpu')
            tacotron2 = models.get_model(
                'Tacotron2', checkpoint['config'], to_cuda=False)
            tacotron2.eval()
            mel = tacotron2.infer(phrase)[0].cuda()
            model.eval()
            audio = model.infer(mel, sigma=0.6).cpu()
            model.train()
    else:
        raise NotImplementedError(
            "unknown model requested: {}".format(model_name))
    audio = audio[0].numpy()
    audio = audio.astype('int16')
    write_wav(filepath, sampling_rate, audio)

# adapted from: https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
# Following snippet is licensed under MIT license


@contextmanager
def evaluating(model):
    '''Temporarily switch to evaluation mode.'''
    istrain = model.training
    try:
        model.eval()
        yield model
    finally:
        if istrain:
            model.train()


def validate(model, criterion, valset, iteration, batch_size, world_size,
             collate_fn, distributed_run, rank, batch_to_gpu):
    """Handles all the validation scoring and printing"""
    with evaluating(model), torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, num_workers=1, shuffle=False,
                                sampler=val_sampler,
                                batch_size=batch_size, pin_memory=False,
                                collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y, len_x = batch_to_gpu(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, world_size).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    LOGGER.log(key="val_iter_loss", value=reduced_val_loss)


def adjust_learning_rate(epoch, optimizer, learning_rate,
                         anneal_steps, anneal_factor):

    p = 0
    if anneal_steps is not None:
        for i, a_step in enumerate(anneal_steps):
            if epoch >= int(a_step):
                p = p+1

    if anneal_factor == 0.3:
        lr = learning_rate*((0.1 ** (p//2))*(1.0 if p % 2 == 0 else 0.3))
    else:
        lr = learning_rate*(anneal_factor ** p)

    if optimizer.param_groups[0]['lr'] != lr:
        LOGGER.log_event("learning_rate changed",
                         value=str(optimizer.param_groups[0]['lr']) + " -> " + str(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Training')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    LOGGER.set_model_name("Tacotron2_PyT")
    LOGGER.set_backends([
        dllg.StdOutBackend(log_file=None,
                           logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1),
        dllg.JsonBackend(log_file=args.log_file if args.rank == 0 else None,
                         logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1)
    ])

    LOGGER.timed_block_start("run")
    LOGGER.register_metric(tags.TRAIN_ITERATION_LOSS,
                           metric_scope=dllg.TRAIN_ITER_SCOPE)
    LOGGER.register_metric("iter_time",
                           metric_scope=dllg.TRAIN_ITER_SCOPE)
    LOGGER.register_metric("epoch_time",
                           metric_scope=dllg.EPOCH_SCOPE)
    LOGGER.register_metric("run_time",
                           metric_scope=dllg.RUN_SCOPE)
    LOGGER.register_metric("val_iter_loss",
                           metric_scope=dllg.EPOCH_SCOPE)
    LOGGER.register_metric("train_epoch_items/sec",
                           metric_scope=dllg.EPOCH_SCOPE)
    LOGGER.register_metric("train_epoch_avg_items/sec",
                           metric_scope=dllg.EPOCH_SCOPE)
    LOGGER.register_metric("train_epoch_avg_loss",
                           metric_scope=dllg.EPOCH_SCOPE)

    log_hardware()

    model_name = args.model_name
    parser = models.parse_model_args(model_name, parser)
    parser.parse_args()

    args = parser.parse_args()

    log_args(args)

    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    distributed_run = args.world_size > 1
    if distributed_run:
        init_distributed(args, args.world_size, args.rank, args.group_name)

    LOGGER.log(key=tags.RUN_START)
    run_start_time = time.time()

    model_config = models.get_model_config(model_name, args)
    model = models.get_model(model_name, model_config,
                             to_cuda=True,
                             uniform_initialize_bn_weight=not args.disable_uniform_initialize_bn_weight)

    if not args.amp_run and distributed_run:
        model = DDP(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    if args.amp_run:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        if distributed_run:
            model = DDP(model)

    try:
        sigma = args.sigma
    except AttributeError:
        sigma = None

    criterion = loss_functions.get_loss_function(model_name, sigma)

    try:
        n_frames_per_step = args.n_frames_per_step
    except AttributeError:
        n_frames_per_step = None

    collate_fn = data_functions.get_collate_function(
        model_name, n_frames_per_step)
    trainset = data_functions.get_data_loader(
        model_name, args.dataset_path, args.training_files, args)
    train_sampler = DistributedSampler(trainset) if distributed_run else None
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=train_sampler,
                              batch_size=args.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)

    valset = data_functions.get_data_loader(
        model_name, args.dataset_path, args.validation_files, args)

    batch_to_gpu = data_functions.get_batch_to_gpu(model_name)

    iteration = 0
    model.train()

    LOGGER.log(key=tags.TRAIN_LOOP)

    for epoch in range(args.epochs):
        LOGGER.epoch_start()
        epoch_start_time = time.time()
        LOGGER.log(key=tags.TRAIN_EPOCH_START, value=epoch)

        # used to calculate avg items/sec over epoch
        reduced_num_items_epoch = 0

        # used to calculate avg loss over epoch
        train_epoch_avg_loss = 0.0
        train_epoch_avg_items_per_sec = 0.0
        num_iters = 0

        # if overflow at the last iteration then do not save checkpoint
        overflow = False

        for i, batch in enumerate(train_loader):
            print("Batch: {}/{} epoch {}".format(i, len(train_loader), epoch))
            LOGGER.iteration_start()
            iter_start_time = time.time()
            LOGGER.log(key=tags.TRAIN_ITER_START, value=i)

            start = time.perf_counter()
            adjust_learning_rate(epoch, optimizer, args.learning_rate,
                                 args.anneal_steps, args.anneal_factor)

            model.zero_grad()
            x, y, num_items = batch_to_gpu(batch)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            if distributed_run:
                reduced_loss = reduce_tensor(loss.data, args.world_size).item()
                reduced_num_items = reduce_tensor(num_items.data, 1).item()
            else:
                reduced_loss = loss.item()
                reduced_num_items = num_items.item()
            if np.isnan(reduced_loss):
                raise Exception("loss is NaN")

            LOGGER.log(key=tags.TRAIN_ITERATION_LOSS, value=reduced_loss)

            train_epoch_avg_loss += reduced_loss
            num_iters += 1

            # accumulate number of items processed in this epoch
            reduced_num_items_epoch += reduced_num_items

            if args.amp_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.grad_clip_thresh)
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_thresh)

            optimizer.step()

            iteration += 1

            LOGGER.log(key=tags.TRAIN_ITER_STOP, value=i)

            iter_stop_time = time.time()
            iter_time = iter_stop_time - iter_start_time
            items_per_sec = reduced_num_items/iter_time
            train_epoch_avg_items_per_sec += items_per_sec

            LOGGER.log(key="train_iter_items/sec",
                       value=items_per_sec)
            LOGGER.log(key="iter_time", value=iter_time)
            LOGGER.iteration_stop()

        LOGGER.log(key=tags.TRAIN_EPOCH_STOP, value=epoch)
        epoch_stop_time = time.time()
        epoch_time = epoch_stop_time - epoch_start_time

        LOGGER.log(key="train_epoch_items/sec",
                   value=(reduced_num_items_epoch/epoch_time))
        LOGGER.log(key="train_epoch_avg_items/sec",
                   value=(train_epoch_avg_items_per_sec/num_iters if num_iters > 0 else 0.0))
        LOGGER.log(key="train_epoch_avg_loss", value=(
            train_epoch_avg_loss/num_iters if num_iters > 0 else 0.0))
        LOGGER.log(key="epoch_time", value=epoch_time)

        LOGGER.log(key=tags.EVAL_START, value=epoch)

        validate(model, criterion, valset, iteration,
                 args.batch_size, args.world_size, collate_fn,
                 distributed_run, args.rank, batch_to_gpu)

        LOGGER.log(key=tags.EVAL_STOP, value=epoch)

        if (epoch % args.epochs_per_checkpoint == 0) and args.rank == 0:
            checkpoint_path = os.path.join(
                args.output_directory, "checkpoint_{}_{}".format(model_name, epoch))
            save_checkpoint(model, epoch, model_config, checkpoint_path)
            save_sample(model_name, model, args.waveglow_checkpoint,
                        args.tacotron2_checkpoint, args.phrase_path,
                        os.path.join(args.output_directory, "sample_{}_{}.wav".format(model_name, iteration)), args.sampling_rate)

        LOGGER.epoch_stop()

    run_stop_time = time.time()
    run_time = run_stop_time - run_start_time
    LOGGER.log(key="run_time", value=run_time)
    LOGGER.log(key=tags.RUN_FINAL)

    print("training time", run_stop_time - run_start_time)

    LOGGER.timed_block_stop("run")

    if args.rank == 0:
        LOGGER.finish()


if __name__ == '__main__':
    main()
