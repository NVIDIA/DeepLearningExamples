# coding=utf-8
# Copyright (c) 2019-2021 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

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

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import csv
import os
import time
import argparse
import random
import logging
import h5py
from tqdm import tqdm, trange
from typing import Final, Any, Callable
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import math

import modeling
from schedulers import PolyWarmUpScheduler
from lamb_amp_opt.fused_lamb import FusedLAMBAMP

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from utils import is_main_process, format_step, get_world_size, get_rank
from torch.nn.parallel import DistributedDataParallel as DDP
from schedulers import LinearWarmUpScheduler

import dllogger

import lddl.torch


# Enabling the TorchScript Runtime Backend NVFuser
torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_bailout_depth(20)

# Track whether a SIGTERM (cluster time up) has been handled
timeout_sent = False

import signal
# handle SIGTERM sent from the scheduler and mark so we
# can gracefully save & exit
def signal_handler(sig, frame):
    global timeout_sent
    timeout_sent = True

signal.signal(signal.SIGTERM, signal_handler)


class BertPretrainingCriterion(torch.nn.Module):

    sequence_output_is_dense: Final[bool]

    def __init__(self, vocab_size, sequence_output_is_dense=False):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size
        self.sequence_output_is_dense = sequence_output_is_dense

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        if self.sequence_output_is_dense:
            # prediction_scores are already dense
            masked_lm_labels_flat = masked_lm_labels.view(-1)
            mlm_labels = masked_lm_labels_flat[masked_lm_labels_flat != -1]
            masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss


class SyncFreeStats :
    def __init__(self) :
        self.host_stats = {}
        self.device_stats = {}
        self.device_funcs = {}

    def add_stat(self, name, dtype=torch.int32, device_tensor=None, device_func=None) :
        if device_tensor is not None :
            assert dtype == device_tensor.dtype, "Error: dtype do not match: {} {}".format(dtype, device_tensor.dtype)
        self.host_stats[name] = torch.zeros(1, dtype=dtype).pin_memory()
        self.device_stats[name] = device_tensor
        self.device_funcs[name] = device_func

    def copy_from_device(self) :
        for name in self.host_stats.keys() :
            # Apply device function to device stat
            if self.device_stats[name] is not None and self.device_funcs[name] is not None:
                self.host_stats[name].copy_(self.device_funcs[name](self.device_stats[name]), non_blocking=True)
            elif self.device_stats[name] is not None :
                self.host_stats[name].copy_(self.device_stats[name], non_blocking=True)
            elif self.device_funcs[name] is not None :
                self.host_stats[name].copy_(self.device_funcs[name](), non_blocking=True)

    def host_stat(self, name) :
        assert name in self.host_stats
        return self.host_stats[name]

    def host_stat_value(self, name) :
        assert name in self.host_stats
        return self.host_stats[name].item()

    def update_host_stat(self, name, tensor) :
        self.host_stats[name] = tensor

    def device_stat(self, name) :
        assert self.device_stats[name] is not None
        return self.device_stats[name]

    def update_device_stat(self, name, tensor) :
        self.device_stats[name] = tensor


def parse_arguments():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .parquet files for the task.")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument('--vocab_file',
                        type=str,
                        default=None,
                        required=True,
                        help="Vocabulary mapping/file BERT was pretrainined on")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument('--amp',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss.')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=100,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Whether to train with seq len 512")
    parser.add_argument('--resume_phase2',
                        default=False,
                        action='store_true',
                        help="Whether to resume training with seq len 512")
    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="Whether to do allreduces during gradient accumulation steps.")
    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to do fp16 allreduce post accumulation.")
    parser.add_argument('--phase1_end_step',
                        type=int,
                        default=7038,
                        help="Number of training steps in Phase1 - seq len 128")
    parser.add_argument('--init_loss_scale',
                        type=int,
                        default=2**20,
                        help="Initial loss scaler value")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--json-summary', type=str, default="results/dllogger.json",
                        help='If provided, the json summary will be written to'
                             'the specified file.')
    parser.add_argument("--use_env",
                        action='store_true',
                        help="Whether to read local rank from ENVVAR")
    parser.add_argument('--disable_progress_bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')
    parser.add_argument('--steps_this_run', type=int, default=-1,
                        help='If provided, only run this many steps before exiting')
    parser.add_argument("--profile",
                        default=False,
                        action='store_true',
                        help="Whether to profile model.")
    parser.add_argument("--profile-start",
                        default=0,
                        type=int,
                        help="Delay profiling to start step.")
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='number of DataLoader worker processes per rank')
    # optimizations controlled by command line arguments
    parser.add_argument("--no_dense_sequence_output",
                        default=False,
                        action='store_true',
                        help="Disable dense sequence output")
    parser.add_argument("--disable_jit_fusions",
                        default=False,
                        action='store_true',
                        help="Disable jit fusions.")
    parser.add_argument("--cuda_graphs",
                        default=False,
                        action='store_true',
                        help="Enable Cuda Graphs.")

    args = parser.parse_args()
    args.fp16 = args.fp16 or args.amp

    if args.steps_this_run < 0:
        args.steps_this_run = args.max_steps

    return args

def setup_training(args):

    assert (torch.cuda.is_available())

    if args.local_rank == -1:
        device = torch.device("cuda", 0)
        args.n_gpu = 1 # torch.cuda.device_count()
        args.allreduce_post_accumulation = False
        args.allreduce_post_accumulation_fp16 = False
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        if args.cuda_graphs :
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.n_gpu = 1

    if is_main_process():
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.json_summary),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    else:
        dllogger.init(backends=[])

    dllogger.metadata("e2e_train_time", {"unit": "s"})
    dllogger.metadata("training_sequences_per_second", {"unit": "sequences/s"})
    dllogger.metadata("final_loss", {"unit": None})
    dllogger.metadata("raw_train_time", {"unit": "s"})

    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train:
        raise ValueError(" `do_train`  must be True.")

    if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (
            os.listdir(args.output_dir) and any([i.startswith('ckpt') for i in os.listdir(args.output_dir)])):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if (not args.resume_from_checkpoint or not os.path.exists(args.output_dir)) and is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args

def prepare_model_and_optimizer(args, device, sequence_output_is_dense):

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = modeling.BertForPreTraining(config, sequence_output_is_dense=sequence_output_is_dense)

    checkpoint = None
    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if args.resume_step == -1 and not args.init_checkpoint:
            model_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
            args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])

        global_step = args.resume_step if not args.init_checkpoint else 0

        if not args.init_checkpoint:
            checkpoint = torch.load(os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step)), map_location=device)
        else:
            checkpoint = torch.load(args.init_checkpoint, map_location=device)

        model.load_state_dict(checkpoint['model'], strict=False)

        if args.phase2 and not args.init_checkpoint:
            global_step -= args.phase1_end_step
        if args.init_checkpoint:
            args.resume_step = 0
        if is_main_process():
            print("resume step from ", args.resume_step)

    model.to(device)

    # If allreduce_post_accumulation_fp16 is not set, Native AMP Autocast is
    # used along with FP32 gradient accumulation and all-reduce
    if args.fp16 and args.allreduce_post_accumulation_fp16:
        model.half()

    if not args.disable_jit_fusions :
        model = torch.jit.script(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
                             lr=args.learning_rate)
    lr_scheduler = PolyWarmUpScheduler(optimizer,
                                       warmup=args.warmup_proportion,
                                       total_steps=args.max_steps,
                                       base_lr=args.learning_rate,
                                       device=device)
    grad_scaler = torch.cuda.amp.GradScaler(init_scale=args.init_loss_scale, enabled=args.fp16)

    model.checkpoint_activations(args.checkpoint_activations)

    if args.resume_from_checkpoint:
        # For phase2 from scratch, need to reset the learning rate and step count in the checkpoint. Else restore values in checkpoint.
        if (args.phase2 and not args.resume_phase2) or args.init_checkpoint :
            for group in checkpoint['optimizer']['param_groups'] :
                group['step'].zero_()
                group['lr'].fill_(args.learning_rate)
        else :
            if 'grad_scaler' in checkpoint and (not args.phase2 or args.resume_phase2):
                grad_scaler.load_state_dict(checkpoint['grad_scaler'])
        optimizer.load_state_dict(checkpoint['optimizer'])  # , strict=False)

    if args.local_rank != -1:
        # Cuda Graphs requires that DDP is captured on a side stream
        # It is important to synchronize the streams after the DDP initialization
        # so anything after sees properly initialized model weights across GPUs
        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream) :
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, bucket_cap_mb=torch.cuda.get_device_properties(device).total_memory, gradient_as_bucket_view=True)
        torch.cuda.current_stream().wait_stream(side_stream)

        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
        def scale_by_grad_accum_steps_wrapper(hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:

            def scale_by_grad_accum_steps_wrapper_hook(
                hook_state, bucket: dist.GradBucket
            ) -> torch.futures.Future[torch.Tensor]:
                bucket.set_buffer(bucket.buffer().div_(args.gradient_accumulation_steps))
                fut = hook(hook_state, bucket)
                return fut

            return scale_by_grad_accum_steps_wrapper_hook

        # With gradient accumulation, the DDP comm hook divides the gradients by the number
        # gradient accumulation steps
        if args.gradient_accumulation_steps > 1:
            model.register_comm_hook(None, scale_by_grad_accum_steps_wrapper(allreduce_hook))

    optimizer.setup_fp32_params()

    criterion = BertPretrainingCriterion(config.vocab_size, sequence_output_is_dense=sequence_output_is_dense)

    if (args.resume_from_checkpoint and not args.phase2) or (args.resume_phase2) or args.init_checkpoint:
        start_epoch = checkpoint.get('epoch', 0)
    else:
        start_epoch = 0

    return model, optimizer, grad_scaler, lr_scheduler, checkpoint, global_step, criterion, start_epoch


def checkpoint_step(args, epoch, global_step, model, optimizer, grad_scaler, last3_checkpoint_paths) :
    torch.cuda.synchronize()
    if is_main_process() and not args.skip_checkpoint:
        # Save a trained model
        dllogger.log(step="PARAMETER", data={"checkpoint_step": global_step})
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self
        if args.resume_step < 0 or not args.phase2:
            output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step))
        else:
            output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step + args.phase1_end_step))
        if args.do_train:
            torch.save({'model': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'grad_scaler': grad_scaler.state_dict(),
                        'epoch': epoch}, output_save_file)

            # The new checkpoint could have a name already in
            # last3_checkpoint_paths. In this case, torch.save will overwrite
            # the old file; thus, we need to take the name out of
            # last3_checkpoint_paths and append it to the last.
            if output_save_file in last3_checkpoint_paths:
                last3_checkpoint_paths.remove(output_save_file)
            last3_checkpoint_paths.append(output_save_file)
            if len(last3_checkpoint_paths) > 3:
                ckpt_to_be_removed = last3_checkpoint_paths.pop(0)
                os.remove(ckpt_to_be_removed)


def take_training_step(args, grad_scaler, model, criterion, batch, stats):
    with torch.cuda.amp.autocast(enabled=(args.fp16 and not args.allreduce_post_accumulation_fp16)) :
        prediction_scores, seq_relationship_score = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'], masked_lm_labels=batch['labels'])
        loss = criterion(prediction_scores, seq_relationship_score, batch['labels'], batch['next_sentence_labels'])

    stats.device_stat('average_loss').add_(loss.detach())
    grad_scaler.scale(loss).backward()


def take_optimizer_step(args, lr_scheduler, optimizer, grad_scaler, device, stats):
    lr_scheduler.step()  # learning rate warmup
    grad_scaler.step(optimizer)

    # Stats copying is located here prior to the infinity check being reset
    # in GradScaler::update()
    stats.copy_from_device()

    grad_scaler.update()
    optimizer.zero_grad(set_to_none=True)


def main():
    global timeout_sent

    args = parse_arguments()

    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)

    device, args = setup_training(args)
    dllogger.log(step="PARAMETER", data={"Config": [str(args)]})

    # Prepare optimizer
    model, optimizer, grad_scaler, lr_scheduler, checkpoint, global_resume_step, criterion, epoch = prepare_model_and_optimizer(args, device, sequence_output_is_dense=not args.no_dense_sequence_output)
    # Prepare the data loader.
    if is_main_process():
        tic = time.perf_counter()
    train_dataloader = lddl.torch.get_bert_pretrain_data_loader(
        args.input_dir,
        local_rank=max(args.local_rank, 0),
        vocab_file=args.vocab_file,
        data_loader_kwargs={
            'batch_size': args.train_batch_size * args.n_gpu,
            'num_workers': args.num_workers,
            'pin_memory': True,
        },
        base_seed=args.seed,
        log_dir=None if args.output_dir is None else os.path.join(args.output_dir, 'lddl_log'),
        log_level=logging.WARNING,
        start_epoch=epoch,
    )
    if is_main_process():
        print('get_bert_pretrain_data_loader took {} s!'.format(time.perf_counter() - tic))

    if is_main_process():
        dllogger.log(step="PARAMETER", data={"SEED": args.seed})
        dllogger.log(step="PARAMETER", data={"train_start": True})
        dllogger.log(step="PARAMETER", data={"batch_size_per_gpu": args.train_batch_size})
        dllogger.log(step="PARAMETER", data={"learning_rate": args.learning_rate})

    model.train()
    most_recent_ckpts_paths = []

    stats = SyncFreeStats()
    # Host Only Stats
    stats.add_stat('model_step')
    # Device/Host Sync-ed Stats
    stats.add_stat('optimizer_step', dtype=torch.int32, device_func=(lambda: optimizer.param_groups[0]['step']))
    stats.add_stat('average_loss', dtype=torch.float32, device_tensor=torch.zeros(1, dtype=torch.float32, device=device))
    stats.add_stat('learning_rate', dtype=torch.float32, device_func=(lambda: optimizer.param_groups[0]['lr']))
    if grad_scaler.is_enabled():
        # This stat only indicates a skipped step occurred.  It does not accumulate the number of skipped steps
        stats.add_stat('skip_optimizer_step', dtype=torch.float32, device_func=(lambda: grad_scaler._found_inf_per_device(optimizer)[device]))
        stats.add_stat('skipped_optimizer_steps', dtype=torch.float32, device_tensor=torch.zeros(1, dtype=torch.float32, device=device),
                                                  device_func=(lambda x: x.add_(grad_scaler._found_inf_per_device(optimizer)[device])))
    else:
        stats.add_stat('skip_optimizer_step', dtype=torch.float32)
        stats.add_stat('skipped_optimizer_steps', dtype=torch.float32)

    static_gpu_batch = None
    full_cudagraph = None
    grad_accum_cudagraph = None
    if args.cuda_graphs:
        static_gpu_batch = {
            'input_ids': torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
            'token_type_ids': torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
            'attention_mask': torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
            'labels': torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
            'next_sentence_labels': torch.ones(args.train_batch_size, dtype=torch.int64, device=device),
        }

        side_stream = torch.cuda.Stream()

        # Warmup Steps - includes jitting fusions
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(11):
                take_training_step(args, grad_scaler, model, criterion, static_gpu_batch, stats)
                take_optimizer_step(args, lr_scheduler, optimizer, grad_scaler, device, stats)
        torch.cuda.current_stream().wait_stream(side_stream)

        # Capture Graph
        full_cudagraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(full_cudagraph):
            take_training_step(args, grad_scaler, model, criterion, static_gpu_batch, stats)
            take_optimizer_step(args, lr_scheduler, optimizer, grad_scaler, device, stats)

        # Warmup Steps - includes jitting fusions
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                with model.no_sync():
                    take_training_step(args, grad_scaler, model, criterion, static_gpu_batch, stats)
        torch.cuda.current_stream().wait_stream(side_stream)

        # Capture Graph
        grad_accum_cudagraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(grad_accum_cudagraph):
            with model.no_sync():
                take_training_step(args, grad_scaler, model, criterion, static_gpu_batch, stats)

    train_iter = tqdm(
        train_dataloader,
        desc="Iteration",
        disable=args.disable_progress_bar,
        total=len(train_dataloader),
    ) if is_main_process() else train_dataloader


    raw_train_start = None

    # avoid nvfuser compilation times in measuring perf with phase2 binning
    # ideally skip > 3 * num_bins fwd+bwd iterations to start measuring perf 
    skip_fwd_bwd_for_perf = 4
    if args.phase2: #we use 8 bins with phase2
        skip_fwd_bwd_for_perf = 50 

    while True:
        for step, batch in enumerate(train_iter):
            # The first training step is 1 and not 0 when gradient accumulating
            # in order to avoid an optimizer step on the very first step
            stats.host_stat('model_step').add_(1)
            grad_accumulation_step = (stats.host_stat_value('model_step') % args.gradient_accumulation_steps) != 0

            if raw_train_start is None and step == skip_fwd_bwd_for_perf:
                raw_train_start = time.time()

            # Execute Model Step
            if args.cuda_graphs:
                for k in batch.keys():
                    static_gpu_batch[k].copy_(batch[k], non_blocking=True)
                if grad_accumulation_step:
                    grad_accum_cudagraph.replay()
                else:
                    full_cudagraph.replay()
            else:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                if args.allreduce_post_accumulation and grad_accumulation_step:
                    with model.no_sync():
                        take_training_step(args, grad_scaler, model, criterion, batch, stats)
                else:
                    take_training_step(args, grad_scaler, model, criterion, batch, stats)

                if not grad_accumulation_step:
                    take_optimizer_step(args, lr_scheduler, optimizer, grad_scaler, device, stats)

            # Log Optimizer Step
            if (not grad_accumulation_step) or timeout_sent:
                static_optimizer_step = stats.host_stat_value('model_step') // args.gradient_accumulation_steps
                dynamic_optimizer_step = static_optimizer_step - int(stats.host_stat_value('skipped_optimizer_steps')) + global_resume_step
                no_log_steps = static_optimizer_step % args.log_freq

                # Log Final Step (MAYBE)
                # Since the stats are asynchronously pushed from the GPU to CPU, they are not always reliable
                # Therefore, a synchronization is required to guarantee you see the intended value.
                # Without a synchronization, it is possible for some GPUs to go through the exit conditional
                # and others to not because they accidentally see a different value for `skipped_optimizer_steps`.
                # In order to remove most device syncs, synchronizations only begin in the last few steps
                # where the skipped step count matters.
                if static_optimizer_step + global_resume_step >= args.steps_this_run or timeout_sent:
                    torch.cuda.synchronize()
                    dynamic_optimizer_step = static_optimizer_step - int(stats.host_stat_value('skipped_optimizer_steps')) + global_resume_step
                    if dynamic_optimizer_step >= args.steps_this_run or timeout_sent:
                        train_time_raw = time.time() - raw_train_start

                        last_num_steps = args.log_freq if no_log_steps == 0 else no_log_steps
                        stats.device_stat('average_loss').div_(last_num_steps * args.gradient_accumulation_steps)
                        if (torch.distributed.is_initialized()):
                            stats.device_stat('average_loss').div_(get_world_size())
                            torch.distributed.all_reduce(stats.device_stat('average_loss'))

                        # We block on this copy to insure the final value
                        stats.host_stat('average_loss').copy_(stats.device_stat('average_loss'))
                        if is_main_process():
                            dllogger.log(step=(epoch, dynamic_optimizer_step,), data={"final_loss": stats.host_stat_value('average_loss')})

                        checkpoint_step(args, epoch, dynamic_optimizer_step, model, optimizer, grad_scaler, most_recent_ckpts_paths)

                        return args, train_time_raw, stats, skip_fwd_bwd_for_perf

                if no_log_steps == 0:
                    if is_main_process():
                        dllogger.log(step=(epoch, dynamic_optimizer_step,),
                                     data={"average_loss": stats.host_stat_value('average_loss') / (args.log_freq * args.gradient_accumulation_steps),
                                           "learning_rate": stats.host_stat_value('learning_rate'),
                                           "skipped_steps": int(stats.host_stat_value('skipped_optimizer_steps'))})
                        if stats.host_stat_value('skip_optimizer_step') > 0.:
                            dllogger.log(step="PARAMETER", data={"loss_scale": grad_scaler._get_scale_async().item()})

                    stats.device_stat('average_loss').zero_()

                    if not args.skip_checkpoint and (dynamic_optimizer_step % args.num_steps_per_checkpoint == 0):
                        checkpoint_step(args, epoch, dynamic_optimizer_step, model, optimizer, grad_scaler, most_recent_ckpts_paths)

        epoch += 1


if __name__ == "__main__":

    now = time.time()
    args, train_time_raw, stats, skip_fwd_bwd_for_perf = main()
    gpu_count = args.n_gpu
    if torch.distributed.is_initialized():
        gpu_count = get_world_size()
    if is_main_process():
        e2e_time = time.time() - now
        training_perf = args.train_batch_size * gpu_count * (stats.host_stat_value('model_step') - skip_fwd_bwd_for_perf) / train_time_raw
        dllogger.log(step=tuple(), data={"e2e_train_time": e2e_time,
                                         "training_sequences_per_second": training_perf,
                                         "final_loss": stats.host_stat_value('average_loss'),
                                         "raw_train_time": train_time_raw })
    dllogger.flush()
