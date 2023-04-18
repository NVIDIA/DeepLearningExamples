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

import common.filter_warnings

import argparse
import copy
import io
import os
import sys
import random
from functools import partial
from itertools import cycle, islice
from pathlib import Path

import torch
import numpy as np
from contextlib import suppress as empty_context
from torch.nn.parallel import DistributedDataParallel

import wav2vec2.arg_parser
from common import tb_dllogger as logger
from common.dataset import adjust_max_tokens, get_batch_iterator
from common.fairseq.data import Dictionary
from common.fairseq.dist import ModuleProxyWrapper
from common.fairseq.utils import multiply_grads
from common.helpers import (Checkpointer, num_weights, to_gpu,
                            init_multi_tensor_ema, apply_multi_tensor_ema)
from common.optimizers import get_optimizer, lr_exp_policy, lr_poly_policy
from common.utils import print_once, set_torch_seed, setup_distributed
from wav2vec2.criterion import Wav2vecCriterion, CTCCriterion
from wav2vec2.logging import init_logger, W2v2Metrics, W2v2FineTuningMetrics
from wav2vec2.utils import build_model, load_dataset


@torch.no_grad()
def validate(epoch, step, valid_loader, model, ema_model, criterion,
             val_metrics, val_ema_metrics, world_size, fp16, bf16):

    val_losses = []
    val_wer = []
    for model, metrics, scope in [(model, val_metrics, 'val'),
                                  (ema_model, val_ema_metrics, 'val_ema')]:
        if model is None:
            continue

        model.eval()
        criterion.eval()
        metrics._start_accumulating(None, True, scope=scope)
        output_keys = None

        assert len(valid_loader) > 1, (
            'Validation needs at least 2 iterations to handle empty batches.')

        for batch in valid_loader:
            is_empty_batch = len(batch) == 0
            if not is_empty_batch:
                to_gpu(batch, fp16=fp16, bf16=bf16)

                loss, _, logging_output = criterion(model, batch)

                if output_keys is None:
                    output_keys = logging_output.keys()

            else:
                assert output_keys is not None, (
                    f'Invalid iters num: {len(valid_loader)}')
                logging_output = {k: 0 for k in output_keys}

            logging_output['ignore'] = int(is_empty_batch)
            metrics.log_scalars(logging_output)
            metrics.all_reduce(world_size)
            metrics.accumulate()

        metrics.finish_val(scope=scope)
        logger.log(() if epoch is None else (epoch,),  metrics, scope=scope,
                   tb_iter=step)

        val_losses.append(metrics.metrics[scope]['loss'])
        if 'wer' in metrics.metrics[scope]:
            val_wer.append(metrics.metrics[scope]['wer'])
        model.train()
        criterion.train()

    return val_losses, val_wer


def main():
    parser = argparse.ArgumentParser(
        description='wav2vec 2.0 Deep Learning Example')
    wav2vec2.arg_parser.populate(parser)
    args = parser.parse_args()

    assert not args.bf16 or args.fp32_pos_conv, (
        "bfloat16 requires casting positional convolutions to float32")

    if args.mode == 'finetune':
        wav2vec2.utils.update_args_for_finetuning(args, args.w2v_path)

    head = lambda list_: list_[0]  # fairseq compat, scalars wrapped w/ lists
    args.lr = head(args.lr)
    args.update_freq = head(args.update_freq)

    assert(torch.cuda.is_available())
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    world_size = setup_distributed(args.local_rank)
    args.world_size = world_size  # For FP16Optimizer
    print_once(f"World size: {world_size}")

    assert args.seed is not None, (
        "Random seed is used to ensure same model weights across all devices. "
        "To allow None, draw a seed and synchronize across devices")

    set_torch_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    random.seed(args.seed + args.local_rank)

    pre_training = (args.mode == 'pretrain')

    checkpointer = Checkpointer(args, 'wav2vec2')

    if not pre_training:
        assert args.labels or checkpointer.last_state, \
            "Supply output labels or resume from a checkpoint."
        if checkpointer.last_state is not None:
            f = io.StringIO(checkpointer.last_state["output_labels"])
        else:
            f = open(Path(args.data, f"dict.{args.labels}.txt"))
        target_dictionary = Dictionary.load(f)
        f.seek(0)
        checkpointer.output_labels = f.read()
        f.close()

        Metrics = W2v2FineTuningMetrics
        criterion = CTCCriterion(target_dictionary, post_process='letter')
    else:
        target_dictionary = None
        Metrics = W2v2Metrics
        criterion = Wav2vecCriterion(args)

    kw = {'benchmark_epochs': args.benchmark_epochs_num, 'cuda': not args.cpu}
    metrics = Metrics(**kw)
    val_metrics = Metrics(scopes=['val'], **kw)
    val_ema_metrics = Metrics(scopes=['val_ema'], **kw)

    init_logger(args.output_dir, args.log_file, args.ema)
    logger.log_parameters(vars(args), tb_subset='train')

    assert args.update_freq >= 1

    model, seq_gen, tokenizer = build_model(args, args.mode, target_dictionary)
    model.cuda()
    print_once(f'Model size: {num_weights(model) / 10 ** 6:.1f}M params\n')

    print_once('Setting up datasets...')
    train_dataset = load_dataset(args.train_subset, args, target_dictionary,
                                 with_labels=not pre_training, training=True)
    valid_dataset = load_dataset(args.valid_subset, args, target_dictionary,
                                 with_labels=not pre_training, training=False)

    # Future-proof for adoption of native AMP
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    lr_kw = {'initial_lr_scale': args.initial_lr_scale,
             'final_lr_scale': args.final_lr_scale,
             'warmup_steps': args.warmup_updates,
             'hold_steps': args.hold_updates,
             'num_steps': args.max_update,
             'lr': args.lr}
    if args.lr_policy == 'poly':
        adjust_lr = partial(lr_poly_policy, power=args.lr_poly_power, **lr_kw)
    elif args.lr_policy == 'exp':
        adjust_lr = partial(lr_exp_policy, decay=args.lr_exp_decay, **lr_kw)
    else:
        raise ValueError

    assert args.fp16 + args.bf16 <= 1, (
        "Select a single mechanism for mixed precision training.")

    checkpointer.maybe_load_state(model=model)

    if args.bf16:
        model.to(dtype=torch.bfloat16)

    if args.fp16:
        model.half()

    if (args.fp16 or args.bf16) and args.fp32_pos_conv:
        w2v = model.w2v_encoder.w2v_model if args.mode == 'finetune' else model
        w2v.encoder.pos_conv.to(dtype=torch.float32)

    multi_gpu = world_size > 1
    if multi_gpu:
        model = DistributedDataParallel(model, device_ids=[args.local_rank],
                                        output_device=args.local_rank,
                                        find_unused_parameters=True)
        model = ModuleProxyWrapper(model)

    args.bf16_disable_loss_scaler = False  # TODO Add support in the future
    optim = get_optimizer(model, args)
    adjust_lr(1, optim)

    if args.ema > 0.0:
        raise NotImplementedError(
            "EMA disabled, see https://github.com/pytorch/pytorch/issues/28594"
        )
    else:
        ema_model = None

    train_state = {'step': 0, 'epoch': 1, 'best_val_loss': float('inf'),
                   'best_val_wer': float('inf')}
    checkpointer.maybe_load_state(ema_model=ema_model, optimizer=optim,
                                  scaler=scaler, train_state=train_state)

    shard_id = int(os.getenv("RANK", args.local_rank))

    train_loader, sampler = get_batch_iterator(
        train_dataset,
        True,
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=(args.max_tokens, args.max_tokens),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=world_size,
        shard_id=shard_id,
        num_workers=args.num_workers,
        num_concat_batches=args.num_concat_batches)

    valid_loader, _ = get_batch_iterator(
        valid_dataset,
        False,
        max_tokens=args.max_tokens_valid,
        max_sentences=args.batch_size_valid,
        max_positions=(sys.maxsize, sys.maxsize),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=world_size,
        shard_id=shard_id,
        num_workers=args.num_workers,
        num_concat_batches=args.num_concat_batches)

    steps_per_epoch = len(train_loader) // args.update_freq

    checkpointer.maybe_load_state(train_loader=train_loader)
    checkpointer.last_state = None

    print_once(model)
    model.train()

    step, epoch = train_state['step'], train_state['epoch']
    start_step = step
    start_epoch = epoch

    while step < args.max_update:  # training loop
        set_torch_seed(args.seed + step)  # reproducibility after resuming
        metrics.start_epoch(epoch)
        sampler.set_epoch(epoch)

        optim.zero_grad()

        itr = islice(train_loader, steps_per_epoch * args.update_freq)
        for batch, accum_batches in zip(itr, cycle(range(args.update_freq))):

            if accum_batches == 0:
                step += 1
                model.set_num_updates(step)
                metrics.start_iter(accum_batches)

            to_gpu(batch, fp16=args.fp16, bf16=args.bf16)

            # use context manager to prevent redundant sync of gradients
            if (multi_gpu and accum_batches + 1 < args.update_freq):
                ctx = model.no_sync()
            else:
                ctx = empty_context()

            with ctx:
                loss, _, logging_output = criterion(model, batch)

                if args.fp16 or args.bf16:
                    optim.backward(loss)
                else:
                    scaler.scale(loss).backward()
                # at this point, loss is scaled by loss_scale
                # and averaged over different devices (because of DDP) (*)

            metrics.log_scalars(logging_output)

            if (accum_batches + 1) % args.update_freq == 0:
                metrics.all_reduce(world_size)

                # scales gradients update by world_size
                # (to restore sum of gradients - see (*))
                # divided by step_ntoks to average over tokens.
                grads_mult_factor = world_size / metrics.partials['sample_size']

                if args.optimizer == 'adam' and not (args.fp16 or args.bf16):
                    # adam and non-amp optimizer - can use 'scale' kwarg for step
                    # and defer grad multiplication
                    pass
                elif args.fp16 or args.bf16:
                    optim.multiply_grads(grads_mult_factor)
                else:
                    multiply_grads(optim, grads_mult_factor)

                try:
                    if args.fp16 or args.bf16:
                        # calculate grad norm, maybe clip
                        grad_norm = optim.clip_grad_norm(args.clip_norm)

                    if args.optimizer == 'adam' and not (args.fp16 or args.bf16):
                        scaler.step(optim, scale=1. / grads_mult_factor)
                    else:
                        scaler.step(optim)

                    scaler.update()
                    model.set_num_updates(step)

                except OverflowError as e:
                    print_once(f"Grad overflow, ignoring grad. {str(e)}")
                    grad_norm = torch.tensor(0.0).cuda()

                optim.zero_grad()

                if args.ema > 0.0:
                    apply_multi_tensor_ema(args.ema, *mt_ema_params)

                if args.fp16 or args.bf16:
                    metrics['loss_scale'] = optim.scaler.loss_scale

                metrics['lr'] = optim.param_groups[0]['lr']
                metrics.accumulate()
                metrics.finish_iter()

                if step % args.log_frequency == 0:
                    metrics.finish_logging_interval()
                    epoch_step = step % steps_per_epoch or steps_per_epoch
                    logger.log((epoch, epoch_step, steps_per_epoch),
                               metrics, scope='train', tb_iter=step)

                adjust_lr(step, optim)

            if step >= args.max_update:
                break

            # NOTE this will brake when resuming training on a different dataset
            assert step <= steps_per_epoch * epoch
            # end of iter

        metrics.finish_epoch()
        logger.log((epoch,), metrics, scope='train_avg', flush_log=True,
                   tb_iter=step)

        print_once('Validating...')
        val_losses, val_wer = validate(
            epoch, step, valid_loader, model, ema_model, criterion,
            val_metrics, val_ema_metrics, world_size, args.fp16, args.bf16)

        # save best ckpt based on non-EMA val results
        checkpointer.maybe_save(model, ema_model, optim, scaler, train_state,
                                step, epoch, val_losses, val_wer, args)

        if 0 < args.epochs_this_job <= epoch + 1 - start_epoch:
            print_once(f'Reached {args.epochs_this_job} epochs in this run.')
            break

        if step >= args.max_update:
            print_once(f'Reached {step} total updates.')
            break

        epoch += 1  # end of epoch

    # finished training
    if step > start_step:
        logger.log((), metrics, scope='train_benchmark')
        logger.log((), val_metrics, scope='val')
        logger.log((), val_ema_metrics, scope='val_ema', flush_log=True)

    print_once(f'Finished after reaching update {step}.')


if __name__ == "__main__":
    main()
