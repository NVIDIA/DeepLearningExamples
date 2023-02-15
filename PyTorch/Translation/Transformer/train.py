#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
#-------------------------------------------------------------------------
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import collections
import os
import math
import time
import ctypes

from copy import deepcopy

import torch
import sacrebleu
import dllogger as DLLogger

from fairseq import data, distributed_utils, options, utils, tokenizer
from fairseq.ddp_trainer import DDPTrainer
from fairseq.meters import StopwatchMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.data import data_utils, load_dataset_splits
from fairseq.models import build_model
from fairseq.log_helper import setup_logger, reset_perf_meters

def main(args):

    print(args)
    setup_logger(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.local_rank)
    if args.distributed_world_size > 1:
        assert torch.distributed.is_initialized()
        torch.distributed.broadcast(torch.tensor([1], device="cuda"), 0)
        torch.cuda.synchronize()
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    ctypes.CDLL('libcudart.so').cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    ctypes.CDLL('libcudart.so').cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    torch.manual_seed(args.seed)

    src_dict, tgt_dict = data_utils.load_dictionaries(args)
    add_extra_items_to_checkpoint({'src_dict': src_dict, 'tgt_dict': tgt_dict})
    datasets = load_dataset_splits(args, ['train', 'valid', 'test'], src_dict, tgt_dict)

    model = build_model(args)
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))

    # Build trainer
    if torch.cuda.get_device_capability(0)[0] >= 7 and not args.amp:
        print('| NOTICE: your device may support faster training with --amp')
    trainer = DDPTrainer(args, model)
    print('| model {}, criterion {}'.format(args.arch, trainer.criterion.__class__.__name__))
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    epoch_itr = data.EpochBatchIterator(
        dataset=datasets[args.train_subset],
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=args.max_positions,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )
    # Load the latest checkpoint if one is available
    load_checkpoint(args, trainer, epoch_itr)

    # Send a dummy batch to warm the caching allocator
    dummy_batch = data_utils.get_dummy_batch(args.max_tokens, src_dict, tgt_dict)
    trainer.dummy_train_step(dummy_batch)

    # Sanity check
    if args.do_sanity_check:
        print('Performing sanity check...')
        sanity_score = score(args, trainer, datasets['test'], src_dict, tgt_dict, 'test.raw.de')
        DLLogger.log(step='SANITY_CHECK', data={'sanity_check_score': sanity_score}, verbosity=1)

    # Train until the learning rate gets too small or model reaches target score
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    tgt_bleu = args.target_bleu or math.inf
    current_bleu = 0.0
    best_bleu = -1.0
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    run_summary = {'loss': float('inf'),
                   'val_loss': float('inf'),
                   'speed': 0,
                   'accuracy': 0}

    while lr >= args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update and current_bleu < tgt_bleu:
        DLLogger.log(step=trainer.get_num_updates()+1, data={'epoch': epoch_itr.epoch}, verbosity=0)
        # train for one epoch
        train(args, trainer, epoch_itr)
        DLLogger.log(step=trainer.get_num_updates(), data={'walltime': train_meter.sum}, verbosity=1)
        DLLogger.log(step=trainer.get_num_updates(),
                     data={'avg_epoch_loss': trainer.avg_loss_meter.avg}, verbosity=1)

        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, datasets, valid_subsets)
            valid_bleu = score(args, trainer, datasets[valid_subsets[0]], src_dict, tgt_dict, 'valid.raw.de')
            DLLogger.log(step=trainer.get_num_updates(),
                         data={'val_loss': valid_losses[0], 'val_bleu': valid_bleu}, verbosity=1)

        # Eval BLEU score
        if args.online_eval or (tgt_bleu is not math.inf):
            current_bleu = score(args, trainer, datasets[args.gen_subset], src_dict, tgt_dict, 'test.raw.de')
            DLLogger.log(step=trainer.get_num_updates(), data={'test_bleu': current_bleu}, verbosity=1)
            best_bleu = max(best_bleu, current_bleu)

        run_summary['val_loss'] = min(run_summary['val_loss'], valid_losses[0])
        run_summary['accuracy'] = best_bleu if best_bleu >= 0 else valid_bleu
        run_summary['loss'] = valid_losses[0]
        run_summary['speed'] = trainer.throughput_meter.u_avg

        # Only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    train_meter.stop()
    run_summary['walltime'] = train_meter.sum
    DLLogger.log(step=(), data=run_summary, verbosity=0)
    print('| done training in {:.1f} seconds'.format(train_meter.sum))

def train(args, trainer, epoch_itr):
    """Train the model for one epoch."""

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr()

    # update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    max_update = args.max_update or math.inf
    num_batches = len(epoch_itr)
    torch.cuda.synchronize()
    begin = time.time()

    # reset meters
    DLLogger.flush()
    trainer.get_throughput_meter().reset()

    for i, sample in enumerate(itr):
        if i < num_batches - 1 and (i + 1) % update_freq > 0:
            # buffer updates according to --update-freq
            trainer.train_step(sample, update_params=False, last_step=(i == len(itr)-1))
            continue
        else:
            trainer.train_step(sample, update_params=True, last_step=(i == len(itr)-1))

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_throughput_meter().reset()
            reset_perf_meters()

        if (i+1) % args.log_interval == 0:
            DLLogger.flush()

        if trainer.get_num_updates() >= max_update:
            break

    torch.cuda.synchronize()
    print('Epoch time:', time.time() - begin)

    # Print epoch stats and reset training meters
    DLLogger.log(step=trainer.get_num_updates(),
                 data={'speed': trainer.get_throughput_meter().avg}, verbosity=0)
    DLLogger.flush()

def validate(args, trainer, datasets, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    valid_losses = []
    for subset in subsets:

        if len(subsets) > 1:
            print('Validating on \'{}\' subset'.format(subset))

        # Initialize data iterator
        itr = data.EpochBatchIterator(
            dataset=datasets[subset],
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=args.max_positions,
            required_batch_size_multiple=8,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
        ).next_epoch_itr(shuffle=False)

        # reset validation loss meters
        DLLogger.flush()

        subset_losses = []
        for sample in itr:
            loss = trainer.valid_step(sample)
            subset_losses.append(loss)
        subset_loss = sum(subset_losses)/len(subset_losses)

        DLLogger.flush()

        valid_losses.append(subset_loss)
        print(f'Validation loss on subset {subset}: {subset_loss}')

    return valid_losses

def score(args, trainer, dataset, src_dict, tgt_dict, ref_file):

    torch.cuda.synchronize()
    begin = time.time()

    src_dict = deepcopy(src_dict)  # This is necessary, generation of translations
    tgt_dict = deepcopy(tgt_dict)  # alters target dictionary messing up with the rest of training

    model = trainer.get_model()

    # Initialize data iterator
    itr = data.EpochBatchIterator(
        dataset=dataset,
        max_tokens=None,
        max_sentences=max(8, min(math.ceil(1024/args.distributed_world_size), 128)),
        max_positions=args.max_positions,
        required_batch_size_multiple=8,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    translator = SequenceGenerator(
	[model],
        tgt_dict.get_metadata(),
        maxlen=args.max_target_positions - 1,  # do not include EOS token
        beam_size=args.beam,
	stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
	len_penalty=args.lenpen, unk_penalty=args.unkpen,
	sampling=args.sampling, sampling_topk=args.sampling_topk, minlen=args.min_len,
        use_amp=args.amp,
        )
    # Generate and compute BLEU
    predictions = []
    translations = translator.generate_batched_itr(
            itr, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
            cuda=True, timer=gen_timer, prefix_size=args.prefix_size,
            )

    for sample_id, src_tokens, _, hypos in translations:
        # Process input and grount truth
        src_str = src_dict.string(src_tokens, args.remove_bpe)

        # Process top predictions
        for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
            _, hypo_str, _ = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                align_dict=None,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe
                )

            # Score only the top hypothesis
            if i == 0:
                hypo_str = tokenizer.Tokenizer.detokenize(hypo_str, 'de')
                predictions.append('{}\t{}'.format(sample_id, hypo_str))

    if args.distributed_world_size > 1:
        predictions = _all_gather_predictions(predictions)

    with open(os.path.join(args.data, ref_file), 'r') as reference:
        refs = [reference.readlines()]
    # reducing indexed predictions as strings is more memory efficient than reducing tuples
    predictions = [tuple(item.split('\t')) for item in predictions]
    predictions = [(int(item[0]), item[1]) for item in predictions]
    predictions.sort(key=lambda tup: tup[0])
    predictions = [hypo[1] + ('\n' if hypo[1][-1] != '\n' else '') for hypo in predictions]
    sacrebleu_score = sacrebleu.corpus_bleu(predictions, refs, lowercase=not args.test_cased_bleu).score

    if args.save_predictions:
        os.makedirs(os.path.join(args.save_dir, 'predictions'), exist_ok=True)
        fname = ref_file + '.pred.update_{}'.format(trainer.get_num_updates())
        save_path = os.path.join(args.save_dir, 'predictions', fname)
        with open(save_path, 'w') as f:
            f.write(''.join(predictions))

    DLLogger.log(step=trainer.get_num_updates(),
                 data={'inference tokens/s': float(args.distributed_world_size) / gen_timer.avg},
                 verbosity=0)
    DLLogger.flush()
    if gen_timer.sum != 0:
        print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
            len(predictions),
            gen_timer.n,
            gen_timer.sum,
            len(predictions) / gen_timer.sum,
            float(args.distributed_world_size)/gen_timer.avg
            ))

    torch.cuda.synchronize()
    print('| Eval completed in: {:.2f}s | {}CASED BLEU {:.2f}'.format(
        time.time()-begin,
        '' if args.test_cased_bleu else 'UN',
        sacrebleu_score
        ))

    return sacrebleu_score

def _all_gather_predictions(predictions):
    ready = False
    all_ready = False
    reduced_predictions = []
    max_size = 65000
    while not all_ready:
        lst_len = len(predictions)
        size = 2000     # some extra space for python stuff
        n = 0
        while n < lst_len:
            str_len = len(predictions[n].encode('utf8')) + 8  # per string pickle overhead
            if size + str_len >= max_size:
                break
            size += str_len
            n += 1
        chunk = predictions[:n]
        predictions = predictions[n:]
        if not predictions:
            ready = True
        chunk = (ready, chunk)
        torch.cuda.synchronize()
        gathered = distributed_utils.all_gather_list(chunk, max_size=65000)
        torch.cuda.synchronize()
        reduced_predictions += [t[1] for t in gathered]
        all_ready = all([t[0] for t in gathered])

    reduced_predictions = [item for sublist in reduced_predictions for item in sublist]

    return reduced_predictions


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if epoch_itr.epoch % args.save_interval != 0:
        return
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = end_of_epoch and not args.no_epoch_checkpoints
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'best': save_checkpoint.best,
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    extra_state.update(save_checkpoint.extra_items)

    checkpoints = [os.path.join(args.save_dir, 'checkpoints', fn)
                   for fn, cond in checkpoint_conds.items() if cond]
    if checkpoints:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)


def add_extra_items_to_checkpoint(items):
    if not hasattr(save_checkpoint, 'extra_items'):
        save_checkpoint.extra_items = {}
    save_checkpoint.extra_items.update(items)

def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, 'checkpoints', args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path)
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']


if __name__ == '__main__':
    parser = options.get_training_parser()
    ARGS = options.parse_args_and_arch(parser)

    distributed_utils.distributed_init(ARGS)

    main(ARGS)
