#!/usr/bin/env python

# Copyright (c) 2017 Elad Hoffer
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import logging
import os
import sys
import time
from ast import literal_eval

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed

import seq2seq.data.config as config
import seq2seq.train.trainer as trainers
import seq2seq.utils as utils
from seq2seq.data.dataset import LazyParallelDataset
from seq2seq.data.dataset import ParallelDataset
from seq2seq.data.dataset import TextDataset
from seq2seq.data.tokenizer import Tokenizer
from seq2seq.inference.translator import Translator
from seq2seq.models.gnmt import GNMT
from seq2seq.train.smoothing import LabelSmoothing
from seq2seq.train.table import TrainingTable


def parse_args():
    """
    Parse commandline arguments.
    """
    def exclusive_group(group, name, default, help):
        destname = name.replace('-', '_')
        subgroup = group.add_mutually_exclusive_group(required=False)
        subgroup.add_argument(f'--{name}', dest=f'{destname}',
                              action='store_true',
                              help=f'{help} (use \'--no-{name}\' to disable)')
        subgroup.add_argument(f'--no-{name}', dest=f'{destname}',
                              action='store_false', help=argparse.SUPPRESS)
        subgroup.set_defaults(**{destname: default})

    parser = argparse.ArgumentParser(
        description='GNMT training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataset
    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--dataset-dir', default='data/wmt16_de_en',
                         help='path to the directory with training/test data')

    dataset.add_argument('--src-lang',
                         default='en',
                         help='source language')
    dataset.add_argument('--tgt-lang',
                         default='de',
                         help='target language')

    dataset.add_argument('--vocab',
                         default='vocab.bpe.32000',
                         help='path to the vocabulary file \
                         (relative to DATASET_DIR directory)')
    dataset.add_argument('-bpe', '--bpe-codes', default='bpe.32000',
                         help='path to the file with bpe codes \
                         (relative to DATASET_DIR directory)')

    dataset.add_argument('--train-src',
                         default='train.tok.clean.bpe.32000.en',
                         help='path to the training source data file \
                         (relative to DATASET_DIR directory)')
    dataset.add_argument('--train-tgt',
                         default='train.tok.clean.bpe.32000.de',
                         help='path to the training target data file \
                         (relative to DATASET_DIR directory)')

    dataset.add_argument('--val-src',
                         default='newstest_dev.tok.clean.bpe.32000.en',
                         help='path to the validation source data file \
                         (relative to DATASET_DIR directory)')
    dataset.add_argument('--val-tgt',
                         default='newstest_dev.tok.clean.bpe.32000.de',
                         help='path to the validation target data file \
                         (relative to DATASET_DIR directory)')

    dataset.add_argument('--test-src',
                         default='newstest2014.tok.bpe.32000.en',
                         help='path to the test source data file \
                         (relative to DATASET_DIR directory)')
    dataset.add_argument('--test-tgt',
                         default='newstest2014.de',
                         help='path to the test target data file \
                         (relative to DATASET_DIR directory)')

    # results
    results = parser.add_argument_group('results setup')
    results.add_argument('--results-dir', default='results',
                         help='path to directory with results, it will be \
                         automatically created if it does not exist')
    results.add_argument('--save-dir', default='gnmt',
                         help='defines subdirectory within RESULTS_DIR for \
                         results from this training run')
    results.add_argument('--print-freq', default=10, type=int,
                         help='print log every PRINT_FREQ batches')

    # model
    model = parser.add_argument_group('model setup')
    model.add_argument('--hidden-size', default=1024, type=int,
                       help='hidden size of the model')
    model.add_argument('--num-layers', default=4, type=int,
                       help='number of RNN layers in encoder and in decoder')
    model.add_argument('--dropout', default=0.2, type=float,
                       help='dropout applied to input of RNN cells')

    exclusive_group(group=model, name='share-embedding', default=True,
                    help='use shared embeddings for encoder and decoder')

    model.add_argument('--smoothing', default=0.1, type=float,
                       help='label smoothing, if equal to zero model will use \
                       CrossEntropyLoss, if not zero model will be trained \
                       with label smoothing loss')

    # setup
    general = parser.add_argument_group('general setup')
    general.add_argument('--math', default='fp16',
                         choices=['fp16', 'fp32', 'manual_fp16'],
                         help='precision')
    general.add_argument('--seed', default=None, type=int,
                         help='master seed for random number generators, if \
                         "seed" is undefined then the master seed will be \
                         sampled from random.SystemRandom()')
    general.add_argument('--prealloc-mode', default='always', type=str,
                         choices=['off', 'once', 'always'],
                         help='controls preallocation')

    exclusive_group(group=general, name='eval', default=True,
                    help='run validation and test after every epoch')
    exclusive_group(group=general, name='env', default=True,
                    help='print info about execution env')
    exclusive_group(group=general, name='cuda', default=True,
                    help='enables cuda')
    exclusive_group(group=general, name='cudnn', default=True,
                    help='enables cudnn')
    exclusive_group(group=general, name='log-all-ranks', default=True,
                    help='enables logging from all distributed ranks, if \
                    disabled then only logs from rank 0 are reported')

    # training
    training = parser.add_argument_group('training setup')
    dataset.add_argument('--train-max-size', default=None, type=int,
                         help='use at most TRAIN_MAX_SIZE elements from \
                         training dataset (useful for benchmarking), by \
                         default uses entire dataset')
    training.add_argument('--train-batch-size', default=128, type=int,
                          help='training batch size per worker')
    training.add_argument('--train-global-batch-size', default=None, type=int,
                          help='global training batch size, this argument \
                          does not have to be defined, if it is defined it \
                          will be used to automatically \
                          compute train_iter_size \
                          using the equation: train_iter_size = \
                          train_global_batch_size // (train_batch_size * \
                          world_size)')
    training.add_argument('--train-iter-size', metavar='N', default=1,
                          type=int,
                          help='training iter size, training loop will \
                          accumulate gradients over N iterations and execute \
                          optimizer every N steps')
    training.add_argument('--epochs', default=6, type=int,
                          help='max number of training epochs')

    training.add_argument('--grad-clip', default=5.0, type=float,
                          help='enables gradient clipping and sets maximum \
                          norm of gradients')
    training.add_argument('--train-max-length', default=50, type=int,
                          help='maximum sequence length for training \
                          (including special BOS and EOS tokens)')
    training.add_argument('--train-min-length', default=0, type=int,
                          help='minimum sequence length for training \
                          (including special BOS and EOS tokens)')
    training.add_argument('--train-loader-workers', default=2, type=int,
                          help='number of workers for training data loading')
    training.add_argument('--batching', default='bucketing', type=str,
                          choices=['random', 'sharding', 'bucketing'],
                          help='select batching algorithm')
    training.add_argument('--shard-size', default=80, type=int,
                          help='shard size for "sharding" batching algorithm, \
                          in multiples of global batch size')
    training.add_argument('--num-buckets', default=5, type=int,
                          help='number of buckets for "bucketing" batching \
                          algorithm')

    # optimizer
    optimizer = parser.add_argument_group('optimizer setup')
    optimizer.add_argument('--optimizer', type=str, default='Adam',
                           help='training optimizer')
    optimizer.add_argument('--lr', type=float, default=2.00e-3,
                           help='learning rate')
    optimizer.add_argument('--optimizer-extra', type=str,
                           default="{}",
                           help='extra options for the optimizer')

    # mixed precision loss scaling
    loss_scaling = parser.add_argument_group(
        'mixed precision loss scaling setup'
        )
    loss_scaling.add_argument('--init-scale', type=float, default=8192,
                              help='initial loss scale')
    loss_scaling.add_argument('--upscale-interval', type=float, default=128,
                              help='loss upscaling interval')

    # scheduler
    scheduler = parser.add_argument_group('learning rate scheduler setup')
    scheduler.add_argument('--warmup-steps', type=str, default='200',
                           help='number of learning rate warmup iterations')
    scheduler.add_argument('--remain-steps', type=str, default='0.666',
                           help='starting iteration for learning rate decay')
    scheduler.add_argument('--decay-interval', type=str, default='None',
                           help='interval between learning rate decay steps')
    scheduler.add_argument('--decay-steps', type=int, default=4,
                           help='max number of learning rate decay steps')
    scheduler.add_argument('--decay-factor', type=float, default=0.5,
                           help='learning rate decay factor')

    # validation
    val = parser.add_argument_group('validation setup')
    val.add_argument('--val-batch-size', default=64, type=int,
                     help='batch size for validation')
    val.add_argument('--val-max-length', default=125, type=int,
                     help='maximum sequence length for validation \
                     (including special BOS and EOS tokens)')
    val.add_argument('--val-min-length', default=0, type=int,
                     help='minimum sequence length for validation \
                     (including special BOS and EOS tokens)')
    val.add_argument('--val-loader-workers', default=0, type=int,
                     help='number of workers for validation data loading')

    # test
    test = parser.add_argument_group('test setup')
    test.add_argument('--test-batch-size', default=128, type=int,
                      help='batch size for test')
    test.add_argument('--test-max-length', default=150, type=int,
                      help='maximum sequence length for test \
                      (including special BOS and EOS tokens)')
    test.add_argument('--test-min-length', default=0, type=int,
                      help='minimum sequence length for test \
                      (including special BOS and EOS tokens)')
    test.add_argument('--beam-size', default=5, type=int,
                      help='beam size')
    test.add_argument('--len-norm-factor', default=0.6, type=float,
                      help='length normalization factor')
    test.add_argument('--cov-penalty-factor', default=0.1, type=float,
                      help='coverage penalty factor')
    test.add_argument('--len-norm-const', default=5.0, type=float,
                      help='length normalization constant')
    test.add_argument('--intra-epoch-eval', metavar='N', default=0, type=int,
                      help='evaluate within training epoch, this option will \
                      enable extra N equally spaced evaluations executed \
                      during each training epoch')
    test.add_argument('--test-loader-workers', default=0, type=int,
                      help='number of workers for test data loading')

    # checkpointing
    chkpt = parser.add_argument_group('checkpointing setup')
    chkpt.add_argument('--start-epoch', default=0, type=int,
                       help='manually set initial epoch counter')
    chkpt.add_argument('--resume', default=None, type=str, metavar='PATH',
                       help='resumes training from checkpoint from PATH')
    chkpt.add_argument('--save-all', action='store_true', default=False,
                       help='saves checkpoint after every epoch')
    chkpt.add_argument('--save-freq', default=5000, type=int,
                       help='save checkpoint every SAVE_FREQ batches')
    chkpt.add_argument('--keep-checkpoints', default=0, type=int,
                       help='keep only last KEEP_CHECKPOINTS checkpoints, \
                       affects only checkpoints controlled by --save-freq \
                       option')

    # benchmarking
    benchmark = parser.add_argument_group('benchmark setup')
    benchmark.add_argument('--target-perf', default=None, type=float,
                           help='target training performance (in tokens \
                           per second)')
    benchmark.add_argument('--target-bleu', default=None, type=float,
                           help='target accuracy')

    # distributed
    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
                             help='global rank of the process, do not set!')
    distributed.add_argument('--local_rank', default=0, type=int,
                             help='local rank of the process, do not set!')

    args = parser.parse_args()

    args.lang = {'src': args.src_lang, 'tgt': args.tgt_lang}

    args.save_dir = os.path.join(args.results_dir, args.save_dir)
    args.vocab = os.path.join(args.dataset_dir, args.vocab)
    args.bpe_codes = os.path.join(args.dataset_dir, args.bpe_codes)
    args.train_src = os.path.join(args.dataset_dir, args.train_src)
    args.train_tgt = os.path.join(args.dataset_dir, args.train_tgt)
    args.val_src = os.path.join(args.dataset_dir, args.val_src)
    args.val_tgt = os.path.join(args.dataset_dir, args.val_tgt)
    args.test_src = os.path.join(args.dataset_dir, args.test_src)
    args.test_tgt = os.path.join(args.dataset_dir, args.test_tgt)

    args.warmup_steps = literal_eval(args.warmup_steps)
    args.remain_steps = literal_eval(args.remain_steps)
    args.decay_interval = literal_eval(args.decay_interval)

    return args


def set_iter_size(train_iter_size, train_global_batch_size, train_batch_size):
    """
    Automatically set train_iter_size based on train_global_batch_size,
    world_size and per-worker train_batch_size

    :param train_global_batch_size: global training batch size
    :param train_batch_size: local training batch size
    """
    if train_global_batch_size is not None:
        global_bs = train_global_batch_size
        bs = train_batch_size
        world_size = utils.get_world_size()
        assert global_bs % (bs * world_size) == 0
        train_iter_size = global_bs // (bs * world_size)
        logging.info(f'Global batch size was set, '
                     f'Setting train_iter_size to {train_iter_size}')
    return train_iter_size


def build_criterion(vocab_size, padding_idx, smoothing):
    if smoothing == 0.:
        logging.info(f'Building CrossEntropyLoss')
        criterion = nn.CrossEntropyLoss(ignore_index=padding_idx, size_average=False)
    else:
        logging.info(f'Building LabelSmoothingLoss (smoothing: {smoothing})')
        criterion = LabelSmoothing(padding_idx, smoothing)

    return criterion


def main():
    """
    Launches data-parallel multi-gpu training.
    """
    training_start = time.time()
    args = parse_args()
    device = utils.set_device(args.cuda, args.local_rank)
    utils.init_distributed(args.cuda)
    args.rank = utils.get_rank()

    if not args.cudnn:
        torch.backends.cudnn.enabled = False

    # create directory for results
    os.makedirs(args.save_dir, exist_ok=True)

    # setup logging
    log_filename = f'log_rank_{utils.get_rank()}.log'
    utils.setup_logging(args.log_all_ranks,
                        os.path.join(args.save_dir, log_filename))

    if args.env:
        utils.log_env_info()

    logging.info(f'Saving results to: {args.save_dir}')
    logging.info(f'Run arguments: {args}')

    args.train_iter_size = set_iter_size(args.train_iter_size,
                                         args.train_global_batch_size,
                                         args.train_batch_size)

    worker_seeds, shuffling_seeds = utils.setup_seeds(args.seed,
                                                      args.epochs,
                                                      device)
    worker_seed = worker_seeds[args.rank]
    logging.info(f'Worker {args.rank} is using worker seed: {worker_seed}')
    torch.manual_seed(worker_seed)

    # build tokenizer
    pad_vocab = utils.pad_vocabulary(args.math)
    tokenizer = Tokenizer(args.vocab, args.bpe_codes, args.lang, pad_vocab)

    # build datasets
    train_data = LazyParallelDataset(
        src_fname=args.train_src,
        tgt_fname=args.train_tgt,
        tokenizer=tokenizer,
        min_len=args.train_min_length,
        max_len=args.train_max_length,
        sort=False,
        max_size=args.train_max_size,
        )

    val_data = ParallelDataset(
        src_fname=args.val_src,
        tgt_fname=args.val_tgt,
        tokenizer=tokenizer,
        min_len=args.val_min_length,
        max_len=args.val_max_length,
        sort=True,
        )

    test_data = TextDataset(
        src_fname=args.test_src,
        tokenizer=tokenizer,
        min_len=args.test_min_length,
        max_len=args.test_max_length,
        sort=True,
        )

    vocab_size = tokenizer.vocab_size

    # build GNMT model
    model_config = {'hidden_size': args.hidden_size,
                    'vocab_size': vocab_size,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                    'batch_first': False,
                    'share_embedding': args.share_embedding,
                    }
    model = GNMT(**model_config).to(device)
    logging.info(model)

    batch_first = model.batch_first

    # define loss function (criterion) and optimizer
    criterion = build_criterion(vocab_size, config.PAD,
                                args.smoothing).to(device)

    opt_config = {'optimizer': args.optimizer, 'lr': args.lr}
    opt_config.update(literal_eval(args.optimizer_extra))
    logging.info(f'Training optimizer config: {opt_config}')

    scheduler_config = {'warmup_steps': args.warmup_steps,
                        'remain_steps': args.remain_steps,
                        'decay_interval': args.decay_interval,
                        'decay_steps': args.decay_steps,
                        'decay_factor': args.decay_factor}

    logging.info(f'Training LR schedule config: {scheduler_config}')

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info(f'Number of parameters: {num_parameters}')

    batching_opt = {'shard_size': args.shard_size,
                    'num_buckets': args.num_buckets}
    # get data loaders
    train_loader = train_data.get_loader(batch_size=args.train_batch_size,
                                         seeds=shuffling_seeds,
                                         batch_first=batch_first,
                                         shuffle=True,
                                         batching=args.batching,
                                         batching_opt=batching_opt,
                                         num_workers=args.train_loader_workers)

    val_loader = val_data.get_loader(batch_size=args.val_batch_size,
                                     batch_first=batch_first,
                                     shuffle=False,
                                     num_workers=args.val_loader_workers)

    test_loader = test_data.get_loader(batch_size=args.test_batch_size,
                                       batch_first=batch_first,
                                       shuffle=False,
                                       pad=True,
                                       num_workers=args.test_loader_workers)

    translator = Translator(model=model,
                            tokenizer=tokenizer,
                            loader=test_loader,
                            beam_size=args.beam_size,
                            max_seq_len=args.test_max_length,
                            len_norm_factor=args.len_norm_factor,
                            len_norm_const=args.len_norm_const,
                            cov_penalty_factor=args.cov_penalty_factor,
                            print_freq=args.print_freq,
                            reference=args.test_tgt,
                            )

    # create trainer
    total_train_iters = len(train_loader) // args.train_iter_size * args.epochs
    save_info = {
        'model_config': model_config,
        'config': args,
        'tokenizer': tokenizer.get_state()
        }
    loss_scaling = {
        'init_scale': args.init_scale,
        'upscale_interval': args.upscale_interval
        }
    trainer_options = dict(
        model=model,
        criterion=criterion,
        grad_clip=args.grad_clip,
        iter_size=args.train_iter_size,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        save_info=save_info,
        opt_config=opt_config,
        scheduler_config=scheduler_config,
        train_iterations=total_train_iters,
        keep_checkpoints=args.keep_checkpoints,
        math=args.math,
        loss_scaling=loss_scaling,
        print_freq=args.print_freq,
        intra_epoch_eval=args.intra_epoch_eval,
        translator=translator,
        prealloc_mode=args.prealloc_mode,
        )

    trainer = trainers.Seq2SeqTrainer(**trainer_options)

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(checkpoint_file, 'model_best.pth')
        if os.path.isfile(checkpoint_file):
            trainer.load(checkpoint_file)
        else:
            logging.error(f'No checkpoint found at {args.resume}')

    # training loop
    best_loss = float('inf')
    training_perf = []
    break_training = False
    test_bleu = None
    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f'Starting epoch {epoch}')

        train_loader.sampler.set_epoch(epoch)

        trainer.epoch = epoch
        train_loss, train_perf = trainer.optimize(train_loader)
        training_perf.append(train_perf)

        # evaluate on validation set
        if args.eval:
            logging.info(f'Running validation on dev set')
            val_loss, val_perf = trainer.evaluate(val_loader)

            # remember best prec@1 and save checkpoint
            if args.rank == 0:
                is_best = val_loss < best_loss
                best_loss = min(val_loss, best_loss)
                trainer.save(save_all=args.save_all, is_best=is_best)

        if args.eval:
            utils.barrier()
            eval_fname = f'eval_epoch_{epoch}'
            eval_path = os.path.join(args.save_dir, eval_fname)
            _, eval_stats = translator.run(
                calc_bleu=True,
                epoch=epoch,
                eval_path=eval_path,
                )
            test_bleu = eval_stats['bleu']
            if args.target_bleu and test_bleu >= args.target_bleu:
                logging.info(f'Target accuracy reached')
                break_training = True

        acc_log = []
        acc_log += [f'Summary: Epoch: {epoch}']
        acc_log += [f'Training Loss: {train_loss:.4f}']
        if args.eval:
            acc_log += [f'Validation Loss: {val_loss:.4f}']
            acc_log += [f'Test BLEU: {test_bleu:.2f}']

        perf_log = []
        perf_log += [f'Performance: Epoch: {epoch}']
        perf_log += [f'Training: {train_perf:.0f} Tok/s']
        if args.eval:
            perf_log += [f'Validation: {val_perf:.0f} Tok/s']

        if args.rank == 0:
            logging.info('\t'.join(acc_log))
            logging.info('\t'.join(perf_log))

        logging.info(f'Finished epoch {epoch}')
        if break_training:
            break

    utils.barrier()
    training_stop = time.time()
    training_time = training_stop - training_start
    logging.info(f'Total training time {training_time:.0f} s')

    table = TrainingTable()
    avg_training_perf = sum(training_perf) / len(training_perf)
    table.add(utils.get_world_size(), args.train_batch_size, test_bleu,
              avg_training_perf, training_time)
    if utils.get_rank() == 0:
        table.write('Training Summary', args.math)

    passed = utils.benchmark(test_bleu, args.target_bleu,
                             train_perf, args.target_perf)
    if not passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
