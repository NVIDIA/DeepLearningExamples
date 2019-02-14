#!/usr/bin/env python
import argparse
import os
import logging
from ast import literal_eval

import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
import torch.distributed as dist
import torch.optim

from seq2seq.models.gnmt import GNMT
from seq2seq.train.smoothing import LabelSmoothing
from seq2seq.data.dataset import TextDataset
from seq2seq.data.dataset import ParallelDataset
from seq2seq.data.tokenizer import Tokenizer
from seq2seq.utils import setup_logging
import seq2seq.data.config as config
import seq2seq.train.trainer as trainers
from seq2seq.inference.inference import Translator


def parse_args():
    """
    Parse commandline arguments.
    """
    parser = argparse.ArgumentParser(description='GNMT training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataset
    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--dataset-dir', default=None, required=True,
                         help='path to directory with training/validation data')
    dataset.add_argument('--max-size', default=None, type=int,
                         help='use at most MAX_SIZE elements from training \
                        dataset (useful for benchmarking), by default \
                        uses entire dataset')

    # results
    results = parser.add_argument_group('results setup')
    results.add_argument('--results-dir', default='results',
                         help='path to directory with results, it it will be \
                        automatically created if does not exist')
    results.add_argument('--save', default='gnmt_wmt16',
                         help='defines subdirectory within RESULTS_DIR for \
                        results from this training run')
    results.add_argument('--print-freq', default=10, type=int,
                         help='print log every PRINT_FREQ batches')

    # model
    model = parser.add_argument_group('model setup')
    model.add_argument('--model-config',
                       default="{'hidden_size': 1024,'num_layers': 4, \
                        'dropout': 0.2, 'share_embedding': True}",
                       help='GNMT architecture configuration')
    model.add_argument('--smoothing', default=0.1, type=float,
                       help='label smoothing, if equal to zero model will use \
                        CrossEntropyLoss, if not zero model will be trained \
                        with label smoothing loss')

    # setup
    general = parser.add_argument_group('general setup')
    general.add_argument('--math', default='fp16', choices=['fp32', 'fp16'],
                         help='arithmetic type')
    general.add_argument('--seed', default=None, type=int,
                         help='set random number generator seed')
    general.add_argument('--disable-eval', action='store_true', default=False,
                         help='disables validation after every epoch')
    general.add_argument('--workers', default=0, type=int,
                         help='number of workers for data loading')

    cuda_parser = general.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true',
                             help='enables cuda (use \'--no-cuda\' to disable)')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                             help=argparse.SUPPRESS)
    cuda_parser.set_defaults(cuda=True)

    cudnn_parser = general.add_mutually_exclusive_group(required=False)
    cudnn_parser.add_argument('--cudnn', dest='cudnn', action='store_true',
                              help='enables cudnn (use \'--no-cudnn\' to disable)')
    cudnn_parser.add_argument('--no-cudnn', dest='cudnn', action='store_false',
                              help=argparse.SUPPRESS)
    cudnn_parser.set_defaults(cudnn=True)

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--batch-size', default=128, type=int,
                          help='batch size for training')
    training.add_argument('--epochs', default=8, type=int,
                          help='number of total epochs to run')
    training.add_argument('--optimization-config',
                          default="{'optimizer': 'Adam', 'lr': 5e-4}", type=str,
                          help='optimizer config')
    training.add_argument('--grad-clip', default=5.0, type=float,
                          help='enabled gradient clipping and sets maximum \
                        gradient norm value')
    training.add_argument('--max-length-train', default=50, type=int,
                          help='maximum sequence length for training')
    training.add_argument('--min-length-train', default=0, type=int,
                          help='minimum sequence length for training')

    bucketing_parser = training.add_mutually_exclusive_group(required=False)
    bucketing_parser.add_argument('--bucketing', dest='bucketing', action='store_true',
                                  help='enables bucketing (use \'--no-bucketing\' to disable)')
    bucketing_parser.add_argument('--no-bucketing', dest='bucketing', action='store_false',
                                  help=argparse.SUPPRESS)
    bucketing_parser.set_defaults(bucketing=True)

    # validation
    validation = parser.add_argument_group('validation setup')
    validation.add_argument('--val-batch-size', default=128, type=int,
                            help='batch size for validation')
    validation.add_argument('--max-length-val', default=80, type=int,
                            help='maximum sequence length for validation')
    validation.add_argument('--min-length-val', default=0, type=int,
                            help='minimum sequence length for validation')

    # test
    test = parser.add_argument_group('test setup')
    test.add_argument('--test-batch-size', default=128, type=int,
                      help='batch size for test')
    test.add_argument('--max-length-test', default=150, type=int,
                      help='maximum sequence length for test')
    test.add_argument('--min-length-test', default=0, type=int,
                      help='minimum sequence length for test')
    test.add_argument('--beam-size', default=5, type=int,
                      help='beam size')
    test.add_argument('--len-norm-factor', default=0.6, type=float,
                      help='length normalization factor')
    test.add_argument('--cov-penalty-factor', default=0.1, type=float,
                      help='coverage penalty factor')
    test.add_argument('--len-norm-const', default=5.0, type=float,
                      help='length normalization constant')
    test.add_argument('--target-bleu', default=None, type=float,
                      help='target accuracy')
    test.add_argument('--intra-epoch-eval', default=0, type=int,
                      help='evaluate within epoch')

    # checkpointing
    checkpoint = parser.add_argument_group('checkpointing setup')
    checkpoint.add_argument('--start-epoch', default=0, type=int,
                            help='manually set initial epoch counter')
    checkpoint.add_argument('--resume', default=None, type=str, metavar='PATH',
                            help='resumes training from checkpoint from PATH')
    checkpoint.add_argument('--save-all', action='store_true', default=False,
                            help='saves checkpoint after every epoch')
    checkpoint.add_argument('--save-freq', default=5000, type=int,
                            help='save checkpoint every SAVE_FREQ batches')
    checkpoint.add_argument('--keep-checkpoints', default=0, type=int,
                            help='keep only last KEEP_CHECKPOINTS checkpoints, \
                        affects only checkpoints controlled by --save-freq \
                        option')

    # distributed support
    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
                             help='rank of the process, do not set! Done by multiproc module')
    distributed.add_argument('--world-size', default=1, type=int,
                             help='number of processes, do not set! Done by multiproc module')
    distributed.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                             help='url used to set up distributed training')

    return parser.parse_args()


def build_criterion(vocab_size, padding_idx, smoothing):
    if smoothing == 0.:
        logging.info(f'Building CrossEntropyLoss')
        loss_weight = torch.ones(vocab_size)
        loss_weight[padding_idx] = 0
        criterion = nn.CrossEntropyLoss(weight=loss_weight, size_average=False)
    else:
        logging.info(f'Building LabelSmoothingLoss (smoothing: {smoothing})')
        criterion = LabelSmoothing(padding_idx, smoothing)

    return criterion


def main():
    """
    Launches data-parallel multi-gpu training.
    """
    args = parse_args()

    if not args.cudnn:
        torch.backends.cudnn.enabled = False
    if args.seed is not None:
        torch.manual_seed(args.seed + args.rank)

    # initialize distributed backend
    distributed = args.world_size > 1
    if distributed:
        backend = 'nccl' if args.cuda else 'gloo'
        dist.init_process_group(backend=backend, rank=args.rank,
                                init_method=args.dist_url,
                                world_size=args.world_size)

    # create directory for results
    save_path = os.path.join(args.results_dir, args.save)
    args.save_path = save_path
    os.makedirs(save_path, exist_ok=True)

    # setup logging
    log_filename = f'log_gpu_{args.rank}.log'
    setup_logging(os.path.join(save_path, log_filename))

    logging.info(f'Saving results to: {save_path}')
    logging.info(f'Run arguments: {args}')

    if args.cuda:
        torch.cuda.set_device(args.rank)

    # build tokenizer
    tokenizer = Tokenizer(os.path.join(args.dataset_dir, config.VOCAB_FNAME))

    # build datasets
    train_data = ParallelDataset(
        src_fname=os.path.join(args.dataset_dir, config.SRC_TRAIN_FNAME),
        tgt_fname=os.path.join(args.dataset_dir, config.TGT_TRAIN_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_train,
        max_len=args.max_length_train,
        sort=False,
        max_size=args.max_size)

    val_data = ParallelDataset(
        src_fname=os.path.join(args.dataset_dir, config.SRC_VAL_FNAME),
        tgt_fname=os.path.join(args.dataset_dir, config.TGT_VAL_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_val,
        max_len=args.max_length_val,
        sort=True)

    test_data = TextDataset(
        src_fname=os.path.join(args.dataset_dir, config.SRC_TEST_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_test,
        max_len=args.max_length_test,
        sort=False)

    vocab_size = tokenizer.vocab_size

    # build GNMT model
    model_config = dict(vocab_size=vocab_size, math=args.math,
                        **literal_eval(args.model_config))
    model = GNMT(**model_config)
    logging.info(model)

    batch_first = model.batch_first

    # define loss function (criterion) and optimizer
    criterion = build_criterion(vocab_size, config.PAD, args.smoothing)
    opt_config = literal_eval(args.optimization_config)
    logging.info(f'Training optimizer: {opt_config}')

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info(f'Number of parameters: {num_parameters}')

    # get data loaders
    train_loader = train_data.get_loader(batch_size=args.batch_size,
                                         batch_first=batch_first,
                                         shuffle=True,
                                         bucketing=args.bucketing,
                                         num_workers=args.workers,
                                         drop_last=True)

    val_loader = val_data.get_loader(batch_size=args.val_batch_size,
                                     batch_first=batch_first,
                                     shuffle=False,
                                     num_workers=args.workers,
                                     drop_last=False)

    test_loader = test_data.get_loader(batch_size=args.test_batch_size,
                                       batch_first=batch_first,
                                       shuffle=False,
                                       num_workers=args.workers,
                                       drop_last=False)

    translator = Translator(model=model,
                            tokenizer=tokenizer,
                            loader=test_loader,
                            beam_size=args.beam_size,
                            max_seq_len=args.max_length_test,
                            len_norm_factor=args.len_norm_factor,
                            len_norm_const=args.len_norm_const,
                            cov_penalty_factor=args.cov_penalty_factor,
                            cuda=args.cuda,
                            print_freq=args.print_freq,
                            dataset_dir=args.dataset_dir,
                            target_bleu=args.target_bleu,
                            save_path=args.save_path)

    # create trainer
    trainer_options = dict(
        criterion=criterion,
        grad_clip=args.grad_clip,
        save_path=save_path,
        save_freq=args.save_freq,
        save_info={'config': args, 'tokenizer': tokenizer},
        opt_config=opt_config,
        batch_first=batch_first,
        keep_checkpoints=args.keep_checkpoints,
        math=args.math,
        print_freq=args.print_freq,
        cuda=args.cuda,
        distributed=distributed,
        intra_epoch_eval=args.intra_epoch_eval,
        translator=translator)

    trainer_options['model'] = model
    trainer = trainers.Seq2SeqTrainer(**trainer_options)

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth')
        if os.path.isfile(checkpoint_file):
            trainer.load(checkpoint_file)
        else:
            logging.error(f'No checkpoint found at {args.resume}')

    # training loop
    best_loss = float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f'Starting epoch {epoch}')

        if distributed:
            train_loader.sampler.set_epoch(epoch)

        trainer.epoch = epoch
        train_loss, train_perf = trainer.optimize(train_loader)

        # evaluate on validation set
        if args.rank == 0 and not args.disable_eval:
            logging.info(f'Running validation on dev set')
            val_loss, val_perf = trainer.evaluate(val_loader)

            # remember best prec@1 and save checkpoint
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            trainer.save(save_all=args.save_all, is_best=is_best)

        break_training = False
        if not args.disable_eval:
            test_bleu, break_training = translator.run(calc_bleu=True,
                                                       epoch=epoch)

        if args.rank == 0 and not args.disable_eval:
            logging.info(f'Summary: Epoch: {epoch}\t'
                         f'Training Loss: {train_loss:.4f}\t'
                         f'Validation Loss: {val_loss:.4f}\t'
                         f'Test BLEU: {test_bleu:.2f}')
            logging.info(f'Performance: Epoch: {epoch}\t'
                         f'Training: {train_perf:.0f} Tok/s\t'
                         f'Validation: {val_perf:.0f} Tok/s')
        else:
            logging.info(f'Summary: Epoch: {epoch}\t'
                         f'Training Loss {train_loss:.4f}')
            logging.info(f'Performance: Epoch: {epoch}\t'
                         f'Training: {train_perf:.0f} Tok/s')

        logging.info(f'Finished epoch {epoch}')
        if break_training:
            break

if __name__ == '__main__':
    main()
