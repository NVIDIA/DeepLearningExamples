#!/usr/bin/env python
import argparse
import logging
import os
import sys
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
from seq2seq.inference.inference import Translator
from seq2seq.models.gnmt import GNMT
from seq2seq.train.smoothing import LabelSmoothing


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
    dataset.add_argument('--max-size', default=None, type=int,
                         help='use at most MAX_SIZE elements from training \
                         dataset (useful for benchmarking), by default \
                         uses entire dataset')

    # results
    results = parser.add_argument_group('results setup')
    results.add_argument('--results-dir', default='results',
                         help='path to directory with results, it will be \
                         automatically created if it does not exist')
    results.add_argument('--save', default='gnmt',
                         help='defines subdirectory within RESULTS_DIR for \
                         results from this training run')
    results.add_argument('--print-freq', default=10, type=int,
                         help='print log every PRINT_FREQ batches')

    # model
    model = parser.add_argument_group('model setup')
    model.add_argument('--hidden-size', default=1024, type=int,
                       help='model hidden size')
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
    general.add_argument('--math', default='fp16', choices=['fp16', 'fp32'],
                         help='arithmetic type')
    general.add_argument('--seed', default=None, type=int,
                         help='master seed for random number generators, if \
                         "seed" is undefined then the master seed will be \
                         sampled from random.SystemRandom()')

    exclusive_group(group=general, name='eval', default=True,
                    help='run validation and test after every epoch')
    exclusive_group(group=general, name='env', default=True,
                    help='print info about execution env')
    exclusive_group(group=general, name='cuda', default=True,
                    help='enables cuda')
    exclusive_group(group=general, name='cudnn', default=True,
                    help='enables cudnn')

    # training
    training = parser.add_argument_group('training setup')
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
    training.add_argument('--max-length-train', default=50, type=int,
                          help='maximum sequence length for training \
                          (including special BOS and EOS tokens)')
    training.add_argument('--min-length-train', default=0, type=int,
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
    val.add_argument('--max-length-val', default=125, type=int,
                     help='maximum sequence length for validation \
                     (including special BOS and EOS tokens)')
    val.add_argument('--min-length-val', default=0, type=int,
                     help='minimum sequence length for validation \
                     (including special BOS and EOS tokens)')
    val.add_argument('--val-loader-workers', default=0, type=int,
                     help='number of workers for validation data loading')

    # test
    test = parser.add_argument_group('test setup')
    test.add_argument('--test-batch-size', default=128, type=int,
                      help='batch size for test')
    test.add_argument('--max-length-test', default=150, type=int,
                      help='maximum sequence length for test \
                      (including special BOS and EOS tokens)')
    test.add_argument('--min-length-test', default=0, type=int,
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

    args.warmup_steps = literal_eval(args.warmup_steps)
    args.remain_steps = literal_eval(args.remain_steps)
    args.decay_interval = literal_eval(args.decay_interval)

    return args


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


@utils.timer('TOTAL RUNTIME', sync_gpu=False)
def main():
    """
    Launches data-parallel multi-gpu training.
    """
    args = parse_args()
    device = utils.set_device(args.cuda, args.local_rank)
    distributed = utils.init_distributed(args.cuda)
    args.rank = utils.get_rank()

    if not args.cudnn:
        torch.backends.cudnn.enabled = False

    # create directory for results
    save_path = os.path.join(args.results_dir, args.save)
    args.save_path = save_path
    os.makedirs(save_path, exist_ok=True)

    # setup logging
    log_filename = f'log_rank_{utils.get_rank()}.log'
    utils.setup_logging(os.path.join(save_path, log_filename))

    if args.env:
        utils.log_env_info()

    logging.info(f'Saving results to: {save_path}')
    logging.info(f'Run arguments: {args}')

    # automatically set train_iter_size based on train_global_batch_size,
    # world_size and per-worker train_batch_size
    if args.train_global_batch_size is not None:
        global_bs = args.train_global_batch_size
        bs = args.train_batch_size
        world_size = utils.get_world_size()
        assert global_bs % (bs * world_size) == 0
        args.train_iter_size = global_bs // (bs * world_size)
        logging.info(f'Global batch size was set in the config, '
                     f'Setting train_iter_size to {args.train_iter_size}')

    worker_seeds, shuffling_seeds = utils.setup_seeds(args.seed, args.epochs,
                                                      device)
    worker_seed = worker_seeds[args.rank]
    logging.info(f'Worker {args.rank} is using worker seed: {worker_seed}')
    torch.manual_seed(worker_seed)

    # build tokenizer
    pad_vocab = utils.pad_vocabulary(args.math)
    tokenizer = Tokenizer(os.path.join(args.dataset_dir, config.VOCAB_FNAME),
                          pad_vocab)

    # build datasets
    train_data = LazyParallelDataset(
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
        sort=True)

    vocab_size = tokenizer.vocab_size

    # build GNMT model
    model_config = {'hidden_size': args.hidden_size,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout, 'batch_first': False,
                    'share_embedding': args.share_embedding}
    model = GNMT(vocab_size=vocab_size, **model_config)
    logging.info(model)

    batch_first = model.batch_first

    # define loss function (criterion) and optimizer
    criterion = build_criterion(vocab_size, config.PAD, args.smoothing)

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
    total_train_iters = len(train_loader) // args.train_iter_size * args.epochs
    save_info = {'model_config': model_config, 'config': args, 'tokenizer':
                 tokenizer.get_state()}
    trainer_options = dict(
        criterion=criterion,
        grad_clip=args.grad_clip,
        iter_size=args.train_iter_size,
        save_path=save_path,
        save_freq=args.save_freq,
        save_info=save_info,
        opt_config=opt_config,
        scheduler_config=scheduler_config,
        train_iterations=total_train_iters,
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
    break_training = False
    test_bleu = None
    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f'Starting epoch {epoch}')

        train_loader.sampler.set_epoch(epoch)

        trainer.epoch = epoch
        train_loss, train_perf = trainer.optimize(train_loader)

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
            test_bleu, break_training = translator.run(calc_bleu=True,
                                                       epoch=epoch)

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
    passed = utils.benchmark(test_bleu, args.target_bleu,
                             train_perf, args.target_perf)
    return passed


if __name__ == '__main__':
    passed = main()
    if not passed:
        sys.exit(1)
