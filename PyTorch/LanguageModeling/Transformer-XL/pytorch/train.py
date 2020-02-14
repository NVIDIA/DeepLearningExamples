# coding: utf-8

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import functools
import itertools
import logging
import math
import os
import sys
import time

import dllogger
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from apex import amp
from torch.nn.parallel import DistributedDataParallel

import lamb
import utils
from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.data_parallel import BalancedDataParallel
from utils.exp_utils import AverageMeter
from utils.exp_utils import benchmark
from utils.exp_utils import create_exp_dir
from utils.exp_utils import log_env_info


def parse_args():
    parent_parser = argparse.ArgumentParser(
        description='PyTorch Transformer-XL Language Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
        )

    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)
    cfg_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    cfg_parser.add_argument('--config', default='default')
    cfg_parser.add_argument('--config_file', default='config.yaml')

    config_args, _ = cfg_parser.parse_known_args()

    if config_args.config is not None and config_args.config_file is not None:
        with open(config_args.config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)[config_args.config]['train']
    else:
        config = {}

    general = parser.add_argument_group('general setup')
    general.add_argument('--work_dir', default='LM-TFM', type=str,
                         help='Directory for the results')
    general.add_argument('--append_dataset', action='store_true',
                         help='Automatically append dataset name to work_dir')
    general.add_argument('--append_time', action='store_true',
                         help='Automatically append current time to work_dir')
    general.add_argument('--cuda', action='store_true',
                         help='Run training on a GPU using CUDA')
    general.add_argument('--fp16', action='store_true',
                         help='Run training in fp16/mixed precision')
    general.add_argument('--restart', type=str, default='',
                         help='Restart training from the saved checkpoint')
    general.add_argument('--debug', action='store_true',
                         help='Run in debug mode (do not create exp dir)')
    general.add_argument('--log_all_ranks', action='store_true',
                         help='Enable logging from all distributed ranks')
    general.add_argument('--save-all', action='store_true',
                         help='Save all checkpoints')
    general.add_argument('--no_env', action='store_true',
                         help='Do not print info on execution env')
    general.add_argument('--log_interval', type=int, default=10,
                         help='Report interval')
    general.add_argument('--target_throughput', type=float, default=None,
                         help='Target training throughput (for benchmarking)')
    general.add_argument('--target_perplexity', type=float, default=None,
                         help='Target validation perplexity (for benchmarking)')

    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--data', type=str, default='../data/wikitext-103',
                         help='Location of the data corpus')
    dataset.add_argument('--dataset', type=str, default='wt103',
                         choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                         help='Dataset name')
    dataset.add_argument('--vocab', type=str, default='word', choices=['word', 'bpe'],
                         help='Type of vocabulary')

    model = parser.add_argument_group('model setup')
    model.add_argument('--n_layer', type=int, default=16,
                       help='Number of total layers')
    model.add_argument('--n_head', type=int, default=8,
                       help='Number of heads')
    model.add_argument('--d_head', type=int, default=64,
                       help='Head dimension')
    model.add_argument('--d_embed', type=int, default=-1,
                       help='Embedding dimension')
    model.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    model.add_argument('--d_inner', type=int, default=2048,
                       help='Inner dimension in feedforward layer')
    model.add_argument('--dropout', type=float, default=0.1,
                       help='Global dropout rate')
    model.add_argument('--dropatt', type=float, default=0.0,
                       help='Attention probability dropout rate')
    model.add_argument('--pre_lnorm', action='store_true',
                       help='Apply LayerNorm to the input instead of the output')
    model.add_argument('--attn_type', type=int, default=0,
                       help='Attention type. 0 for ours, 1 for Shaw et al,'
                       '2 for Vaswani et al, 3 for Al Rfou et al.')
    model.add_argument('--not_tied', action='store_true',
                       help='Do not tie the word embedding and softmax weights')
    model.add_argument('--clamp_len', type=int, default=-1,
                       help='Use the same pos embeddings after clamp_len')
    model.add_argument('--adaptive', action='store_true',
                       help='Use adaptive softmax')
    model.add_argument('--div_val', type=int, default=1,
                       help='Dividend value for adaptive input and softmax')
    model.add_argument('--sample_softmax', type=int, default=-1,
                       help='Number of samples in sampled softmax')
    model.add_argument('--init', default='normal', type=str,
                       help='Parameter initializer to use')
    model.add_argument('--emb_init', default='normal', type=str,
                       help='Parameter initializer to use')
    model.add_argument('--init_range', type=float, default=0.1,
                       help='Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--emb_init_range', type=float, default=0.01,
                       help='Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--init_std', type=float, default=0.02,
                       help='Parameters initialized by N(0, init_std)')
    model.add_argument('--proj_init_std', type=float, default=0.01,
                       help='Parameters initialized by N(0, init_std)')

    opt = parser.add_argument_group('optimizer setup')
    opt.add_argument('--optim', default='jitlamb', type=str,
                     choices=['adam', 'sgd', 'adagrad', 'lamb', 'jitlamb'],
                     help='Optimizer to use')
    opt.add_argument('--lr', type=float, default=0.01,
                     help='Initial learning rate')
    opt.add_argument('--mom', type=float, default=0.0,
                     help='Momentum for sgd')
    opt.add_argument('--scheduler', default='cosine', type=str,
                     choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                     help='LR scheduler to use')
    opt.add_argument('--max_step_scheduler', type=int, default=None,
                     help='Max number of training steps for LR scheduler')
    opt.add_argument('--warmup_step', type=int, default=1000,
                     help='Number of iterations for LR warmup')
    opt.add_argument('--decay_rate', type=float, default=0.5,
                     help='Decay factor when ReduceLROnPlateau is used')
    opt.add_argument('--lr_min', type=float, default=0.0,
                     help='Minimum learning rate during annealing')
    opt.add_argument('--clip', type=float, default=0.25,
                     help='Gradient clipping')
    opt.add_argument('--weight_decay', type=float, default=0.0,
                     help='Weight decay for adam|lamb')
    opt.add_argument('--clip_nonemb', action='store_true',
                     help='Only clip the gradient of non-embedding params')
    opt.add_argument('--patience', type=int, default=0,
                     help='Patience')
    opt.add_argument('--eta_min', type=float, default=0.001,
                     help='Min learning rate for cosine scheduler')

    training = parser.add_argument_group('training setup')
    training.add_argument('--max_step', type=int, default=40000,
                          help='Max number of training steps')
    training.add_argument('--batch_size', type=int, default=256,
                          help='Global batch size')
    training.add_argument('--local_batch_size', type=int, default=None,
                          help='Local (per-device) batch size, this setting \
                          overrides global --batch_size and sets batch_size \
                          to local_batch_size * world_size')
    training.add_argument('--batch_chunk', type=int, default=1,
                          help='Split batch into chunks and train with '
                          'gradient accumulation')
    training.add_argument('--roll', action='store_true',
                          help='Enable random shifts within each data stream')
    training.add_argument('--tgt_len', type=int, default=192,
                          help='Number of tokens to predict')
    training.add_argument('--ext_len', type=int, default=0,
                          help='Length of the extended context')
    training.add_argument('--mem_len', type=int, default=192,
                          help='Length of the retained previous heads')
    training.add_argument('--seed', type=int, default=1111,
                          help='Random seed')
    training.add_argument('--multi_gpu', default=None, type=str,
                          choices=['ddp', 'dp'],
                          help='Use multiple GPU')
    training.add_argument('--gpu0_bsz', type=int, default=-1,
                          help='Batch size on gpu 0 (for "dp" backend)')
    training.add_argument('--same_length', action='store_true',
                          help='Use the same attn length for all tokens')
    training.add_argument('--varlen', action='store_true',
                          help='Use variable length')

    val = parser.add_argument_group('validation setup')
    val.add_argument('--eval_tgt_len', type=int, default=192,
                     help='Number of tokens to predict for evaluation')
    val.add_argument('--eval_batch_size', type=int, default=16,
                     help='Eval batch size')
    val.add_argument('--eval_max_steps', type=int, default=-1,
                     help='Max eval steps')
    val.add_argument('--eval_interval', type=int, default=5000,
                     help='Evaluation interval')

    dist = parser.add_argument_group('distributed setup')
    dist.add_argument('--local_rank',  type=int,
                      default=os.getenv('LOCAL_RANK', 0),
                      help='Used for multi-process training.')

    parser.set_defaults(**config)
    args, _ = parser.parse_known_args()

    args.tied = not args.not_tied

    if args.d_embed < 0:
        args.d_embed = args.d_model

    assert args.ext_len >= 0, 'extended context length must be non-negative'
    assert args.batch_size % args.batch_chunk == 0

    return args


def save_checkpoint(args, model, model_config, optimizer, scheduler, vocab,
                    train_step, best_val_loss, work_dir, name='checkpoint.pt'):
    if args.fp16:
        amp_state = amp.state_dict()
    else:
        amp_state = None

    state = {
        'args': args,
        'model_config': model_config,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'vocab': vocab,
        'amp_state': amp_state,
        'train_step': train_step,
        'best_val_loss': best_val_loss,
        }

    with utils.distributed.sync_workers() as rank:
        path = os.path.join(work_dir, name)
        logging.info(f'Saving checkpoint to {path}')
        if rank == 0:
            torch.save(state, path)


def load_checkpoint(path):
    if os.path.isdir(path):
        path = os.path.join(path, 'checkpoint_last.pt')

    dst = f'cuda:{torch.cuda.current_device()}'
    logging.info(f'Loading checkpoint from {path}')
    checkpoint = torch.load(path, map_location=dst)
    return checkpoint


def init_weight(weight, args):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m, args):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight, args)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight, args)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight, args)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
        if hasattr(m, 'out_layers_weights'):
            for i in range(len(m.out_layers_weights)):
                if m.out_layers_weights[i] is not None:
                    init_weight(m.out_layers_weights[i], args)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb, args)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias, args)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias, args)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)


def update_dropout(m, args):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout


def update_dropatt(m, args):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt


def evaluate(eval_iter, model, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model.reset_length(tgt_len=args.eval_tgt_len,
                           ext_len=args.ext_len + args.tgt_len - args.eval_tgt_len,
                           mem_len=args.mem_len
                           )
    else:
        model.reset_length(tgt_len=args.eval_tgt_len,
                           ext_len=args.ext_len,
                           mem_len=args.mem_len + args.tgt_len - args.eval_tgt_len,
                           )

    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = None
        for i, (data, target, seq_len, warm) in enumerate(eval_iter):
            if args.eval_max_steps > 0 and i >= args.eval_max_steps:
                break
            loss, mems = model(data, target, mems)
            loss = loss.float().mean()
            if warm:
                assert (not mems) or all([m.size(0) == model.mem_len for m in mems])
                total_loss += seq_len * loss.item()
                total_len += seq_len

    # Switch back to the training mode
    model.reset_length(tgt_len=args.tgt_len,
                       ext_len=args.ext_len,
                       mem_len=args.mem_len
                       )
    model.train()

    return total_loss / total_len


def train(tr_iter, va_iter, model, para_model, model_config, optimizer,
          optimizer_sparse, scheduler, scheduler_sparse, vocab, epoch, train_step,
          best_val_loss, meters, args):
    # Turn on training mode which enables dropout.
    model.train()

    train_loss = 0
    target_tokens = 0
    log_step = 0
    log_start_time = time.time()

    mems = [None for _ in range(args.batch_chunk)]
    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter

    for batch, (data, target, seq_len, _) in enumerate(train_iter):
        log_step += 1
        target_tokens += target.numel()

        model.zero_grad()

        data_chunks = torch.chunk(data, args.batch_chunk, 1)
        target_chunks = torch.chunk(target, args.batch_chunk, 1)

        for i in range(args.batch_chunk):
            data_i = data_chunks[i].contiguous()
            target_i = target_chunks[i].contiguous()
            loss, mems[i] = para_model(data_i, target_i, mems[i])
            loss = loss.float().mean().type_as(loss) / args.batch_chunk

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            train_loss += loss.float().item()

        if args.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        if optimizer_sparse:
            optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if optimizer_sparse:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step - args.warmup_step)
                    if scheduler_sparse:
                        scheduler_sparse.step(train_step - args.warmup_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)
            if scheduler_sparse:
                scheduler_sparse.step(train_step)

        if train_step % args.log_interval == 0:
            cur_loss = train_loss / log_step
            cur_loss = utils.distributed.all_reduce_item(cur_loss, op='mean')
            train_loss = 0

            elapsed = time.time() - log_start_time
            avg_elapsed = elapsed / log_step
            avg_elapsed = utils.distributed.all_reduce_item(avg_elapsed, op='max')
            log_start_time = time.time()
            log_step = 0

            lr = optimizer.param_groups[0]['lr']
            throughput = target_tokens / elapsed
            throughput = utils.distributed.all_reduce_item(throughput, op='sum')
            meters['train_throughput'].update(throughput)
            target_tokens = 0

            log_str = '| epoch {:3d} step {:>8d} | batches {:>6d} / {:d} | lr {:.3e} ' \
                '| ms/batch {:5.1f} | tok/s {:7.0f} | loss {:5.2f}'.format(
                    epoch,
                    train_step,
                    batch+1,
                    tr_iter.n_batch,
                    lr,
                    avg_elapsed * 1000,
                    throughput,
                    cur_loss,
                    )

            dllogger_data = {
                'epoch': epoch,
                'train_batch': batch+1,
                'lr': lr,
                'train_time/batch': avg_elapsed * 1000,
                'train_throughput': throughput,
                'train_loss': cur_loss,
                }

            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
                dllogger_data['train_bits_per_character'] = cur_loss / math.log(2)
            else:
                log_str += ' | ppl {:9.2f}'.format(math.exp(cur_loss))
                dllogger_data['train_perplexity'] = math.exp(cur_loss)

            logging.info(log_str)
            dllogger.log(step=train_step, data=dllogger_data)

        if train_step % args.eval_interval == 0:
            eval_start_time = time.time()
            val_loss = evaluate(va_iter, model, args)
            val_loss = utils.distributed.all_reduce_item(val_loss, op='mean')

            logging.info('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.2f}'.format(
                          train_step // args.eval_interval,
                          train_step,
                          (time.time() - eval_start_time),
                          val_loss,
                          )

            dllogger_data = {
                'valid_elapsed': (time.time() - eval_start_time),
                'valid_loss': val_loss,
                }

            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(val_loss / math.log(2))
                dllogger_data['valid_bits_per_character'] = val_loss / math.log(2)
            else:
                log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
                dllogger_data['valid_perplexity'] = math.exp(val_loss)
            logging.info(log_str)
            logging.info('-' * 100)
            dllogger.log(step=train_step, data=dllogger_data)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
                if not args.debug:
                    name = 'checkpoint_best.pt'
                    save_checkpoint(args, model, model_config, optimizer,
                                    scheduler, vocab, train_step,
                                    best_val_loss, args.work_dir, name)

            # Always save after eval if save_all is true and not debug
            if not args.debug and args.save_all:
                name = f'checkpoint_{train_step}.pt'
                save_checkpoint(args, model, model_config, optimizer,
                                scheduler, vocab, train_step, best_val_loss,
                                args.work_dir, name)

            # Save last checkpoint if not debug and not save_all
            if not args.debug and not args.save_all:
                name = 'checkpoint_last.pt'
                save_checkpoint(args, model, model_config, optimizer,
                                scheduler, vocab, train_step, best_val_loss,
                                args.work_dir, name)

            # dev-performance based learning rate annealing
            if args.scheduler == 'dev_perf':
                scheduler.step(val_loss)
                if scheduler_sparse:
                    scheduler_sparse.step(val_loss)

            # subtract eval time from timers for training
            log_start_time += time.time() - eval_start_time

        if train_step == args.max_step:
            break
    return train_step, best_val_loss


def main():
    args = parse_args()

    # Initialize device and distributed backend
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda' if args.cuda else 'cpu')
    utils.distributed.init_distributed(args.cuda)

    args.work_dir = utils.exp_utils.build_work_dir_name(args.work_dir,
                                                        args.dataset,
                                                        args.append_dataset,
                                                        args.append_time,
                                                        )

    with utils.distributed.sync_workers() as rank:
        if rank == 0:
            create_exp_dir(args.work_dir,
                           scripts_to_save=['train.py', 'mem_transformer.py'],
                           debug=args.debug)

    # Setup logging
    if args.log_all_ranks:
        log_file = f'train_log_rank_{utils.distributed.get_rank()}.log'
    else:
        log_file = f'train_log.log'
    dllog_file = f'train_log.json'
    log_file = os.path.join(args.work_dir, log_file)
    dllog_file = os.path.join(args.work_dir, dllog_file)

    if args.debug:
        log_file = os.devnull
        dllog_file = os.devnull

    utils.exp_utils.setup_logging(log_all_ranks=args.log_all_ranks,
                                  filename=log_file,
                                  )
    utils.exp_utils.setup_dllogger(enabled=True, filename=dllog_file)

    if args.local_batch_size is not None:
        world_size = utils.distributed.get_world_size()
        args.batch_size = world_size * args.local_batch_size
        logging.info(f'--local_batch_size was set, adjusting global batch size'
                     f' to {args.batch_size} (local_batch_size * world_size)')

    logging.info(args)
    dllogger.log(step='PARAMETER', data=vars(args))

    if not args.no_env:
        log_env_info()

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ###########################################################################
    # Load data
    ###########################################################################
    corpus = get_lm_corpus(args.data, args.dataset, args.vocab)
    ntokens = len(corpus.vocab)
    vocab = corpus.vocab
    args.n_token = ntokens

    if args.mem_len == 0:
        eval_mem_len = 0
    else:
        eval_mem_len = args.mem_len + args.tgt_len - args.eval_tgt_len

    tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len)
    va_iter = corpus.get_iterator('valid', args.eval_batch_size,
                                  args.eval_tgt_len, device=device,
                                  mem_len=eval_mem_len, ext_len=args.ext_len)
    te_iter = corpus.get_iterator('test', args.eval_batch_size,
                                  args.eval_tgt_len, device=device,
                                  mem_len=eval_mem_len, ext_len=args.ext_len)

    # adaptive softmax / embedding
    cutoffs, tie_projs = [], [False]
    if args.adaptive:
        assert args.dataset in ['wt103', 'lm1b']
        if args.dataset == 'wt103':
            cutoffs = [19997, 39997, 199997]
            tie_projs += [True] * len(cutoffs)
        elif args.dataset == 'lm1b':
            cutoffs = [59997, 99997, 639997]
            tie_projs += [False] * len(cutoffs)

    ###########################################################################
    # Build the model
    ###########################################################################
    model_config = {
        'n_token': ntokens,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'd_model': args.d_model,
        'd_head': args.d_head,
        'd_inner': args.d_inner,
        'dropout': args.dropout,
        'dropatt': args.dropatt,
        'dtype': None,
        'tie_weight': args.tied,
        'd_embed': args.d_embed,
        'div_val': args.div_val,
        'tie_projs': tie_projs,
        'pre_lnorm': args.pre_lnorm,
        'tgt_len': args.tgt_len,
        'ext_len': args.ext_len,
        'mem_len': args.mem_len,
        'cutoffs': cutoffs,
        'same_length': args.same_length,
        'attn_type': args.attn_type,
        'clamp_len': args.clamp_len,
        'sample_softmax': args.sample_softmax,
        }

    model = MemTransformerLM(**model_config)

    model.apply(functools.partial(weights_init, args=args))
    # ensure embedding init is not overridden by out_layer in case of weight sharing
    model.word_emb.apply(functools.partial(weights_init, args=args))

    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

    # optimizer
    if args.optim.lower() == 'sgd':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
            optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=args.mom)
            optimizer_sparse = None
    elif args.optim.lower() == 'adam':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
            optimizer = optim.Adam(dense_params, lr=args.lr,
                                   weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
            optimizer_sparse = None
    elif args.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
        optimizer_sparse = None
    elif args.optim.lower() == 'lamb':
        optimizer = lamb.Lamb(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
        optimizer_sparse = None
    elif args.optim.lower() == 'jitlamb':
        optimizer = lamb.JITLamb(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
        optimizer_sparse = None

    model = model.to(device)

    if args.fp16:
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level='O2',
            )

    if args.multi_gpu == 'ddp' and torch.distributed.is_initialized():
        para_model = DistributedDataParallel(model,
                                             device_ids=[args.local_rank],
                                             output_device=args.local_rank,
                                             broadcast_buffers=False,
                                             find_unused_parameters=True,
                                             )
    elif args.multi_gpu == 'dp':
        if args.gpu0_bsz >= 0:
            para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
                                              model, dim=1).to(device)
        else:
            para_model = nn.DataParallel(model, dim=1).to(device)
    else:
        para_model = model

    # scheduler
    if args.scheduler == 'cosine':
        if args.max_step_scheduler:
            max_step = args.max_step_scheduler
        else:
            max_step = args.max_step

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max_step - args.warmup_step, eta_min=args.eta_min)
        if args.sample_softmax > 0 and optimizer_sparse is not None:
            scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_sparse, max_step - args.warmup_step,
                eta_min=args.eta_min)
        else:
            scheduler_sparse = None
    elif args.scheduler == 'inv_sqrt':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            if step == 0 and args.warmup_step == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > args.warmup_step \
                    else step / (args.warmup_step ** 1.5)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        if args.sample_softmax > 0 and optimizer_sparse is not None:
            scheduler_sparse = optim.lr_scheduler.LambdaLR(
                optimizer_sparse,
                lr_lambda=lr_lambda
                )
        else:
            scheduler_sparse = None
    elif args.scheduler == 'dev_perf':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.decay_rate, patience=args.patience,
            min_lr=args.lr_min,
            )
        if args.sample_softmax > 0 and optimizer_sparse is not None:
            scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_sparse, factor=args.decay_rate, patience=args.patience,
                min_lr=args.lr_min,
                )
        else:
            scheduler_sparse = None
    elif args.scheduler == 'constant':
        pass

    logging.info('=' * 100)
    for k, v in args.__dict__.items():
        logging.info('    - {} : {}'.format(k, v))
    logging.info('=' * 100)
    logging.info('#params = {}'.format(args.n_all_param))
    logging.info('#non emb params = {}'.format(args.n_nonemb_param))

    train_step = 0
    best_val_loss = None

    if args.restart:
        checkpoint = load_checkpoint(args.restart)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        if args.fp16:
            amp.load_state_dict(checkpoint['amp_state'])
        train_step = checkpoint['train_step']
        best_val_loss = checkpoint['best_val_loss']

        model.apply(functools.partial(update_dropout, args=args))
        model.apply(functools.partial(update_dropatt, args=args))

    meters = {}
    warmup = args.mem_len // args.tgt_len + 2
    meters['train_throughput'] = AverageMeter(warmup=warmup)
    ###########################################################################
    # Train
    ###########################################################################
    # Loop over epochs.
    # At any point you can hit Ctrl + C to break out of training early.
    start_time = time.time()
    try:
        for epoch in itertools.count(start=1):
            if args.roll:
                tr_iter.roll()
            train_step, best_val_loss = train(
                tr_iter, va_iter, model, para_model, model_config, optimizer,
                optimizer_sparse, scheduler, scheduler_sparse, vocab, epoch,
                train_step, best_val_loss, meters, args
                )

            if train_step == args.max_step:
                logging.info('-' * 100)
                logging.info('End of training')
                break
    except KeyboardInterrupt:
        logging.info('-' * 100)
        logging.info('Exiting from training early')
    elapsed = time.time() - start_time

    ###########################################################################
    # Test
    ###########################################################################
    summary = {}
    test_path = os.path.join(args.work_dir, 'checkpoint_best.pt')
    if not args.debug and os.path.exists(test_path):
        # Load the best saved model.
        checkpoint = load_checkpoint(test_path)
        model.load_state_dict(checkpoint['model_state'])

        # Run on test data.
        test_start_time = time.time()
        test_loss = evaluate(te_iter, model, args)
        test_loss = utils.distributed.all_reduce_item(test_loss, 'mean')
        test_elapsed = time.time() - test_start_time

        logging.info('=' * 100)
        if args.dataset in ['enwik8', 'text8']:
            logging.info('| End of training | test time: {:5.2f}s | test loss {:5.2f} | test bpc {:9.5f}'.format(
                test_elapsed, test_loss, test_loss / math.log(2)))
        else:
            logging.info('| End of training | test time: {:5.2f}s | test loss {:5.2f} | test ppl {:9.3f}'.format(
                test_elapsed, test_loss, math.exp(test_loss)))
        logging.info('=' * 100)

        summary.update({
            'test_elapsed': test_elapsed,
            'test_loss': test_loss,
            })

        if args.dataset in ['enwik8', 'text8']:
            summary['test_bits_per_character'] = test_loss / math.log(2)
        else:
            summary['test_perplexity'] = math.exp(test_loss)

    logging.info(f'Training time: {(elapsed / 60):.2f} minutes')
    logging.info(f'Training throughput: {meters["train_throughput"].avg:.2f} tok/s')

    if best_val_loss:
        val_perplexity = math.exp(best_val_loss)
    else:
        val_perplexity = None

    summary.update({
        'train_throughput': meters['train_throughput'].avg,
        'train_elapsed': elapsed / 60,
        'valid_loss': best_val_loss,
        'valid_perplexity': val_perplexity,
        })
    dllogger.log(step=tuple(), data=summary)

    passed = benchmark(
        target_perplexity=args.target_perplexity,
        test_perplexity=val_perplexity,
        target_throughput=args.target_throughput,
        test_throughput=meters['train_throughput'].avg
        )
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
