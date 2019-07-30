# Copyright (c) 2018, deepakn94, codyaustun, robieta. All rights reserved.
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
#
# -----------------------------------------------------------------------
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

import torch.jit
from apex.optimizers import FusedAdam
import logging
import os
import sys
import math
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

import utils
import dataloading
from neumf import NeuMF

from logger.logger import LOGGER, timed_block, timed_function
from logger import tags
from logger.autologging import log_hardware, log_args

from apex.parallel import DistributedDataParallel as DDP
from apex import amp

LOGGER.model = 'ncf'

def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model")
    parser.add_argument('--data', type=str,
                        help='Path to test and training data files')
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='Number of epochs for training')
    parser.add_argument('-b', '--batch_size', type=int, default=2**20,
                        help='Number of examples for each iteration')
    parser.add_argument('--valid_batch_size', type=int, default=2**20,
                        help='Number of examples in each validation chunk')
    parser.add_argument('-f', '--factors', type=int, default=64,
                        help='Number of predictive factors')
    parser.add_argument('--layers', nargs='+', type=int,
                        default=[256, 256, 128, 64],
                        help='Sizes of hidden layers for MLP')
    parser.add_argument('-n', '--negative_samples', type=int, default=4,
                        help='Number of negative examples per interaction')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.0045,
                        help='Learning rate for optimizer')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='Rank for test examples to be considered a hit')
    parser.add_argument('--seed', '-s', type=int, default=1,
                        help='Manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float, default=1.0,
                        help='Stop training early at threshold')
    parser.add_argument('--beta1', '-b1', type=float, default=0.25,
                        help='Beta1 for Adam')
    parser.add_argument('--beta2', '-b2', type=float, default=0.5,
                        help='Beta1 for Adam')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Epsilon for Adam')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability, if equal to 0 will not use dropout at all')
    parser.add_argument('--checkpoint_dir', default='/data/checkpoints/', type=str,
                        help='Path to the directory storing the checkpoint file')
    parser.add_argument('--load_checkpoint_path', default=None, type=str,
                        help='Path to the checkpoint file to be loaded before training/evaluation')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', type=str,
                        help='Passing "test" will only run a single evaluation, otherwise full training will be performed')
    parser.add_argument('--grads_accumulated', default=1, type=int,
                        help='Number of gradients to accumulate before performing an optimization step')
    parser.add_argument('--opt_level', default='O2', type=str,
                        help='Optimization level for Automatic Mixed Precision',
                        choices=['O0', 'O2'])
    parser.add_argument('--local_rank', default=0, type=int, help='Necessary for multi-GPU training')
    return parser.parse_args()


def init_distributed(local_rank=0):
    distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        '''
        Set cuda device so everything is done on the right GPU.
        THIS MUST BE DONE AS SOON AS POSSIBLE.
        '''
        torch.cuda.set_device(local_rank)

        '''Initialize distributed communication'''
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    logging_logger = logging.getLogger('mlperf_compliance')
    if local_rank > 0:
        sys.stdout = open('/dev/null', 'w')
        sys.stderr = open('/dev/null', 'w')
        logging_logger.setLevel(logging.ERROR)

    logging_nvlogger = logging.getLogger('nv_dl_logger')
    if local_rank > 0:
        sys.stdout = open('/dev/null', 'w')
        sys.stderr = open('/dev/null', 'w')
        logging_nvlogger.setLevel(logging.ERROR)
    
    return distributed, int(os.environ['WORLD_SIZE'])


def val_epoch(model, x, y, dup_mask, real_indices, K, samples_per_user, num_user,
              epoch=None, distributed=False):
    model.eval()

    with torch.no_grad():
        p = []
        for u,n in zip(x,y):
            p.append(model(u, n, sigmoid=True).detach())

        temp = torch.cat(p).view(-1,samples_per_user)
        del x, y, p

        # set duplicate results for the same item to -1 before topk
        temp[dup_mask] = -1
        out = torch.topk(temp,K)[1]
        # topk in pytorch is stable(if not sort)
        # key(item):value(prediction) pairs are ordered as original key(item) order
        # so we need the first position of real item(stored in real_indices) to check if it is in topk
        ifzero = (out == real_indices.view(-1,1))
        hits = ifzero.sum()
        ndcg = (math.log(2) / (torch.nonzero(ifzero)[:,1].view(-1).to(torch.float)+2).log_()).sum()

    LOGGER.log(key=tags.EVAL_SIZE, value={"epoch": epoch, "value": num_user * samples_per_user})
    LOGGER.log(key=tags.EVAL_HP_NUM_USERS, value=num_user)
    LOGGER.log(key=tags.EVAL_HP_NUM_NEG, value=samples_per_user - 1)

    if distributed:
        torch.distributed.all_reduce(hits, op=torch.distributed.reduce_op.SUM)
        torch.distributed.all_reduce(ndcg, op=torch.distributed.reduce_op.SUM)

    hr = hits.item() / num_user
    ndcg = ndcg.item() / num_user

    model.train()
    return hr, ndcg


def main():
    log_hardware()
    args = parse_args()
    args.distributed, args.world_size = init_distributed(args.local_rank)
    log_args(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print("Saving results to {}".format(args.checkpoint_dir))
    if not os.path.exists(args.checkpoint_dir) and args.checkpoint_dir != '':
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # The default of np.random.choice is replace=True, so does pytorch random_()
    LOGGER.log(key=tags.PREPROC_HP_SAMPLE_EVAL_REPLACEMENT, value=True)
    LOGGER.log(key=tags.INPUT_HP_SAMPLE_TRAIN_REPLACEMENT, value=True)
    LOGGER.log(key=tags.INPUT_STEP_EVAL_NEG_GEN)

    # sync workers before timing
    if args.distributed:
        torch.distributed.broadcast(torch.tensor([1], device="cuda"), 0)
    torch.cuda.synchronize()

    main_start_time = time.time()
    LOGGER.log(key=tags.RUN_START)

    train_ratings = torch.load(args.data+'/train_ratings.pt', map_location=torch.device('cuda:{}'.format(args.local_rank)))
    test_ratings = torch.load(args.data+'/test_ratings.pt', map_location=torch.device('cuda:{}'.format(args.local_rank)))
    test_negs = torch.load(args.data+'/test_negatives.pt', map_location=torch.device('cuda:{}'.format(args.local_rank)))

    valid_negative = test_negs.shape[1]
    LOGGER.log(key=tags.PREPROC_HP_NUM_EVAL, value=valid_negative)


    nb_maxs = torch.max(train_ratings, 0)[0]
    nb_users = nb_maxs[0].item() + 1
    nb_items = nb_maxs[1].item() + 1
    LOGGER.log(key=tags.INPUT_SIZE, value=len(train_ratings))

    all_test_users = test_ratings.shape[0]

    test_users, test_items, dup_mask, real_indices = dataloading.create_test_data(test_ratings, test_negs, args)

    # make pytorch memory behavior more consistent later
    torch.cuda.empty_cache()

    LOGGER.log(key=tags.INPUT_BATCH_SIZE, value=args.batch_size)
    LOGGER.log(key=tags.INPUT_ORDER)  # we shuffled later with randperm

    # Create model
    model = NeuMF(nb_users, nb_items,
                  mf_dim=args.factors,
                  mlp_layer_sizes=args.layers,
                  dropout=args.dropout)

    optimizer = FusedAdam(model.parameters(), lr=args.learning_rate,
                          betas=(args.beta1, args.beta2), eps=args.eps, eps_inside_sqrt=False)

    criterion = nn.BCEWithLogitsLoss(reduction='none') # use torch.mean() with dim later to avoid copy to host
    # Move model and loss to GPU
    model = model.cuda()
    criterion = criterion.cuda()

    if args.opt_level == "O2":
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level,
                                          keep_batchnorm_fp32=False, loss_scale='dynamic')

    if args.distributed:
        model = DDP(model)

    local_batch = args.batch_size // args.world_size
    traced_criterion = torch.jit.trace(criterion.forward,
                                       (torch.rand(local_batch,1),torch.rand(local_batch,1)))

    print(model)
    print("{} parameters".format(utils.count_parameters(model)))
    LOGGER.log(key=tags.OPT_LR, value=args.learning_rate)
    LOGGER.log(key=tags.OPT_NAME, value="Adam")
    LOGGER.log(key=tags.OPT_HP_ADAM_BETA1, value=args.beta1)
    LOGGER.log(key=tags.OPT_HP_ADAM_BETA2, value=args.beta2)
    LOGGER.log(key=tags.OPT_HP_ADAM_EPSILON, value=args.eps)
    LOGGER.log(key=tags.MODEL_HP_LOSS_FN, value=tags.VALUE_BCE)

    if args.load_checkpoint_path:
        state_dict = torch.load(args.load_checkpoint_path)
        model.load_state_dict(state_dict)

    if args.mode == 'test':
        LOGGER.log(key=tags.EVAL_START, value=0)
        start = time.time()
        hr, ndcg = val_epoch(model, test_users, test_items, dup_mask, real_indices, args.topk,
                             samples_per_user=valid_negative + 1,
                             num_user=all_test_users, distributed=args.distributed)
        print('HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f}'
              .format(K=args.topk, hit_rate=hr, ndcg=ndcg))
        val_time = time.time() - start
        eval_size = all_test_users * (valid_negative + 1)
        eval_throughput = eval_size / val_time

        LOGGER.log(key=tags.EVAL_ACCURACY, value={"epoch": 0, "value": hr})
        LOGGER.log(key=tags.EVAL_STOP, value=0)
        LOGGER.log(key='best_eval_throughput', value=eval_throughput)
        return
    
    success = False
    max_hr = 0
    train_throughputs, eval_throughputs = [], []

    LOGGER.log(key=tags.TRAIN_LOOP)
    for epoch in range(args.epochs):

        LOGGER.log(key=tags.TRAIN_EPOCH_START, value=epoch)
        LOGGER.log(key=tags.INPUT_HP_NUM_NEG, value=args.negative_samples)
        LOGGER.log(key=tags.INPUT_STEP_TRAIN_NEG_GEN)

        begin = time.time()

        epoch_users, epoch_items, epoch_label = dataloading.prepare_epoch_train_data(train_ratings, nb_items, args)
        num_batches = len(epoch_users)
        for i in range(num_batches // args.grads_accumulated):
            for j in range(args.grads_accumulated):
                batch_idx = (args.grads_accumulated * i) + j
                user = epoch_users[batch_idx]
                item = epoch_items[batch_idx]
                label = epoch_label[batch_idx].view(-1,1)

                outputs = model(user, item)
                loss = traced_criterion(outputs, label).float()
                loss = torch.mean(loss.view(-1), 0)

                if args.opt_level == "O2":
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
            optimizer.step()

            for p in model.parameters():
                p.grad = None

        del epoch_users, epoch_items, epoch_label
        train_time = time.time() - begin
        begin = time.time()

        epoch_samples = len(train_ratings) * (args.negative_samples + 1)
        train_throughput = epoch_samples / train_time
        train_throughputs.append(train_throughput)
        LOGGER.log(key='train_throughput', value=train_throughput)
        LOGGER.log(key=tags.TRAIN_EPOCH_STOP, value=epoch)
        LOGGER.log(key=tags.EVAL_START, value=epoch)

        hr, ndcg = val_epoch(model, test_users, test_items, dup_mask, real_indices, args.topk,
                             samples_per_user=valid_negative + 1,
                             num_user=all_test_users, epoch=epoch, distributed=args.distributed)

        val_time = time.time() - begin
        print('Epoch {epoch}: HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f},'
              ' train_time = {train_time:.2f}, val_time = {val_time:.2f}'
              .format(epoch=epoch, K=args.topk, hit_rate=hr,
                      ndcg=ndcg, train_time=train_time,
                      val_time=val_time))

        LOGGER.log(key=tags.EVAL_ACCURACY, value={"epoch": epoch, "value": hr})
        LOGGER.log(key=tags.EVAL_TARGET, value=args.threshold)
        LOGGER.log(key=tags.EVAL_STOP, value=epoch)

        eval_size = all_test_users * (valid_negative + 1)
        eval_throughput = eval_size / val_time
        eval_throughputs.append(eval_throughput)
        LOGGER.log(key='eval_throughput', value=eval_throughput)

        if hr > max_hr and args.local_rank == 0:
            max_hr = hr
            save_checkpoint_path = os.path.join(args.checkpoint_dir, 'model.pth')
            print("New best hr! Saving the model to: ", save_checkpoint_path)
            torch.save(model.state_dict(), save_checkpoint_path)
            best_model_timestamp = time.time()

        if args.threshold is not None:
            if hr >= args.threshold:
                print("Hit threshold of {}".format(args.threshold))
                success = True
                break

    if args.local_rank == 0:
        LOGGER.log(key='best_train_throughput', value=max(train_throughputs))
        LOGGER.log(key='best_eval_throughput', value=max(eval_throughputs))
        LOGGER.log(key='best_accuracy', value=max_hr)
        LOGGER.log(key='time_to_target', value=time.time() - main_start_time)
        LOGGER.log(key='time_to_best_model', value=best_model_timestamp - main_start_time)

        LOGGER.log(key=tags.RUN_STOP, value={"success": success})
        LOGGER.log(key=tags.RUN_FINAL)

if __name__ == '__main__':
    main()
