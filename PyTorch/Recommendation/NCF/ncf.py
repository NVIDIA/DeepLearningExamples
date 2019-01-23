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
from datetime import datetime
from collections import OrderedDict
from argparse import ArgumentParser

import torch
import torch.nn as nn

import utils
from neumf import NeuMF

from logger.logger import LOGGER, timed_block, timed_function
from logger import tags
from logger.autologging import log_hardware, log_args

from fp_optimizers import Fp16Optimizer
from apex.parallel import DistributedDataParallel as DDP


LOGGER.model = 'ncf'

def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model")
    parser.add_argument('--data', type=str,
                        help='Path to test and training data files')
    parser.add_argument('-e', '--epochs', type=int, default=40,
                        help='Number of epochs for training')
    parser.add_argument('-b', '--batch-size', type=int, default=1048576,
                        help='Number of examples for each iteration')
    parser.add_argument('--valid-batch-size', type=int, default=2**20,
                        help='Number of examples in each validation chunk')
    parser.add_argument('-f', '--factors', type=int, default=64,
                        help='Number of predictive factors')
    parser.add_argument('--layers', nargs='+', type=int,
                        default=[256, 256, 128, 64],
                        help='Sizes of hidden layers for MLP')
    parser.add_argument('-n', '--negative-samples', type=int, default=4,
                        help='Number of negative examples per interaction')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.0045,
                        help='Learning rate for optimizer')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='Rank for test examples to be considered a hit')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float, default=1.0,
                        help='Stop training early at threshold')
    parser.add_argument('--no-fp16', action='store_false', dest='fp16',
                        help='Do not use fp16')
    parser.add_argument('--valid-negative', type=int, default=100,
                        help='Number of negative samples for each positive test example')
    parser.add_argument('--beta1', '-b1', type=float, default=0.25,
                        help='Beta1 for Adam')
    parser.add_argument('--beta2', '-b2', type=float, default=0.5,
                        help='Beta1 for Adam')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Epsilon for Adam')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability, if equal to 0 will not use dropout at all')
    parser.add_argument('--loss-scale', default=8192, type=int,
                        help='Loss scale to use for mixed precision training')
    parser.add_argument('--checkpoint-dir', default='/data/checkpoints/', type=str,
                        help='Path to the directory storing the checkpoint file')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', type=str,
                        help='Passing "test" will only run a single evaluation, otherwise full training will be performed')
    parser.add_argument('--grads_accumulated', default=1, type=int,
                        help='Number of gradients to accumulate before performing an optimization step')
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


def val_epoch(model, x, y, dup_mask, real_indices, K, samples_per_user, num_user, output=None,
              epoch=None, distributed=False):

    start = datetime.now()
    log_2 = math.log(2)

    model.eval()

    with torch.no_grad():
        p = []
        for u,n in zip(x,y):
            p.append(model(u, n, sigmoid=True).detach())

        del x
        del y
        temp = torch.cat(p).view(-1,samples_per_user)
        del p
        # set duplicate results for the same item to -1 before topk
        temp[dup_mask] = -1
        out = torch.topk(temp,K)[1]
        # topk in pytorch is stable(if not sort)
        # key(item):value(predicetion) pairs are ordered as original key(item) order
        # so we need the first position of real item(stored in real_indices) to check if it is in topk
        ifzero = (out == real_indices.view(-1,1))
        hits = ifzero.sum()
        ndcg = (log_2 / (torch.nonzero(ifzero)[:,1].view(-1).to(torch.float)+2).log_()).sum()

    LOGGER.log(key=tags.EVAL_SIZE, value={"epoch": epoch, "value": num_user * samples_per_user})
    LOGGER.log(key=tags.EVAL_HP_NUM_USERS, value=num_user)
    LOGGER.log(key=tags.EVAL_HP_NUM_NEG, value=samples_per_user - 1)

    end = datetime.now()

    if distributed:
        torch.distributed.all_reduce(hits, op=torch.distributed.reduce_op.SUM)
        torch.distributed.all_reduce(ndcg, op=torch.distributed.reduce_op.SUM)

    hits = hits.item()
    ndcg = ndcg.item()

    if output is not None:
        result = OrderedDict()
        result['timestamp'] = datetime.now()
        result['duration'] = end - start
        result['epoch'] = epoch
        result['K'] = K
        result['hit_rate'] = hits/num_user
        result['NDCG'] = ndcg/num_user
        utils.save_result(result, output)

    model.train()
    return hits/num_user, ndcg/num_user


def generate_neg(users, true_mat, item_range, num_neg, sort=False):
    # assuming 1-d tensor input

    # for each user in 'users', generate 'num_neg' negative samples in [0, item_range)
    # also make sure negative sample is not in true sample set with mask
    # true_mat store a mask matrix where true_mat(user, item) = 0 for true sample
    # return (neg_user, neg_item)

    # list to append iterations of result
    neg_u = []
    neg_i = []

    neg_users = users.repeat(num_neg)
    while len(neg_users) > 0: # generate then filter loop
        neg_items = torch.empty_like(neg_users, dtype=torch.int64).random_(0, item_range)
        neg_mask = true_mat[neg_users, neg_items]
        neg_u.append(neg_users.masked_select(neg_mask))
        neg_i.append(neg_items.masked_select(neg_mask))

        neg_users = neg_users.masked_select(1-neg_mask)

    neg_users = torch.cat(neg_u)
    neg_items = torch.cat(neg_i)
    if sort == False:
        return neg_users, neg_items

    sorted_users, sort_indices = torch.sort(neg_users)
    return sorted_users, neg_items[sort_indices]


def main():
    log_hardware()

    args = parse_args()
    args.distributed, args.world_size = init_distributed(args.local_rank)
    log_args(args)

    main_start_time = time.time()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Save configuration to file
    print("Saving results to {}".format(args.checkpoint_dir))
    if not os.path.exists(args.checkpoint_dir) and args.checkpoint_dir != '':
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model.pth')

    # more like load trigger timer now
    LOGGER.log(key=tags.PREPROC_HP_NUM_EVAL, value=args.valid_negative)
    # The default of np.random.choice is replace=True, so does pytorch random_()
    LOGGER.log(key=tags.PREPROC_HP_SAMPLE_EVAL_REPLACEMENT, value=True)
    LOGGER.log(key=tags.INPUT_HP_SAMPLE_TRAIN_REPLACEMENT, value=True)
    LOGGER.log(key=tags.INPUT_STEP_EVAL_NEG_GEN)

    # sync worker before timing.
    if args.distributed:
        torch.distributed.broadcast(torch.tensor([1], device="cuda"), 0)
    torch.cuda.synchronize()

    LOGGER.log(key=tags.RUN_START)
    run_start_time = time.time()

    # load not converted data, just seperate one for test
    train_ratings = torch.load(args.data+'/train_ratings.pt', map_location=torch.device('cuda:{}'.format(args.local_rank)))
    test_ratings = torch.load(args.data+'/test_ratings.pt', map_location=torch.device('cuda:{}'.format(args.local_rank)))

    # get input data
    # get dims
    nb_maxs = torch.max(train_ratings, 0)[0]
    nb_users = nb_maxs[0].item()+1
    nb_items = nb_maxs[1].item()+1
    train_users = train_ratings[:,0]
    train_items = train_ratings[:,1]
    del nb_maxs, train_ratings
    LOGGER.log(key=tags.INPUT_SIZE, value=len(train_users))
    # produce things not change between epoch
    # mask for filtering duplicates with real sample
    # note: test data is removed before create mask, same as reference
    mat = torch.cuda.ByteTensor(nb_users, nb_items).fill_(1)
    mat[train_users, train_items] = 0
    # create label
    train_label = torch.ones_like(train_users, dtype=torch.float32)
    neg_label = torch.zeros_like(train_label, dtype=torch.float32)
    neg_label = neg_label.repeat(args.negative_samples)
    train_label = torch.cat((train_label,neg_label))
    del neg_label
    if args.fp16:
        train_label = train_label.half()

    # produce validation negative sample on GPU
    all_test_users = test_ratings.shape[0]

    test_users = test_ratings[:,0]
    test_pos = test_ratings[:,1].reshape(-1,1)
    test_negs = generate_neg(test_users, mat, nb_items, args.valid_negative, True)[1]

    # create items with real sample at last position
    test_users = test_users.reshape(-1,1).repeat(1,1+args.valid_negative)
    test_items = torch.cat((test_negs.reshape(-1,args.valid_negative), test_pos), dim=1)
    del test_ratings, test_negs

    # generate dup mask and real indice for exact same behavior on duplication compare to reference
    # here we need a sort that is stable(keep order of duplicates)
    # this is a version works on integer
    sorted_items, indices = torch.sort(test_items) # [1,1,1,2], [3,1,0,2]
    sum_item_indices = sorted_items.float()+indices.float()/len(indices[0]) #[1.75,1.25,1.0,2.5]
    indices_order = torch.sort(sum_item_indices)[1] #[2,1,0,3]
    stable_indices = torch.gather(indices, 1, indices_order) #[0,1,3,2]
    # produce -1 mask
    dup_mask = (sorted_items[:,0:-1] == sorted_items[:,1:])
    dup_mask = torch.cat((torch.zeros_like(test_pos, dtype=torch.uint8), dup_mask),dim=1)
    dup_mask = torch.gather(dup_mask,1,stable_indices.sort()[1])
    # produce real sample indices to later check in topk
    sorted_items, indices = (test_items != test_pos).sort()
    sum_item_indices = sorted_items.float()+indices.float()/len(indices[0])
    indices_order = torch.sort(sum_item_indices)[1]
    stable_indices = torch.gather(indices, 1, indices_order)
    real_indices = stable_indices[:,0]
    del sorted_items, indices, sum_item_indices, indices_order, stable_indices, test_pos

    if args.distributed:
        test_users = torch.chunk(test_users, args.world_size)[args.local_rank]
        test_items = torch.chunk(test_items, args.world_size)[args.local_rank]
        dup_mask = torch.chunk(dup_mask, args.world_size)[args.local_rank]
        real_indices = torch.chunk(real_indices, args.world_size)[args.local_rank]

    # make pytorch memory behavior more consistent later
    torch.cuda.empty_cache()

    LOGGER.log(key=tags.INPUT_BATCH_SIZE, value=args.batch_size)
    LOGGER.log(key=tags.INPUT_ORDER)  # we shuffled later with randperm

    print('Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d'
          % (time.time()-run_start_time, nb_users, nb_items, len(train_users),
             nb_users))

    # Create model
    model = NeuMF(nb_users, nb_items,
                  mf_dim=args.factors, mf_reg=0.,
                  mlp_layer_sizes=args.layers,
                  mlp_layer_regs=[0. for i in args.layers],
                  dropout=args.dropout)

    if args.fp16:
        model = model.half()

    print(model)
    print("{} parameters".format(utils.count_parameters(model)))

    # Save model text description
    with open(os.path.join(args.checkpoint_dir, 'model.txt'), 'w') as file:
        file.write(str(model))

    # Add optimizer and loss to graph
    if args.fp16:
        fp_optimizer = Fp16Optimizer(model, args.loss_scale)
        params = fp_optimizer.fp32_params
    else:
        params = model.parameters()

    optimizer = FusedAdam(params, lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps, eps_inside_sqrt=False)
    criterion = nn.BCEWithLogitsLoss(reduction='none') # use torch.mean() with dim later to avoid copy to host
    LOGGER.log(key=tags.OPT_LR, value=args.learning_rate)
    LOGGER.log(key=tags.OPT_NAME, value="Adam")
    LOGGER.log(key=tags.OPT_HP_ADAM_BETA1, value=args.beta1)
    LOGGER.log(key=tags.OPT_HP_ADAM_BETA2, value=args.beta2)
    LOGGER.log(key=tags.OPT_HP_ADAM_EPSILON, value=args.eps)
    LOGGER.log(key=tags.MODEL_HP_LOSS_FN, value=tags.VALUE_BCE)

    # Move model and loss to GPU
    model = model.cuda()
    criterion = criterion.cuda()

    if args.distributed:
        model = DDP(model)
        local_batch = args.batch_size // int(os.environ['WORLD_SIZE'])
    else:
        local_batch = args.batch_size
    traced_criterion = torch.jit.trace(criterion.forward, (torch.rand(local_batch,1),torch.rand(local_batch,1)))

    train_users_per_worker = len(train_label) / int(os.environ['WORLD_SIZE'])
    train_users_begin = int(train_users_per_worker * args.local_rank)
    train_users_end = int(train_users_per_worker * (args.local_rank + 1))

    # Create files for tracking training
    valid_results_file = os.path.join(args.checkpoint_dir, 'valid_results.csv')
    # Calculate initial Hit Ratio and NDCG
    test_x = test_users.view(-1).split(args.valid_batch_size)
    test_y = test_items.view(-1).split(args.valid_batch_size)

    if args.mode == 'test':
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
    
    begin = time.time()
    LOGGER.log(key=tags.EVAL_START, value=-1)
        
    hr, ndcg = val_epoch(model, test_x, test_y, dup_mask, real_indices, args.topk, samples_per_user=test_items.size(1),
                         num_user=all_test_users, distributed=args.distributed)
    val_time = time.time() - begin
    print('Initial HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f}, valid_time: {val_time:.4f}'
          .format(K=args.topk, hit_rate=hr, ndcg=ndcg, val_time=val_time))

    LOGGER.log(key=tags.EVAL_ACCURACY, value={"epoch": -1, "value": hr})
    LOGGER.log(key=tags.EVAL_TARGET, value=args.threshold)
    LOGGER.log(key=tags.EVAL_STOP, value=-1)
    
    if args.mode == 'test':
        return
    
    success = False
    max_hr = 0
    LOGGER.log(key=tags.TRAIN_LOOP)
    train_throughputs = []
    eval_throughputs = []

    for epoch in range(args.epochs):

        LOGGER.log(key=tags.TRAIN_EPOCH_START, value=epoch)
        LOGGER.log(key=tags.INPUT_HP_NUM_NEG, value=args.negative_samples)
        LOGGER.log(key=tags.INPUT_STEP_TRAIN_NEG_GEN)

        begin = time.time()

        # prepare data for epoch
        neg_users, neg_items = generate_neg(train_users, mat, nb_items, args.negative_samples)
        epoch_users = torch.cat((train_users,neg_users))
        epoch_items = torch.cat((train_items,neg_items))

        del neg_users, neg_items

        # shuffle prepared data and split into batches
        epoch_indices = torch.randperm(train_users_end - train_users_begin, device='cuda:{}'.format(args.local_rank))
        epoch_indices += train_users_begin

        epoch_users = epoch_users[epoch_indices]
        epoch_items = epoch_items[epoch_indices]
        epoch_label = train_label[epoch_indices]

        epoch_users_list = epoch_users.split(local_batch)
        epoch_items_list = epoch_items.split(local_batch)
        epoch_label_list = epoch_label.split(local_batch)

        # only print progress bar on rank 0
        num_batches = len(epoch_users_list)
        # handle extremely rare case where last batch size < number of worker
        if len(epoch_users) % args.batch_size < args.world_size:
            print("epoch_size % batch_size < number of worker!")
            exit(1)

        for i in range(num_batches // args.grads_accumulated):
            for j in range(args.grads_accumulated):
                batch_idx = (args.grads_accumulated * i) + j
                user = epoch_users_list[batch_idx]
                item = epoch_items_list[batch_idx]
                label = epoch_label_list[batch_idx].view(-1,1)

                outputs = model(user, item)
                loss = traced_criterion(outputs, label).float()
                loss = torch.mean(loss.view(-1), 0)
                if args.fp16:
                    fp_optimizer.backward(loss)
                else:
                    loss.backward()

            if args.fp16:
                fp_optimizer.step(optimizer)
            else:
                optimizer.step()

            for p in model.parameters():
                    p.grad = None

        del epoch_users, epoch_items, epoch_label, epoch_users_list, epoch_items_list, epoch_label_list, user, item, label
        train_time = time.time() - begin
        begin = time.time()

        epoch_samples = len(train_users) * (args.negative_samples + 1)
        train_throughput = epoch_samples / train_time
        train_throughputs.append(train_throughput)
        LOGGER.log(key='train_throughput', value=train_throughput)
        LOGGER.log(key=tags.TRAIN_EPOCH_STOP, value=epoch)
        LOGGER.log(key=tags.EVAL_START, value=epoch)

        hr, ndcg = val_epoch(model, test_x, test_y, dup_mask, real_indices, args.topk, samples_per_user=test_items.size(1),
                             num_user=all_test_users, output=valid_results_file, epoch=epoch, distributed=args.distributed)

        val_time = time.time() - begin
        print('Epoch {epoch}: HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f},'
              ' train_time = {train_time:.2f}, val_time = {val_time:.2f}'
              .format(epoch=epoch, K=args.topk, hit_rate=hr,
                      ndcg=ndcg, train_time=train_time,
                      val_time=val_time))

        LOGGER.log(key=tags.EVAL_ACCURACY, value={"epoch": epoch, "value": hr})
        LOGGER.log(key=tags.EVAL_TARGET, value=args.threshold)
        LOGGER.log(key=tags.EVAL_STOP, value=epoch)

        eval_size = all_test_users * test_items.size(1)
        eval_throughput = eval_size / val_time
        eval_throughputs.append(eval_throughput)
        LOGGER.log(key='eval_throughput', value=eval_throughput)

        if hr > max_hr and args.local_rank == 0:
            max_hr = hr
            print("New best hr! Saving the model to: ", checkpoint_path)
            torch.save(model.state_dict(), checkpoint_path)

        if args.threshold is not None:
            if hr >= args.threshold:
                print("Hit threshold of {}".format(args.threshold))
                success = True
                break

    LOGGER.log(key='best_train_throughput', value=max(train_throughputs))
    LOGGER.log(key='best_eval_throughput', value=max(eval_throughputs))
    LOGGER.log(key='best_accuracy', value=max_hr)
    LOGGER.log(key='time_to_target', value=time.time() - main_start_time)

    LOGGER.log(key=tags.RUN_STOP, value={"success": success})
    LOGGER.log(key=tags.RUN_FINAL)

if __name__ == '__main__':
    main()
