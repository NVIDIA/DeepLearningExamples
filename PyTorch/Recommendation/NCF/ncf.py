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
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import os
import math
import time
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn

import utils
import dataloading
from neumf import NeuMF
from feature_spec import FeatureSpec
from neumf_constants import USER_CHANNEL_NAME, ITEM_CHANNEL_NAME, LABEL_CHANNEL_NAME

import dllogger


def synchronized_timestamp():
    torch.cuda.synchronize()
    return time.time()

def parse_args():
    parser = ArgumentParser(description="Train a Neural Collaborative"
                                        " Filtering model")
    parser.add_argument('--data', type=str,
                        help='Path to the directory containing the feature specification yaml')
    parser.add_argument('--feature_spec_file', type=str, default='feature_spec.yaml',
                        help='Name of the feature specification file or path relative to the data directory.')
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='Number of epochs for training')
    parser.add_argument('-b', '--batch_size', type=int, default=2 ** 20,
                        help='Number of examples for each iteration. This will be divided by the number of devices')
    parser.add_argument('--valid_batch_size', type=int, default=2 ** 20,
                        help='Number of examples in each validation chunk. This will be the maximum size of a batch '
                             'on each device.')
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
    parser.add_argument('--seed', '-s', type=int, default=None,
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
    parser.add_argument('--checkpoint_dir', default='', type=str,
                        help='Path to the directory storing the checkpoint file, '
                             'passing an empty path disables checkpoint saving')
    parser.add_argument('--load_checkpoint_path', default=None, type=str,
                        help='Path to the checkpoint file to be loaded before training/evaluation')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', type=str,
                        help='Passing "test" will only run a single evaluation; '
                             'otherwise, full training will be performed')
    parser.add_argument('--grads_accumulated', default=1, type=int,
                        help='Number of gradients to accumulate before performing an optimization step')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--log_path', default='log.json', type=str,
                        help='Path for the JSON training log')
    return parser.parse_args()


def init_distributed(args):
    args.world_size = int(os.environ.get('WORLD_SIZE', default=1))
    args.distributed = args.world_size > 1

    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])

        '''
        Set cuda device so everything is done on the right GPU.
        THIS MUST BE DONE AS SOON AS POSSIBLE.
        '''
        torch.cuda.set_device(args.local_rank)

        '''Initialize distributed communication'''
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    else:
        args.local_rank = 0


def val_epoch(model, dataloader: dataloading.TestDataLoader, k, distributed=False, world_size=1):
    model.eval()
    user_feature_name = dataloader.channel_spec[USER_CHANNEL_NAME][0]
    item_feature_name = dataloader.channel_spec[ITEM_CHANNEL_NAME][0]
    label_feature_name = dataloader.channel_spec[LABEL_CHANNEL_NAME][0]
    with torch.no_grad():
        p = []
        labels_list = []
        losses = []
        for batch_dict in dataloader.get_epoch_data():
            user_batch = batch_dict[USER_CHANNEL_NAME][user_feature_name]
            item_batch = batch_dict[ITEM_CHANNEL_NAME][item_feature_name]
            label_batch = batch_dict[LABEL_CHANNEL_NAME][label_feature_name]
            prediction_batch = model(user_batch, item_batch, sigmoid=True).detach()

            loss_batch = torch.nn.functional.binary_cross_entropy(input=prediction_batch.reshape([-1]),
                                                                  target=label_batch)
            losses.append(loss_batch)

            p.append(prediction_batch)
            labels_list.append(label_batch)

        ignore_mask = dataloader.get_ignore_mask().view(-1, dataloader.samples_in_series)
        ratings = torch.cat(p).view(-1, dataloader.samples_in_series)
        ratings[ignore_mask] = -1
        labels = torch.cat(labels_list).view(-1, dataloader.samples_in_series)
        del p, labels_list

        top_indices = torch.topk(ratings, k)[1]

        # Positive items are always first in a given series
        labels_of_selected = torch.gather(labels, 1, top_indices)
        ifzero = (labels_of_selected == 1)
        hits = ifzero.sum()
        ndcg = (math.log(2) / (torch.nonzero(ifzero)[:, 1].view(-1).to(torch.float) + 2).log_()).sum()
        total_validation_loss = torch.mean(torch.stack(losses, dim=0))
        #  torch.nonzero may cause host-device synchronization

    if distributed:
        torch.distributed.all_reduce(hits, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(ndcg, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_validation_loss, op=torch.distributed.ReduceOp.SUM)
        total_validation_loss = total_validation_loss / world_size


    num_test_cases = dataloader.raw_dataset_length / dataloader.samples_in_series
    hr = hits.item() / num_test_cases
    ndcg = ndcg.item() / num_test_cases

    model.train()
    return hr, ndcg, total_validation_loss


def main():
    args = parse_args()
    init_distributed(args)

    if args.local_rank == 0:
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.log_path),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)])
    else:
        dllogger.init(backends=[])

    dllogger.metadata('train_throughput', {"name": 'train_throughput', 'unit': 'samples/s', 'format': ":.3e"})
    dllogger.metadata('best_train_throughput', {'unit': 'samples/s'})
    dllogger.metadata('mean_train_throughput', {'unit': 'samples/s'})
    dllogger.metadata('eval_throughput', {"name": 'eval_throughput', 'unit': 'samples/s', 'format': ":.3e"})
    dllogger.metadata('best_eval_throughput', {'unit': 'samples/s'})
    dllogger.metadata('mean_eval_throughput', {'unit': 'samples/s'})
    dllogger.metadata('train_epoch_time', {"name": 'train_epoch_time', 'unit': 's', 'format': ":.3f"})
    dllogger.metadata('validation_epoch_time', {"name": 'validation_epoch_time', 'unit': 's', 'format': ":.3f"})
    dllogger.metadata('time_to_target', {'unit': 's'})
    dllogger.metadata('time_to_best_model', {'unit': 's'})
    dllogger.metadata('hr@10', {"name": 'hr@10', 'unit': None, 'format': ":.5f"})
    dllogger.metadata('best_accuracy', {'unit': None})
    dllogger.metadata('best_epoch', {'unit': None})
    dllogger.metadata('validation_loss', {"name": 'validation_loss', 'unit': None, 'format': ":.5f"})
    dllogger.metadata('train_loss', {"name": 'train_loss', 'unit': None, 'format': ":.5f"})

    dllogger.log(data=vars(args), step='PARAMETER')

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if not os.path.exists(args.checkpoint_dir) and args.checkpoint_dir:
        print("Saving results to {}".format(args.checkpoint_dir))
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # sync workers before timing
    if args.distributed:
        torch.distributed.broadcast(torch.tensor([1], device="cuda"), 0)
    torch.cuda.synchronize()

    main_start_time = synchronized_timestamp()

    feature_spec_path = os.path.join(args.data, args.feature_spec_file)
    feature_spec = FeatureSpec.from_yaml(feature_spec_path)
    trainset = dataloading.TorchTensorDataset(feature_spec, mapping_name='train', args=args)
    testset = dataloading.TorchTensorDataset(feature_spec, mapping_name='test', args=args)
    train_loader = dataloading.TrainDataloader(trainset, args)
    test_loader = dataloading.TestDataLoader(testset, args)

    # make pytorch memory behavior more consistent later
    torch.cuda.empty_cache()

    # Create model
    user_feature_name = feature_spec.channel_spec[USER_CHANNEL_NAME][0]
    item_feature_name = feature_spec.channel_spec[ITEM_CHANNEL_NAME][0]
    label_feature_name = feature_spec.channel_spec[LABEL_CHANNEL_NAME][0]
    model = NeuMF(nb_users=feature_spec.feature_spec[user_feature_name]['cardinality'],
                  nb_items=feature_spec.feature_spec[item_feature_name]['cardinality'],
                  mf_dim=args.factors,
                  mlp_layer_sizes=args.layers,
                  dropout=args.dropout)

    optimizer = FusedAdam(model.parameters(), lr=args.learning_rate,
                          betas=(args.beta1, args.beta2), eps=args.eps)

    criterion = nn.BCEWithLogitsLoss(reduction='none')  # use torch.mean() with dim later to avoid copy to host
    # Move model and loss to GPU
    model = model.cuda()
    criterion = criterion.cuda()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)

    local_batch = args.batch_size // args.world_size
    traced_criterion = torch.jit.trace(criterion.forward,
                                       (torch.rand(local_batch, 1), torch.rand(local_batch, 1)))

    print(model)
    print("{} parameters".format(utils.count_parameters(model)))

    if args.load_checkpoint_path:
        state_dict = torch.load(args.load_checkpoint_path)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    if args.mode == 'test':
        start = synchronized_timestamp()
        hr, ndcg, val_loss = val_epoch(model, test_loader, args.topk,
                                       distributed=args.distributed, world_size=args.world_size)
        val_time = synchronized_timestamp() - start
        eval_size = test_loader.raw_dataset_length
        eval_throughput = eval_size / val_time

        dllogger.log(step=tuple(), data={'best_eval_throughput': eval_throughput,
                                         'hr@10': hr,
                                         'validation_loss': float(val_loss.item())})
        return

    # this should always be overridden if hr>0.
    # It is theoretically possible for the hit rate to be zero in the first epoch, which would result in referring
    # to an uninitialized variable.
    max_hr = 0
    best_epoch = 0
    best_model_timestamp = synchronized_timestamp()
    train_throughputs, eval_throughputs = [], []
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for epoch in range(args.epochs):

        begin = synchronized_timestamp()
        batch_dict_list = train_loader.get_epoch_data()
        num_batches = len(batch_dict_list)
        for i in range(num_batches // args.grads_accumulated):
            for j in range(args.grads_accumulated):
                batch_idx = (args.grads_accumulated * i) + j
                batch_dict = batch_dict_list[batch_idx]

                user_features = batch_dict[USER_CHANNEL_NAME]
                item_features = batch_dict[ITEM_CHANNEL_NAME]

                user_batch = user_features[user_feature_name]
                item_batch = item_features[item_feature_name]

                label_features = batch_dict[LABEL_CHANNEL_NAME]
                label_batch = label_features[label_feature_name]

                with torch.cuda.amp.autocast(enabled=args.amp):
                    outputs = model(user_batch, item_batch)
                    loss = traced_criterion(outputs, label_batch.view(-1, 1))
                    loss = torch.mean(loss.float().view(-1), 0)

                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            for p in model.parameters():
                p.grad = None

        del batch_dict_list
        train_time = synchronized_timestamp() - begin
        begin = synchronized_timestamp()

        epoch_samples = train_loader.length_after_augmentation
        train_throughput = epoch_samples / train_time
        train_throughputs.append(train_throughput)

        hr, ndcg, val_loss = val_epoch(model, test_loader, args.topk,
                                       distributed=args.distributed, world_size=args.world_size)

        val_time = synchronized_timestamp() - begin
        eval_size = test_loader.raw_dataset_length
        eval_throughput = eval_size / val_time
        eval_throughputs.append(eval_throughput)

        if args.distributed:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            loss = loss / args.world_size

        dllogger.log(step=(epoch,),
                     data={'train_throughput': train_throughput,
                           'hr@10': hr,
                           'train_epoch_time': train_time,
                           'validation_epoch_time': val_time,
                           'eval_throughput': eval_throughput,
                           'validation_loss': float(val_loss.item()),
                           'train_loss': float(loss.item())})

        if hr > max_hr and args.local_rank == 0:
            max_hr = hr
            best_epoch = epoch
            print("New best hr!")
            if args.checkpoint_dir:
                save_checkpoint_path = os.path.join(args.checkpoint_dir, 'model.pth')
                print("Saving the model to: ", save_checkpoint_path)
                torch.save(model.state_dict(), save_checkpoint_path)
            best_model_timestamp = synchronized_timestamp()

        if args.threshold is not None:
            if hr >= args.threshold:
                print("Hit threshold of {}".format(args.threshold))
                break

    if args.local_rank == 0:
        dllogger.log(data={'best_train_throughput': max(train_throughputs),
                           'best_eval_throughput': max(eval_throughputs),
                           'mean_train_throughput': np.mean(train_throughputs),
                           'mean_eval_throughput': np.mean(eval_throughputs),
                           'best_accuracy': max_hr,
                           'best_epoch': best_epoch,
                           'time_to_target': synchronized_timestamp() - main_start_time,
                           'time_to_best_model': best_model_timestamp - main_start_time,
                           'validation_loss': float(val_loss.item()),
                           'train_loss': float(loss.item())},
                     step=tuple())


if __name__ == '__main__':
    main()
