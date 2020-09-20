#!/usr/bin/python3

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from functools import partial
import json
import logging
from argparse import ArgumentParser

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import horovod.tensorflow as hvd
from mpi4py import MPI
import dllogger
import time

from vae.utils.round import round_8
from vae.metrics.recall import recall
from vae.metrics.ndcg import ndcg
from vae.models.train import VAE
from vae.load.preprocessing import load_and_parse_ML_20M

def main():
    hvd.init()
    mpi_comm = MPI.COMM_WORLD

    parser = ArgumentParser(description="Train a Variational Autoencoder for Collaborative Filtering in TensorFlow")
    parser.add_argument('--train', action='store_true',
                        help='Run training of VAE')
    parser.add_argument('--test', action='store_true',
                        help='Run validation of VAE')
    parser.add_argument('--inference_benchmark', action='store_true',
                        help='Measure inference latency and throughput on a variety of batch sizes')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable Automatic Mixed Precision')
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size_train', type=int, default=24576,
                        help='Global batch size for training')
    parser.add_argument('--batch_size_validation', type=int, default=10000,
                        help='Used both for validation and testing')
    parser.add_argument('--validation_step', type=int, default=50,
                        help='Train epochs for one validation')
    parser.add_argument('--warm_up_epochs', type=int, default=5,
                        help='Number of epochs to omit during benchmark')
    parser.add_argument('--total_anneal_steps', type=int, default=15000,
                        help='Number of annealing steps')
    parser.add_argument('--anneal_cap', type=float, default=0.1,
                        help='Annealing cap')
    parser.add_argument('--lam', type=float, default=1.00,
                        help='Regularization parameter')
    parser.add_argument('--lr', type=float, default=0.004,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.90,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.90,
                        help='Adam beta2')
    parser.add_argument('--top_results', type=int, default=100,
                        help='Number of results to be recommended')
    parser.add_argument('--xla', action='store_true', default=False,
                        help='Enable XLA')
    parser.add_argument('--trace', action='store_true', default=False,
                        help='Save profiling traces')
    parser.add_argument('--activation', type=str, default='tanh',
                        help='Activation function')
    parser.add_argument('--log_path', type=str, default='./vae_cf.log',
                        help='Path to the detailed training log to be created')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for TensorFlow and numpy')
    parser.add_argument('--data_dir', default='/data', type=str,
                        help='Directory for storing the training data')
    parser.add_argument('--checkpoint_dir', type=str,
                        default=None,
                        help='Path for saving a checkpoint after the training')
    args = parser.parse_args()
    args.world_size = hvd.size()

    if args.batch_size_train % hvd.size() != 0:
        raise ValueError('Global batch size should be a multiple of the number of workers')

    args.local_batch_size = args.batch_size_train // hvd.size()

    logger = logging.getLogger("VAE")
    if hvd.rank() == 0:
        logger.setLevel(logging.INFO)
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.log_path),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)])
    else:
        dllogger.init(backends=[])
        logger.setLevel(logging.ERROR)

    if args.seed is None:
        if hvd.rank() == 0:
            seed = int(time.time())
        else:
            seed = None

        seed = mpi_comm.bcast(seed, root=0)
    else:
        seed = args.seed

    tf.random.set_random_seed(seed)
    np.random.seed(seed)
    args.seed = seed

    dllogger.log(data=vars(args), step='PARAMETER')

    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # set AMP
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1' if args.amp else '0'

    # load dataset
    (train_data,
     validation_data_input,
     validation_data_true,
     test_data_input,
     test_data_true) = load_and_parse_ML_20M(args.data_dir)

    # make sure all dims and sizes are divisible by 8
    number_of_train_users, number_of_items = train_data.shape
    number_of_items = round_8(number_of_items)

    for data in [train_data,
                 validation_data_input,
                 validation_data_true,
                 test_data_input,
                 test_data_true]:
        number_of_users, _ = data.shape
        data.resize(number_of_users, number_of_items)

    number_of_users, number_of_items = train_data.shape
    encoder_dims = [number_of_items, 600, 200]

    vae = VAE(train_data, encoder_dims, total_anneal_steps=args.total_anneal_steps,
              anneal_cap=args.anneal_cap, batch_size_train=args.local_batch_size,
              batch_size_validation=args.batch_size_validation, lam=args.lam,
              lr=args.lr, beta1=args.beta1, beta2=args.beta2, activation=args.activation,
              xla=args.xla, checkpoint_dir=args.checkpoint_dir, trace=args.trace,
              top_results=args.top_results)

    metrics = {'ndcg@100': partial(ndcg, R=100),
               'recall@20': partial(recall, R=20),
               'recall@50': partial(recall, R=50)}

    if args.train:
        vae.train(n_epochs=args.epochs, validation_data_input=validation_data_input,
                  validation_data_true=validation_data_true,  metrics=metrics,
                  validation_step=args.validation_step)

    if args.test and hvd.size() <= 1:
        test_results = vae.test(test_data_input=test_data_input,
                                test_data_true=test_data_true, metrics=metrics)

        for k, v in test_results.items():
            print("{}:\t{}".format(k, v))
    elif args.test and hvd.size() > 1:
        print("Testing is not supported with horovod multigpu yet")

    elif args.test and hvd.size() > 1:
        print("Testing is not supported with horovod multigpu yet")

    if args.inference_benchmark:
        items_per_user = 10
        item_indices = np.random.randint(low=0, high=10000, size=items_per_user)
        user_indices = np.zeros(len(item_indices))
        indices = np.stack([user_indices, item_indices], axis=1)

        num_batches = 200
        latencies = []
        for i in range(num_batches):
            start_time = time.time()
            _ = vae.query(indices=indices)
            end_time = time.time()

            if i < 10:
                #warmup steps
                continue

            latencies.append(end_time - start_time)

        result_data = {}
        result_data[f'batch_1_mean_throughput'] = 1 / np.mean(latencies)
        result_data[f'batch_1_mean_latency'] = np.mean(latencies)
        result_data[f'batch_1_p90_latency'] = np.percentile(latencies, 90)
        result_data[f'batch_1_p95_latency'] = np.percentile(latencies, 95)
        result_data[f'batch_1_p99_latency'] = np.percentile(latencies, 99)

        dllogger.log(data=result_data, step=tuple())

    vae.close_session()
    dllogger.flush()

if __name__ == '__main__':
    main()
