#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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


import time
import os
import json
import argparse

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from neumf import ncf_model_ops
import dllogger


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference performance of the NCF model")
    parser.add_argument('--load_checkpoint_path', default=None, type=str,
                        help='Path to the checkpoint file to be loaded. If None will use random weights')
    parser.add_argument('--n_users', default=138493, type=int,
                        help='Number of users. Defaults to the number of users in the ml-20m dataset after preprocessing')
    parser.add_argument('--n_items', default=26744, type=int,
                        help='Number of items. Defaults to the number of users in the ml-20m dataset after preprocessing')
    parser.add_argument('-f', '--factors', type=int, default=64,
                        help='Number of predictive factors')
    parser.add_argument('--layers', nargs='+', type=int,
                        default=[256, 256, 128, 64],
                        help='Sizes of hidden layers for MLP')
    parser.add_argument('--batch_sizes', default='1,4,16,64,256,1024,4096,16384,65536,262144,1048576', type=str,
                        help='A list of comma-separated batch size values to benchmark')
    parser.add_argument('--num_batches', default=200, type=int,
                        help='Number of batches for which to measure latency and throughput')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable automatic mixed precision')
    parser.add_argument('--xla', dest='xla', action='store_true', default=False,
                        help='Enable XLA')
    parser.add_argument('--log_path', default='log.json', type=str,
                        help='Path to the path to store benchmark results')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.amp:
        os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

    dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                       filename=args.log_path),
                            dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)])
    dllogger.log(data=vars(args), step='PARAMETER')

    batch_sizes = args.batch_sizes.split(',')
    batch_sizes = [int(s) for s in batch_sizes]
    result_data = {}

    for batch_size in batch_sizes:
        print('Benchmarking batch size', batch_size)
        tf.reset_default_graph()

        # Input tensors
        users = tf.placeholder(tf.int32, shape=(None,))
        items = tf.placeholder(tf.int32, shape=(None,))
        dropout = tf.placeholder_with_default(0.0, shape=())

        # Model ops and saver
        logits_op = ncf_model_ops(users=users, items=items, labels=None, dup_mask=None, mode='INFERENCE',
                                  params={'fp16': False, 'val_batch_size': batch_size, 'num_users': args.n_users,
                                          'num_items': args.n_items, 'num_factors': args.factors, 'mf_reg': 0,
                                          'layer_sizes': args.layers, 'layer_regs': [0. for i in args.layers],
                                          'dropout': 0.0, 'sigmoid': True, 'top_k': None, 'learning_rate': None,
                                          'beta_1': None, 'beta_2': None, 'epsilon': None, 'loss_scale': None, })

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if args.xla:
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        sess = tf.Session(config=config)

        saver = tf.train.Saver()
        if args.load_checkpoint_path:
            saver.restore(sess, args.load_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        sess.run(tf.local_variables_initializer())

        users_batch = np.random.randint(size=batch_size, low=0, high=args.n_users)
        items_batch = np.random.randint(size=batch_size, low=0, high=args.n_items)

        latencies = []
        for i in range(args.num_batches):
            start = time.time()
            _ = sess.run(logits_op, feed_dict={users: users_batch, items: items_batch, dropout: 0.0 })
            end = time.time()

            if i < 10: # warmup iterations
                continue

            latencies.append(end - start)

        result_data[f'batch_{batch_size}_mean_throughput'] = batch_size / np.mean(latencies)
        result_data[f'batch_{batch_size}_mean_latency'] = np.mean(latencies)
        result_data[f'batch_{batch_size}_p90_latency'] = np.percentile(latencies, 90)
        result_data[f'batch_{batch_size}_p95_latency'] = np.percentile(latencies, 95)
        result_data[f'batch_{batch_size}_p99_latency'] = np.percentile(latencies, 99)

    dllogger.log(data=result_data, step=tuple())
    dllogger.flush()


if __name__ == '__main__':
    main()
