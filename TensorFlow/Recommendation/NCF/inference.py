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
import tensorflow as tf
from neumf import ncf_model_ops

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
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for inference')
    parser.add_argument('--num_batches', default=20, type=int,
                        help='Number of batches for which to measure latency and throughput')
    parser.add_argument('--no_amp', dest='amp', action='store_false', default=True,
                        help='Disable mixed precision')
    parser.add_argument('--xla', dest='xla', action='store_true', default=False,
                        help='Enable XLA')
    parser.add_argument('--log_path', default='nvlog.json', type=str,
                        help='Path to the path to store benchmark results')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.amp:
        os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

    # Input tensors
    users = tf.placeholder(tf.int32, shape=(None,))
    items = tf.placeholder(tf.int32, shape=(None,))
    dropout = tf.placeholder_with_default(0.0, shape=())

    # Model ops and saver
    logits_op = ncf_model_ops(
        users=users,
        items=items,
        labels=None,
        dup_mask=None,
        params={
            'fp16': False,
            'val_batch_size': args.batch_size,
            'num_users': args.n_users,
            'num_items': args.n_items,
            'num_factors': args.factors,
            'mf_reg': 0,
            'layer_sizes': args.layers,
            'layer_regs': [0. for i in args.layers],
            'dropout': 0.0,
            'sigmoid': True,
            'top_k': None,
            'learning_rate': None,
            'beta_1': None,
            'beta_2': None,
            'epsilon': None,
            'loss_scale': None,
        },
        mode='INFERENCE'
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if args.xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    if args.load_checkpoint_path:
        saver.restore(sess, args.load_checkpoint_path)
    else:
        # Manual initialize weights
        sess.run(tf.global_variables_initializer())

    sess.run(tf.local_variables_initializer())


    users_batch = np.random.randint(size=args.batch_size, low=0, high=args.n_users)
    items_batch = np.random.randint(size=args.batch_size, low=0, high=args.n_items)

    latencies = []
    for _ in range(args.num_batches):
        start = time.time()
        logits = sess.run(logits_op, feed_dict={users: users_batch, items: items_batch, dropout: 0.0 })
        latencies.append(time.time() - start)

    results = {
        'args' : vars(args),
        'best_inference_throughput' : args.batch_size / min(latencies),
        'best_inference_latency' : min(latencies),
        'inference_latencies' : latencies
    }
    print('RESULTS: ', json.dumps(results, indent=4))
    if args.log_path is not None:
        json.dump(results, open(args.log_path, 'w'), indent=4)

if __name__ == '__main__':
    main()
