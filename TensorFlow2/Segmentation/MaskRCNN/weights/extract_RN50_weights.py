#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import sys
import getopt
import logging
import tensorflow as tf

"""
python weights/extract_RN50_weights.py \
    --checkpoint_dir=weights/mask-rcnn/1555659850/ckpt/model.ckpt \
    --save_to=weights/resnet/extracted_from_maskrcnn \
    --dry_run

python weights/extract_RN50_weights.py \
    --checkpoint_dir=weights/mask-rcnn/1555659850/ckpt/model.ckpt \
    --save_to=weights/resnet/extracted_from_maskrcnn
"""

usage_str = 'python tensorflow_rename_variables.py --checkpoint_dir=weights/inception_v4.ckpt ' \
            '--replace_from=substr --replace_to=substr --add_prefix=abc --dry_run'


def rename(checkpoint_dir, save_to, dry_run, verbose):

    _ = tf.train.get_checkpoint_state(checkpoint_dir)

    with tf.compat.v1.Session() as sess:

        total_vars_loaded = 0

        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):

            if "resnet50" in var_name:
                # Load the variable
                var = tf.train.load_variable(checkpoint_dir, var_name)
                total_vars_loaded += 1
            else:
                continue

            if not dry_run:
                _ = tf.Variable(var, name=var_name[9:])  # remove "resnet50/"
                # _ = tf.Variable(var, name=var_name)

            if verbose:
                print('Loading Variable: %s.' % var_name)

        print("Total Vars Loaded: %d" % total_vars_loaded)

        if not dry_run:

            if not os.path.isdir(save_to):
                os.makedirs(save_to)

            save_path = os.path.join(save_to, "resnet50.ckpt")
            print("Model save location: %s" % save_path)

            # Save the variables
            saver = tf.compat.v1.train.Saver()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.save(sess, save_path)


def main(argv):

    checkpoint_dir = None
    save_to = None
    dry_run = False
    verbose = False

    try:
        opts, args = getopt.getopt(
            argv, 'h', ['help=', 'checkpoint_dir=', 'save_to=', 'verbose', 'dry_run']
        )
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif opt == '--checkpoint_dir':
            checkpoint_dir = arg
        elif opt == '--save_to':
            save_to = arg
        elif opt == '--verbose':
            verbose = True
        elif opt == '--dry_run':
            dry_run = True

    if not checkpoint_dir:
        print('Please specify a checkpoint_dir. Usage:')
        print(usage_str)
        sys.exit(2)

    rename(checkpoint_dir, save_to, dry_run, verbose)


if __name__ == '__main__':

    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    main(sys.argv[1:])
