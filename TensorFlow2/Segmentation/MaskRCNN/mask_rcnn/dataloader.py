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

"""Data loader and processing.

Defines input_fn of Mask-RCNN for TF Estimator. The input_fn includes training
data for category classification, bounding box regression, and number of
positive examples to normalize the loss during training.

"""
import functools
import math
import multiprocessing

import tensorflow as tf

from mask_rcnn.utils.logging_formatter import logging

from mask_rcnn.utils.distributed_utils import MPI_is_distributed
from mask_rcnn.utils.distributed_utils import MPI_rank_and_size
from mask_rcnn.utils.distributed_utils import MPI_rank
from mask_rcnn.utils.distributed_utils import MPI_size

# common functions
from mask_rcnn.dataloader_utils import dataset_parser

from distutils.version import LooseVersion

class InputReader(object):
    """Input reader for dataset."""

    def __init__(
        self,
        file_pattern,
        mode=tf.estimator.ModeKeys.TRAIN,
        num_examples=0,
        use_fake_data=False,
        use_instance_mask=False,
        seed=None
    ):

        self._mode = mode
        self._file_pattern = file_pattern
        self._num_examples = num_examples
        self._use_fake_data = use_fake_data
        self._use_instance_mask = use_instance_mask
        self._seed = seed

    def _create_dataset_parser_fn(self, params):
        """Create parser for parsing input data (dictionary)."""

        return functools.partial(
            dataset_parser,
            mode=self._mode,
            params=params,
            use_instance_mask=self._use_instance_mask,
            seed=self._seed
        )

    def __call__(self, params, input_context=None):

        batch_size = params['batch_size'] if 'batch_size' in params else 1

        try:
            seed = params['seed'] if not MPI_is_distributed() else params['seed'] * MPI_rank()
        except (KeyError, TypeError):
            seed = None

        if MPI_is_distributed():
            n_gpus = MPI_size()

        elif input_context is not None:
            n_gpus = input_context.num_input_pipelines

        else:
            n_gpus = 1

        ##################################################

        dataset = tf.data.Dataset.list_files(
            self._file_pattern,
            shuffle=False
        )

        if self._mode == tf.estimator.ModeKeys.TRAIN:

            if input_context is not None:
                logging.info("Using Dataset Sharding with TF Distributed")
                _num_shards = input_context.num_input_pipelines
                _shard_idx = input_context.input_pipeline_id

            elif MPI_is_distributed():
                logging.info("Using Dataset Sharding with Horovod")
                _shard_idx, _num_shards = MPI_rank_and_size()

            try:
                dataset = dataset.shard(
                    num_shards=_num_shards,
                    index=_shard_idx
                )
                dataset = dataset.shuffle(math.ceil(256 / _num_shards))

            except NameError:  # Not a distributed training setup
                pass

        def _prefetch_dataset(filename):
            return tf.data.TFRecordDataset(filename).prefetch(1)

        dataset = dataset.interleave(
            map_func=_prefetch_dataset,
            cycle_length=32,
            block_length=64,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        if self._num_examples is not None and self._num_examples > 0:
            logging.info("[*] Limiting the amount of sample to: %d" % self._num_examples)
            dataset = dataset.take(self._num_examples)

        dataset = dataset.cache()

        if self._mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(
                buffer_size=4096,
                reshuffle_each_iteration=True,
                seed=seed
            )

            dataset = dataset.repeat()

        # Parse the fetched records to input tensors for model function.
        dataset = dataset.map(
            map_func=self._create_dataset_parser_fn(params),
            num_parallel_calls=16,
        )

        dataset = dataset.batch(
            batch_size=batch_size,
            drop_remainder=True
        )

        if self._use_fake_data:
            # Turn this dataset into a semi-fake dataset which always loop at the
            # first batch. This reduces variance in performance and is useful in
            # testing.
            logging.info("Using Fake Dataset Loop...")
            dataset = dataset.take(1).cache().repeat()

            if self._mode != tf.estimator.ModeKeys.TRAIN:
                dataset = dataset.take(int(5000 / batch_size))

        dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE,
        )

        if not tf.distribute.has_strategy():
            dataset = dataset.apply(
                tf.data.experimental.prefetch_to_device(
                    '/gpu:0',  # With Horovod the local GPU is always 0
                    buffer_size=1,
                )
            )

        data_options = tf.data.Options()

        data_options.experimental_deterministic = seed is not None
        if LooseVersion(tf.__version__) <= LooseVersion("2.0.0"):
            data_options.experimental_distribute.auto_shard = False
        else:
            data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        # data_options.experimental_distribute.auto_shard = False
        data_options.experimental_slack = True

        data_options.experimental_threading.max_intra_op_parallelism = 1
        # data_options.experimental_threading.private_threadpool_size = int(multiprocessing.cpu_count() / n_gpus) * 2

        # ================= experimental_optimization ================= #

        data_options.experimental_optimization.apply_default_optimizations = False

        # data_options.experimental_optimization.autotune = True
        data_options.experimental_optimization.filter_fusion = True
        data_options.experimental_optimization.map_and_batch_fusion = True
        data_options.experimental_optimization.map_and_filter_fusion = True
        data_options.experimental_optimization.map_fusion = True
        data_options.experimental_optimization.map_parallelization = True

        map_vectorization_options = tf.data.experimental.MapVectorizationOptions()
        map_vectorization_options.enabled = True
        map_vectorization_options.use_choose_fastest = True

        data_options.experimental_optimization.map_vectorization = map_vectorization_options

        data_options.experimental_optimization.noop_elimination = True
        data_options.experimental_optimization.parallel_batch = True
        data_options.experimental_optimization.shuffle_and_repeat_fusion = True

        # ========== Stats on TF Data =============
        # aggregator = tf.data.experimental.StatsAggregator()
        # data_options.experimental_stats.aggregator = aggregator
        # data_options.experimental_stats.latency_all_edges = True

        dataset = dataset.with_options(data_options)

        return dataset


if __name__ == "__main__":
    '''
    Data Loading Benchmark Usage:

    # Real Data - Training
    python -m mask_rcnn.dataloader \
        --data_dir="/data/" \
        --batch_size=2 \
        --warmup_steps=200 \
        --benchmark_steps=2000 \
        --training

    # Real Data - Inference
    python -m mask_rcnn.dataloader \
        --data_dir="/data/" \
        --batch_size=8 \
        --warmup_steps=200 \
        --benchmark_steps=2000

    # --------------- #

    # Synthetic Data - Training
    python -m mask_rcnn.dataloader \
        --data_dir="/data/" \
        --batch_size=2 \
        --warmup_steps=200 \
        --benchmark_steps=2000 \
        --training \
        --use_synthetic_data

    # Synthetic Data - Inference
    python -m mask_rcnn.dataloader \
        --data_dir="/data/" \
        --batch_size=8 \
        --warmup_steps=200 \
        --benchmark_steps=2000 \
        --use_synthetic_data

    # --------------- #
    '''

    import os
    import time
    import argparse

    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    tf.compat.v1.disable_eager_execution()

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.set_verbosity(logging.INFO)

    parser = argparse.ArgumentParser(description="MaskRCNN Dataloader Benchmark")

    parser.add_argument(
        '--data_dir', required=True, type=str, help="Directory path which contains the preprocessed DAGM 2007 dataset"
    )

    parser.add_argument(
        '--batch_size', default=64, type=int, required=True, help="""Batch size used to measure performance."""
    )

    parser.add_argument(
        '--warmup_steps',
        default=200,
        type=int,
        required=True,
        help="""Number of steps considered as warmup and not taken into account for performance measurements."""
    )

    parser.add_argument(
        '--benchmark_steps',
        default=200,
        type=int,
        required=True,
        help="Number of steps used to benchmark dataloading performance. Only used in training"
    )

    parser.add_argument(
        '--seed',
        default=666,
        type=int,
        required=False,
        help="""Reproducibility Seed."""
    )

    parser.add_argument("--training", default=False, action="store_true", help="Benchmark in training mode")

    parser.add_argument("--use_synthetic_data", default=False, action="store_true", help="Use synthetic dataset")

    FLAGS, unknown_args = parser.parse_known_args()

    if len(unknown_args) > 0:

        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)

        raise ValueError("Invalid command line arg(s)")

    BURNIN_STEPS = FLAGS.warmup_steps

    if FLAGS.training:
        TOTAL_STEPS = FLAGS.warmup_steps + FLAGS.benchmark_steps
    else:
        TOTAL_STEPS = int(1e6)  # Wait for end of dataset

    if FLAGS.training:
        input_dataset = InputReader(
            file_pattern=os.path.join(FLAGS.data_dir, "train*.tfrecord"),
            mode=tf.estimator.ModeKeys.TRAIN,
            use_fake_data=FLAGS.use_synthetic_data,
            use_instance_mask=True,
            seed=FLAGS.seed
        )

    else:
        input_dataset = InputReader(
            file_pattern=os.path.join(FLAGS.data_dir, "val*.tfrecord"),
            mode=tf.estimator.ModeKeys.PREDICT,
            num_examples=5000,
            use_fake_data=FLAGS.use_synthetic_data,
            use_instance_mask=True,
            seed=FLAGS.seed
        )

    logging.info("[*] Executing Benchmark in %s mode" % ("training" if FLAGS.training else "inference"))
    logging.info("[*] Benchmark using %s data" % ("synthetic" if FLAGS.use_synthetic_data else "real"))

    time.sleep(1)

    # Build the data input
    dataset = input_dataset(
        params={
            "anchor_scale": 8.0,
            "aspect_ratios": [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]],
            "batch_size": FLAGS.batch_size,
            "gt_mask_size": 112,
            "image_size": [1024, 1024],
            "include_groundtruth_in_features": False,
            "augment_input_data": True,
            "max_level": 6,
            "min_level": 2,
            "num_classes": 91,
            "num_scales": 1,
            "rpn_batch_size_per_im": 256,
            "rpn_fg_fraction": 0.5,
            "rpn_min_size": 0.,
            "rpn_nms_threshold": 0.7,
            "rpn_negative_overlap": 0.3,
            "rpn_positive_overlap": 0.7,
            "rpn_post_nms_topn": 1000,
            "rpn_pre_nms_topn": 2000,
            "skip_crowd_during_training": True,
            "use_category": True,
            "visualize_images_summary": False,
        }
    )

    dataset_iterator = dataset.make_initializable_iterator()

    if FLAGS.training:
        X, Y = dataset_iterator.get_next()
    else:
        X = dataset_iterator.get_next()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False

    with tf.device("gpu:0"):

        X_gpu_ops = list()
        Y_gpu_ops = list()

        if FLAGS.training:

            for _, _x in X.items():
                X_gpu_ops.append(tf.identity(_x))

            for _, _y in Y.items():
                Y_gpu_ops.append(tf.identity(_y))

        else:

            for _, _x in X["features"].items():
                X_gpu_ops.append(tf.identity(_x))

        with tf.control_dependencies(X_gpu_ops + Y_gpu_ops):
            input_op = tf.constant(1.0)

        with tf.compat.v1.Session(config=config) as sess:

            sess.run(dataset_iterator.initializer)

            sess.run(tf.compat.v1.global_variables_initializer())

            total_files_processed = 0

            img_per_sec_arr = []
            processing_time_arr = []

            processing_start_time = time.time()

            for step in range(TOTAL_STEPS):

                try:

                    start_time = time.time()
                    sess.run(input_op)
                    elapsed_time = (time.time() - start_time) * 1000

                    imgs_per_sec = (FLAGS.batch_size / elapsed_time) * 1000
                    total_files_processed += FLAGS.batch_size

                    if (step + 1) > BURNIN_STEPS:
                        processing_time_arr.append(elapsed_time)
                        img_per_sec_arr.append(imgs_per_sec)

                    if (step + 1) % 20 == 0 or (step + 1) == TOTAL_STEPS:
                        print(
                            "[STEP %04d] # Batch Size: %03d - Time: %03d msecs - Speed: %6d img/s" %
                            (step + 1, FLAGS.batch_size, elapsed_time, imgs_per_sec)
                        )

                except tf.errors.OutOfRangeError:
                    break

            processing_time = time.time() - processing_start_time

            avg_processing_speed = np.mean(img_per_sec_arr)

            print("\n###################################################################")
            print("*** Data Loading Performance Metrics ***\n")
            print("\t=> Number of Steps: %d" % (step + 1))
            print("\t=> Batch Size: %d" % FLAGS.batch_size)
            print("\t=> Files Processed: %d" % total_files_processed)
            print("\t=> Total Execution Time: %d secs" % processing_time)
            print("\t=> Median Time per step: %3d msecs" % np.median(processing_time_arr))
            print("\t=> Median Processing Speed: %d images/secs" % np.median(img_per_sec_arr))
            print("\t=> Median Processing Time: %.2f msecs/image" % (1 / float(np.median(img_per_sec_arr)) * 1000))
