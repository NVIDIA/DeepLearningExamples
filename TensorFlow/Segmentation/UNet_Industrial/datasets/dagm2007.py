#!/usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
# Copyright (c) Jonathan Dekhtiar - contact@jonathandekhtiar.eu
# All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

# ==============================================================================
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
#
# ==============================================================================

import os
import glob

import tensorflow as tf
import horovod.tensorflow as hvd

from datasets.core import BaseDataset

from utils import hvd_utils

from dllogger import Logger

__all__ = ['DAGM2007_Dataset']


class DAGM2007_Dataset(BaseDataset):

    dataset_name = "DAGM2007"

    def __init__(self, data_dir, class_id):

        if class_id is None:
            raise ValueError("The parameter `class_id` cannot be set to None")

        data_dir = os.path.join(data_dir, "raw_images/private/Class%d" % class_id)

        super(DAGM2007_Dataset, self).__init__(data_dir)

    def _get_data_dirs(self, training):

        if training:
            csv_file = os.path.join(self.data_dir, "train_list.csv")
            image_dir = os.path.join(self.data_dir, "Train")

        else:
            csv_file = os.path.join(self.data_dir, "test_list.csv")
            image_dir = os.path.join(self.data_dir, "Test")

        return image_dir, csv_file

    def get_dataset_runtime_specs(self, training, iter_unit, num_iter, global_batch_size):

        image_dir, _ = self._get_data_dirs(training=training)

        filenames = glob.glob(os.path.join(image_dir, "*.PNG"))
        num_samples = len(filenames)

        num_steps, num_epochs = DAGM2007_Dataset._count_steps(
            iter_unit=iter_unit, num_samples=num_samples, num_iter=num_iter, global_batch_size=global_batch_size
        )

        return filenames, num_samples, num_steps, num_epochs

    def dataset_fn(
        self,
        batch_size,
        training,
        input_shape,
        mask_shape,
        num_threads,
        use_gpu_prefetch,
        normalize_data_method,
        only_defective_images,
        augment_data,
        seed=None
    ):

        super(DAGM2007_Dataset, self).dataset_fn(
            batch_size=batch_size,
            training=training,
            input_shape=input_shape,
            mask_shape=mask_shape,
            num_threads=num_threads,
            use_gpu_prefetch=use_gpu_prefetch,
            normalize_data_method=normalize_data_method,  # [None, "zero_centered", "zero_one"]
            only_defective_images=only_defective_images,
            augment_data=augment_data,
            seed=seed
        )

        shuffle_buffer_size = 10000

        image_dir, csv_file = self._get_data_dirs(training=training)

        mask_image_dir = os.path.join(image_dir, "Label")

        dataset = tf.data.TextLineDataset(csv_file)

        dataset = dataset.skip(1)  # Skip CSV Header

        if only_defective_images:
            dataset = dataset.filter(lambda line: tf.not_equal(tf.strings.substr(line, -1, 1), "0"))

        if hvd_utils.is_using_hvd() and training:
            dataset = dataset.shard(hvd.size(), hvd.rank())

        def _load_dagm_data(line):

            input_image_name, image_mask_name, label = tf.decode_csv(
                line, record_defaults=[[""], [""], [0]], field_delim=','
            )

            def decode_image(filepath, resize_shape, normalize_data_method):
                image_content = tf.read_file(filepath)

                # image = tf.image.decode_image(image_content, channels=resize_shape[-1])
                image = tf.image.decode_png(contents=image_content, channels=resize_shape[-1], dtype=tf.uint8)

                image = tf.image.resize_images(
                    image,
                    size=resize_shape[:2],
                    method=tf.image.ResizeMethod.BILINEAR,  # [BILINEAR, NEAREST_NEIGHBOR, BICUBIC, AREA]
                    align_corners=False,
                    preserve_aspect_ratio=True
                )

                image.set_shape(resize_shape)
                image = tf.cast(image, tf.float32)

                if normalize_data_method == "zero_centered":
                    image = tf.divide(image, 127.5) - 1

                elif normalize_data_method == "zero_one":
                    image = tf.divide(image, 255.0)

                return image

            input_image = decode_image(
                filepath=tf.strings.join([image_dir, input_image_name], separator='/'),
                resize_shape=input_shape,
                normalize_data_method=normalize_data_method,
            )

            mask_image = tf.cond(
                tf.equal(image_mask_name, ""),
                true_fn=lambda: tf.zeros(mask_shape, dtype=tf.float32),
                false_fn=lambda: decode_image(
                    filepath=tf.strings.join([mask_image_dir, image_mask_name], separator='/'),
                    resize_shape=mask_shape,
                    normalize_data_method="zero_one",
                ),
            )

            label = tf.cast(label, tf.int32)

            return tf.data.Dataset.from_tensor_slices(([input_image], [mask_image], [label]))

        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(
                _load_dagm_data,
                cycle_length=batch_size*8,
                block_length=4,
                buffer_output_elements=batch_size*8
            )
        )

        dataset = dataset.cache()

        if training:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=shuffle_buffer_size, seed=seed))

        else:
            dataset = dataset.repeat()

        def _augment_data(input_image, mask_image, label):

            if augment_data:

                if not hvd_utils.is_using_hvd() or hvd.rank() == 0:
                    print("Using data augmentation ...")

                #input_image = tf.image.per_image_standardization(input_image)

                horizontal_flip = tf.random_uniform(shape=(), seed=seed) > 0.5
                input_image = tf.cond(
                    horizontal_flip, lambda: tf.image.flip_left_right(input_image), lambda: input_image
                )
                mask_image = tf.cond(horizontal_flip, lambda: tf.image.flip_left_right(mask_image), lambda: mask_image)

                n_rots = tf.random_uniform(shape=(), dtype=tf.int32, minval=0, maxval=3, seed=seed)
                input_image = tf.image.rot90(input_image, k=n_rots)
                mask_image = tf.image.rot90(mask_image, k=n_rots)

            return (input_image, mask_image), label

        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                map_func=_augment_data,
                num_parallel_calls=num_threads,
                batch_size=batch_size,
                drop_remainder=True,
            )
        )

        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

        if use_gpu_prefetch:
            dataset.apply(tf.data.experimental.prefetch_to_device(device="/gpu:0", buffer_size=4))

        return dataset


if __name__ == "__main__":
    '''
    Data Loading Benchmark Usage:

    # Real Data - Training
    python -m datasets.dagm2007 \
        --data_dir="/data/dagm2007/" \
        --batch_size=64 \
        --warmup_steps=200 \
        --benchmark_steps=2000 \
        --training \
        --class_id=1

    # Real Data - Inference
    python -m datasets.dagm2007 \
        --data_dir="/data/dagm2007/" \
        --batch_size=64 \
        --warmup_steps=200 \
        --benchmark_steps=2000 \
        --class_id=1

    # --------------- #

    # Synthetic Data - Training
    python -m datasets.dagm2007 \
        --data_dir="/data/dagm2007/" \
        --batch_size=64 \
        --warmup_steps=200 \
        --benchmark_steps=2000 \
        --class_id=1 \
        --training \
        --use_synthetic_data

    # Synthetic Data - Inference
    python -m datasets.dagm2007 \
        --data_dir="/data/dagm2007/" \
        --batch_size=64 \
        --warmup_steps=200 \
        --benchmark_steps=2000 \
        --class_id=1 \
        --use_synthetic_data

    # --------------- #
    '''

    import time
    import argparse

    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser(description="DAGM2007_data_loader_benchmark")

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
        help="""Number of steps considered as warmup and not taken into account for performance measurements."""
    )

    parser.add_argument(
        '--class_id',
        default=1,
        choices=range(1, 11),  # between 1 and 10
        type=int,
        required=True,
        help="""Class ID used for benchmark."""
    )

    parser.add_argument("--training", default=False, action="store_true", help="Benchmark in training mode")

    parser.add_argument("--use_synthetic_data", default=False, action="store_true", help="Use synthetic dataset")

    FLAGS, unknown_args = parser.parse_known_args()

    if len(unknown_args) > 0:

        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)

        raise ValueError("Invalid command line arg(s)")

    BURNIN_STEPS = FLAGS.warmup_steps
    TOTAL_STEPS = FLAGS.warmup_steps + FLAGS.benchmark_steps

    dataset = DAGM2007_Dataset(data_dir=FLAGS.data_dir, class_id=FLAGS.class_id)

    _filenames, _num_samples, _num_steps, _num_epochs = dataset.get_dataset_runtime_specs(
        training=FLAGS.training, iter_unit="batch", num_iter=TOTAL_STEPS, global_batch_size=FLAGS.batch_size
    )

    tf.logging.info("[*] Executing Benchmark in %s mode" % ("training" if FLAGS.training else "inference"))
    tf.logging.info("[*] Benchmark using %s data" % ("synthetic" if FLAGS.use_synthetic_data else "real"))

    print()
    tf.logging.info("[*] num_samples: %d" % _num_samples)
    tf.logging.info("[*] num_steps: %d" % _num_steps)
    tf.logging.info("[*] num_epochs: %d" % _num_epochs)

    time.sleep(4)

    if not FLAGS.use_synthetic_data:
        # Build the data input
        dataset = dataset.dataset_fn(
            batch_size=FLAGS.batch_size,
            training=FLAGS.training,
            input_shape=(512, 512, 1),
            mask_shape=(512, 512, 1),
            num_threads=64,
            use_gpu_prefetch=True,
            seed=None
        )

    else:
        # Build the data input
        dataset = dataset.synth_dataset_fn(
            batch_size=FLAGS.batch_size,
            training=FLAGS.training,
            input_shape=(512, 512, 1),
            mask_shape=(512, 512, 1),
            num_threads=64,
            use_gpu_prefetch=True,
            seed=None
        )

    dataset_iterator = dataset.make_initializable_iterator()

    (input_images, mask_images), labels = dataset_iterator.get_next()

    print("Input Image Shape: %s" % (input_images.get_shape()))
    print("Mask Image Shape: %s" % (mask_images.get_shape()))
    print("Label Shape: %s" % (labels.get_shape()))

    input_images = tf.image.resize_image_with_crop_or_pad(input_images, target_height=512, target_width=512)

    with tf.device("/gpu:0"):

        input_images = tf.identity(input_images)
        mask_images = tf.identity(mask_images)
        labels = tf.identity(labels)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    with tf.Session(config=config) as sess:

        sess.run(dataset_iterator.initializer)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.global_variables_initializer())

        total_files_processed = 0

        img_per_sec_arr = []
        processing_time_arr = []

        processing_start_time = time.time()

        for step in range(TOTAL_STEPS):

            start_time = time.time()

            img_batch, mask_batch, lbl_batch = sess.run([input_images, mask_images, labels])

            batch_size = img_batch.shape[0]
            total_files_processed += batch_size

            elapsed_time = (time.time() - start_time) * 1000
            imgs_per_sec = (batch_size / elapsed_time) * 1000

            if (step + 1) > BURNIN_STEPS:
                processing_time_arr.append(elapsed_time)
                img_per_sec_arr.append(imgs_per_sec)

            if (step + 1) % 20 == 0 or (step + 1) == TOTAL_STEPS:

                print(
                    "[STEP %04d] # Files: %03d - Time: %03d msecs - Speed: %6d img/s" %
                    (step + 1, batch_size, elapsed_time, imgs_per_sec)
                )

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

        print("\n*** Debug Shape Information:")
        print(
            "\t[*] Batch Shape: %s - Max Val: %.2f - Min Val: %.2f - Mean: %.2f - Stddev: %.2f" % (
                str(img_batch.shape), np.max(img_batch), np.min(img_batch), float(np.mean(img_batch)),
                float(np.std(img_batch))
            )
        )
        print(
            "\t[*] Mask Shape: %s - Max Val: %.2f - Min Val: %.2f - Mean: %.2f - Stddev: %.2f" % (
                str(mask_batch.shape), np.max(mask_batch), np.min(mask_batch), float(np.mean(mask_batch)),
                float(np.std(mask_batch))
            )
        )
        print(
            "\t[*] Label Shape: %s - Max Val: %.2f - Min Val: %.2f" %
            (str(lbl_batch.shape), np.max(lbl_batch), np.min(lbl_batch))
        )
