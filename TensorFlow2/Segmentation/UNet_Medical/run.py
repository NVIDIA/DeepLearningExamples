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
import os
from time import time

import numpy as np
from PIL import Image
import horovod.tensorflow as hvd
import tensorflow as tf

from utils.losses import partial_losses
from utils.parse_results import process_performance_stats


def restore_checkpoint(model, model_dir):
    try:
        model.load_weights(os.path.join(model_dir, "checkpoint"))
    except:
        print("Failed to load checkpoint, model will have randomly initialized weights.")
    return model


def train(params, model, dataset, logger):
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)
    max_steps = params.max_steps // hvd.size()

    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    if params.use_amp:
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

    ce_loss = tf.keras.metrics.Mean(name='ce_loss')
    f1_loss = tf.keras.metrics.Mean(name='dice_loss')

    @tf.function
    def train_step(features, labels, warmup_batch=False):
        with tf.GradientTape() as tape:
            output_map = model(features)
            crossentropy_loss, dice_loss = partial_losses(output_map, labels)
            added_losses = tf.add(crossentropy_loss, dice_loss, name="total_loss_ref")
            loss = added_losses + params.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in model.trainable_variables
                 if 'batch_normalization' not in v.name])

            if params.use_amp:
                loss = optimizer.get_scaled_loss(loss)
        tape = hvd.DistributedGradientTape(tape)
        gradients = tape.gradient(loss, model.trainable_variables)
        if params.use_amp:
            gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if warmup_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        ce_loss(crossentropy_loss)
        f1_loss(dice_loss)
        return loss

    if params.benchmark:
        assert max_steps * hvd.size() > params.warmup_steps, \
            "max_steps value has to be greater than warmup_steps"
        timestamps = np.zeros((hvd.size(), max_steps * hvd.size() + 1), dtype=np.float32)
        for iteration, (images, labels) in enumerate(dataset.train_fn(drop_remainder=True)):
            t0 = time()
            loss = train_step(images, labels, warmup_batch=iteration == 0).numpy()
            timestamps[hvd.rank(), iteration] = time() - t0
            if iteration >= max_steps * hvd.size():
                break
        timestamps = np.mean(timestamps, axis=0)
        if hvd.rank() == 0:
            throughput_imgps, latency_ms = process_performance_stats(timestamps, params)
            logger.log(step=(),
                       data={"throughput_train": throughput_imgps,
                             "latency_train": latency_ms})
    else:
        for iteration, (images, labels) in enumerate(dataset.train_fn()):
            train_step(images, labels, warmup_batch=iteration == 0)
            if (hvd.rank() == 0) and (iteration % params.log_every == 0):
                logger.log(step=(iteration, max_steps),
                           data={"train_ce_loss": float(ce_loss.result()),
                                 "train_dice_loss": float(f1_loss.result()),
                                 "train_total_loss": float(f1_loss.result() + ce_loss.result())})

                f1_loss.reset_states()
                ce_loss.reset_states()

            if iteration >= max_steps:
                break
        if hvd.rank() == 0:
            model.save_weights(os.path.join(params.model_dir, "checkpoint"))
    logger.flush()


def evaluate(params, model, dataset, logger):
    ce_loss = tf.keras.metrics.Mean(name='ce_loss')
    f1_loss = tf.keras.metrics.Mean(name='dice_loss')

    @tf.function
    def validation_step(features, labels):
        output_map = model(features, training=False)
        crossentropy_loss, dice_loss = partial_losses(output_map, labels)
        ce_loss(crossentropy_loss)
        f1_loss(dice_loss)

    for iteration, (images, labels) in enumerate(dataset.eval_fn(count=1)):
        validation_step(images, labels)
        if iteration >= dataset.eval_size // params.batch_size:
            break
    if dataset.eval_size > 0:
        logger.log(step=(),
                   data={"eval_ce_loss": float(ce_loss.result()),
                         "eval_dice_loss": float(f1_loss.result()),
                         "eval_total_loss": float(f1_loss.result() + ce_loss.result()),
                         "eval_dice_score": 1.0 - float(f1_loss.result())})

    logger.flush()


def predict(params, model, dataset, logger):

    @tf.function
    def prediction_step(features):
        return model(features, training=False)

    if params.benchmark:
        assert params.max_steps > params.warmup_steps, \
            "max_steps value has to be greater than warmup_steps"
        timestamps = np.zeros(params.max_steps + 1, dtype=np.float32)
        for iteration, images in enumerate(dataset.test_fn(count=None, drop_remainder=True)):
            t0 = time()
            prediction_step(images)
            timestamps[iteration] = time() - t0
            if iteration >= params.max_steps:
                break
        throughput_imgps, latency_ms = process_performance_stats(timestamps, params)
        logger.log(step=(),
                   data={"throughput_test": throughput_imgps,
                         "latency_test": latency_ms})
    else:
        predictions = np.concatenate([prediction_step(images).numpy()
                                      for images in dataset.test_fn(count=1)], axis=0)
        binary_masks = [np.argmax(p, axis=-1).astype(np.uint8) * 255 for p in predictions]
        multipage_tif = [Image.fromarray(mask).resize(size=(512, 512), resample=Image.BILINEAR)
                         for mask in binary_masks]

        output_dir = os.path.join(params.model_dir, 'predictions')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        multipage_tif[0].save(os.path.join(output_dir, 'test-masks.tif'),
                              compression="tiff_deflate",
                              save_all=True,
                              append_images=multipage_tif[1:])
    logger.flush()
