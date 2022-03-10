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
import os
from time import time

import numpy as np
from PIL import Image
import horovod.tensorflow as hvd
import tensorflow as tf

from runtime.losses import partial_losses
from runtime.parse_results import process_performance_stats
from model.tf_trt import export_model, TFTRTModel


def train(params, model, dataset, logger):
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)
    max_steps = params.max_steps // hvd.size()

    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    if params.use_amp:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)

    ce_loss = tf.keras.metrics.Mean(name='ce_loss')
    f1_loss = tf.keras.metrics.Mean(name='dice_loss')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    if params.resume_training and params.model_dir:
        checkpoint.restore(tf.train.latest_checkpoint(params.model_dir))

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
        timestamps = []
        for iteration, (images, labels) in enumerate(dataset.train_fn(drop_remainder=True)):
            loss = train_step(images, labels, warmup_batch=iteration == 0).numpy()
            if iteration > params.warmup_steps:
                timestamps.append(time())
            if iteration >= max_steps * hvd.size():
                break

        if hvd.rank() == 0:
            deltas = np.array([timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)])
            stats = process_performance_stats(deltas, hvd.size() * params.batch_size, mode="train")
            logger.log(step=(), data=stats)
    else:
        for iteration, (images, labels) in enumerate(dataset.train_fn()):
            train_step(images, labels, warmup_batch=iteration == 0)
            if hvd.rank() == 0:
                if iteration % params.log_every == 0:
                    logger.log(step=(iteration, max_steps),
                               data={"train_ce_loss": float(ce_loss.result()),
                                     "train_dice_loss": float(f1_loss.result()),
                                     "train_total_loss": float(f1_loss.result() + ce_loss.result())})

                if (params.evaluate_every > 0) and (iteration % params.evaluate_every == 0):
                    evaluate(params, model, dataset, logger, restore_checkpoint=False)

                f1_loss.reset_states()
                ce_loss.reset_states()

            if iteration >= max_steps:
                break
        if hvd.rank() == 0:
            checkpoint.save(file_prefix=os.path.join(params.model_dir, "checkpoint"))
            if params.use_savedmodel:
                prec = 'amp' if params.use_amp else 'fp32'
                model.save(os.path.join(params.model_dir, f'saved_model_{prec}'))
                if params.use_tftrt:
                    export_model(params.model_dir, prec, os.path.join(params.model_dir, f'tf-trt_model_{prec}'))

    logger.flush()


def evaluate(params, model, dataset, logger, restore_checkpoint=True):
    if params.fold is None:
        print("No fold specified for evaluation. Please use --fold [int] to select a fold.")
    ce_loss = tf.keras.metrics.Mean(name='ce_loss')
    f1_loss = tf.keras.metrics.Mean(name='dice_loss')
    if params.model_dir and restore_checkpoint:
        prec = 'amp' if params.use_amp else 'fp32'
        if params.use_savedmodel:
            model = tf.keras.models.load_model(os.path.join(params.model_dir, f'saved_model_{prec}'))
        elif params.use_tftrt:
            model = TFTRTModel(model_dir=params.model_dir, precision=prec)
        else:
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(tf.train.latest_checkpoint(params.model_dir)).expect_partial()

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
    prec = 'amp' if params.use_amp else 'fp32'
    if params.model_dir:
        if params.use_savedmodel:
            model = tf.keras.models.load_model(os.path.join(params.model_dir, f'saved_model_{prec}'))
        elif params.use_tftrt:
            model = TFTRTModel(model_dir=params.model_dir, precision=prec)
        else:
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(tf.train.latest_checkpoint(params.model_dir)).expect_partial()

    @tf.function
    def prediction_step(features):
        return tf.nn.softmax(model(features, training=False), axis=-1)

    if params.benchmark:
        assert params.max_steps > params.warmup_steps, \
            "max_steps value has to be greater than warmup_steps"
        timestamps = []
        for iteration, images in enumerate(dataset.test_fn(count=None, drop_remainder=True)):
            prediction_step(images)
            if iteration > params.warmup_steps:
                timestamps.append(time())
            if iteration >= params.max_steps:
                break

        deltas = np.array([timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)])
        stats = process_performance_stats(deltas, params.batch_size, mode="test")
        logger.log(step=(), data=stats)
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

        print("Predictions saved at {}".format(output_dir))
    logger.flush()
