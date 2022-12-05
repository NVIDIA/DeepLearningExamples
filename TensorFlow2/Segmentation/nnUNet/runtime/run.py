# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from runtime.checkpoint import CheckpointManager
from runtime.losses import DiceCELoss, WeightDecay
from runtime.metrics import Dice, MetricAggregator, make_class_logger_metrics
from runtime.utils import is_main_process, make_empty_dir, progress_bar


def update_best_metrics(old, new, start_time, iteration, watch_metric=None):
    did_change = False
    for metric, value in new.items():
        if metric not in old or old[metric]["value"] < value:
            old[metric] = {"value": value, "timestamp": time.time() - start_time, "iter": int(iteration)}
            if watch_metric == metric:
                did_change = True
    return did_change


def get_scheduler(args, total_steps):
    scheduler = {
        "poly": tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=args.learning_rate,
            end_learning_rate=args.end_learning_rate,
            decay_steps=total_steps,
            power=0.9,
        ),
        "cosine": tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.learning_rate, decay_steps=total_steps
        ),
        "cosine_annealing": tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=args.learning_rate,
            first_decay_steps=args.cosine_annealing_first_cycle_steps,
            alpha=0.1,
        ),
        "none": args.learning_rate,
    }[args.scheduler.lower()]
    return scheduler


def get_optimizer(args, scheduler):
    optimizer = {
        "sgd": tf.keras.optimizers.SGD(learning_rate=scheduler, momentum=args.momentum),
        "adam": tf.keras.optimizers.Adam(learning_rate=scheduler),
        "radam": tfa.optimizers.RectifiedAdam(learning_rate=scheduler),
    }[args.optimizer.lower()]
    if args.lookahead:
        optimizer = tfa.optimizers.Lookahead(optimizer)
    if args.amp:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
    return optimizer


def get_epoch_size(args, batch_size, dataset_size):
    if args.steps_per_epoch:
        return args.steps_per_epoch
    div = args.gpus * (batch_size if args.dim == 3 else args.nvol)
    return (dataset_size + div - 1) // div


def process_performance_stats(deltas, batch_size, mode):
    deltas_ms = 1000 * np.array(deltas)
    throughput_imgps = 1000.0 * batch_size / deltas_ms.mean()
    stats = {f"throughput_{mode}": throughput_imgps, f"latency_{mode}_mean": deltas_ms.mean()}
    for level in [90, 95, 99]:
        stats.update({f"latency_{mode}_{level}": np.percentile(deltas_ms, level)})

    return stats


def benchmark(args, step_fn, data, steps, warmup_steps, logger, mode="train"):
    assert steps > warmup_steps, "Number of benchmarked steps has to be greater then number of warmup steps"
    deltas = []
    wrapped_data = progress_bar(
        enumerate(data),
        quiet=args.quiet,
        desc=f"Benchmark ({mode})",
        unit="step",
        postfix={"phase": "warmup"},
        total=steps,
    )
    start = time.perf_counter()
    for step, (images, labels) in wrapped_data:
        output_map = step_fn(images, labels, warmup_batch=step == 0)
        if step >= warmup_steps:
            deltas.append(time.perf_counter() - start)
            if step == warmup_steps and is_main_process() and not args.quiet:
                wrapped_data.set_postfix(phase="benchmark")
        start = time.perf_counter()
        if step >= steps:
            break

    stats = process_performance_stats(deltas, args.gpus * args.batch_size, mode=mode)
    logger.log_metrics(stats)


def train(args, model, dataset, logger):
    train_data = dataset.train_dataset()

    epochs = args.epochs
    batch_size = args.batch_size if args.dim == 3 else args.nvol
    steps_per_epoch = get_epoch_size(args, batch_size, dataset.train_size())
    total_steps = epochs * steps_per_epoch

    scheduler = get_scheduler(args, total_steps)
    optimizer = get_optimizer(args, scheduler)
    loss_fn = DiceCELoss(
        y_one_hot=True,
        reduce_batch=args.reduce_batch,
        include_background=args.include_background,
    )
    wdecay = WeightDecay(factor=args.weight_decay)
    tstep = tf.Variable(0)

    @tf.function
    def train_step_fn(features, labels, warmup_batch=False):
        features, labels = model.adjust_batch(features, labels)
        with tf.GradientTape() as tape:
            output_map = model(features)
            dice_loss = model.compute_loss(loss_fn, labels, output_map)
            loss = dice_loss + wdecay(model)
            if args.amp:
                loss = optimizer.get_scaled_loss(loss)
        tape = hvd.DistributedGradientTape(tape)
        gradients = tape.gradient(loss, model.trainable_variables)
        if args.amp:
            gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Note: broadcast should be done after the first gradient step to ensure optimizer initialization.
        if warmup_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        return dice_loss

    dice_metrics = MetricAggregator(name="dice")
    checkpoint = CheckpointManager(
        args.ckpt_dir,
        strategy=args.ckpt_strategy,
        resume_training=args.resume_training,
        variables={"model": model, "optimizer": optimizer, "step": tstep, **dice_metrics.checkpoint_metrics()},
    )

    if args.benchmark:
        benchmark(args, train_step_fn, train_data, args.bench_steps, args.warmup_steps, logger)
    else:
        wrapped_data = progress_bar(
            train_data,
            quiet=args.quiet,
            desc="Train",
            postfix={"epoch": 1},
            unit="step",
            total=total_steps - int(tstep),
        )
        start_time = time.time()
        total_train_loss, dice_score = 0.0, 0.0
        for images, labels in wrapped_data:
            if tstep >= total_steps:
                break
            tstep.assign_add(1)
            loss = train_step_fn(images, labels, warmup_batch=tstep == 1)
            total_train_loss += float(loss)
            lr = scheduler(tstep) if callable(scheduler) else scheduler
            metrics = {"loss": float(loss), "learning_rate": float(lr)}
            if tstep % steps_per_epoch == 0:
                epoch = int(tstep // steps_per_epoch)
                if epoch > args.skip_eval:
                    dice = evaluate(args, model, dataset, logger)
                    dice_score = tf.reduce_mean(dice[1:])
                    did_improve = dice_metrics.update(dice_score)
                    metrics = dice_metrics.logger_metrics()
                    metrics.update(make_class_logger_metrics(dice))
                    if did_improve:
                        metrics["time_to_train"] = time.time() - start_time
                    logger.log_metrics(metrics=metrics, step=int(tstep))
                    checkpoint.update(float(dice_score))
                    logger.flush()
                else:
                    checkpoint.update(None)
                if is_main_process() and not args.quiet:
                    wrapped_data.set_postfix(epoch=epoch + 1)
            elif tstep % steps_per_epoch == 0:
                total_train_loss = 0.0

        metrics = {
            "train_loss": round(total_train_loss / steps_per_epoch, 5),
            "val_loss": round(1 - float(dice_score), 5),
            "dice": round(float(dice_metrics.metrics["max"].result()), 5),
        }
        logger.log_metrics(metrics=metrics)
        logger.flush()


def evaluate(args, model, dataset, logger):
    dice = Dice(n_class=model.n_class)

    data_size = dataset.val_size()
    wrapped_data = progress_bar(
        enumerate(dataset.val_dataset()),
        quiet=args.quiet,
        desc="Validation",
        unit="step",
        total=data_size,
    )
    for i, (features, labels) in wrapped_data:
        if args.dim == 2:
            features, labels = features[0], labels[0]
        output_map = model.inference(features)

        dice.update_state(output_map, labels)
        if i + 1 == data_size:
            break
    result = dice.result()
    if args.exec_mode == "evaluate":
        metrics = {
            "eval_dice": float(tf.reduce_mean(result)),
            "eval_dice_nobg": float(tf.reduce_mean(result[1:])),
        }
        logger.log_metrics(metrics)
    return result


def predict(args, model, dataset, logger):
    if args.benchmark:

        @tf.function
        def predict_bench_fn(features, labels, warmup_batch):
            if args.dim == 2:
                features = features[0]
            output_map = model(features, training=False)
            return output_map

        benchmark(
            args,
            predict_bench_fn,
            dataset.test_dataset(),
            args.bench_steps,
            args.warmup_steps,
            logger,
            mode="predict",
        )
    else:
        if args.save_preds:
            prec = "amp" if args.amp else "fp32"
            dir_name = f"preds_task_{args.task}_dim_{args.dim}_fold_{args.fold}_{prec}"
            if args.tta:
                dir_name += "_tta"
            save_dir = args.results / dir_name
            make_empty_dir(save_dir)

        data_size = dataset.test_size()
        wrapped_data = progress_bar(
            enumerate(dataset.test_dataset()),
            quiet=args.quiet,
            desc="Predict",
            unit="step",
            total=data_size,
        )

        for i, (images, meta) in wrapped_data:
            features, _ = model.adjust_batch(images, None)
            pred = model.inference(features, training=False)
            if args.save_preds:
                model.save_pred(pred, meta, idx=i, data_module=dataset, save_dir=save_dir)
            if i + 1 == data_size:
                break


def export_model(args, model):
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(args.ckpt_dir)).expect_partial()

    input_shape = [1, *model.patch_size, model.n_class]
    dummy_input = tf.constant(tf.zeros(input_shape, dtype=tf.float32))
    _ = model(dummy_input, training=False)

    prec = "amp" if args.amp else "fp32"
    path = str(args.results / f"saved_model_task_{args.task}_dim_{args.dim}_{prec}")
    tf.keras.models.save_model(model, str(path))

    trt_prec = trt.TrtPrecisionMode.FP32 if prec == "fp32" else trt.TrtPrecisionMode.FP16
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=path,
        conversion_params=trt.TrtConversionParams(precision_mode=trt_prec),
    )
    converter.convert()

    trt_path = str(args.results / f"trt_saved_model_task_{args.task}_dim_{args.dim}_{prec}")
    converter.save(trt_path)
