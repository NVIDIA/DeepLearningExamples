# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import horovod.tensorflow as hvd
import tensorflow as tf

from trainer.utils.benchmark import ThroughputCalculator
from trainer.utils.evaluator import Evaluator
from trainer.utils.schedulers import LearningRateScheduler
from trainer.utils.trainer import Trainer
from data.outbrain.defaults import MAP_FEATURE_CHANNEL, MULTIHOT_CHANNEL


def run(args, model, config):
    train_dataset = config["train_dataset"]
    eval_dataset = config["eval_dataset"]
    feature_spec = config["feature_spec"]
    multihot_features = feature_spec.get_names_by_channel(MULTIHOT_CHANNEL)
    multihot_hotness_dict = feature_spec.get_multihot_hotnesses(multihot_features)
    steps_per_epoch = len(train_dataset)
    steps_per_epoch = min(hvd.allgather(tf.constant([steps_per_epoch], dtype=tf.int32)))
    steps_per_epoch = steps_per_epoch.numpy()

    steps = int(steps_per_epoch * args.num_epochs)
    deep_optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=args.deep_learning_rate, rho=0.5
    )

    wide_optimizer = tf.keras.optimizers.Ftrl(learning_rate=args.linear_learning_rate)

    if not args.cpu:
        deep_optimizer = hvd.DistributedOptimizer(
            deep_optimizer, compression=hvd.Compression.fp16
        )
        wide_optimizer = hvd.DistributedOptimizer(
            wide_optimizer, compression=hvd.Compression.fp16
        )

    if args.amp:
        deep_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            deep_optimizer, dynamic=True
        )
        wide_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            wide_optimizer, dynamic=True
        )

    scheduler = LearningRateScheduler(
        args=args, steps_per_epoch=steps_per_epoch, optimizer=deep_optimizer
    )

    throughput_calculator = ThroughputCalculator(args)
    compiled_loss = tf.keras.losses.BinaryCrossentropy()

    maybe_map_column = None
    if args.map_calculation_enabled:
        maybe_map_column = feature_spec.get_names_by_channel(MAP_FEATURE_CHANNEL)[0]

    evaluator = Evaluator(
        model=model,
        throughput_calculator=throughput_calculator,
        eval_dataset=eval_dataset,
        compiled_loss=compiled_loss,
        args=args,
        maybe_map_column=maybe_map_column,
        multihot_hotnesses_dict=multihot_hotness_dict,
        num_auc_thresholds=args.num_auc_thresholds
    )

    trainer = Trainer(
        model=model,
        scheduler=scheduler,
        deep_optimizer=deep_optimizer,
        wide_optimizer=wide_optimizer,
        throughput_calculator=throughput_calculator,
        compiled_loss=compiled_loss,
        steps=steps,
        args=args,
        train_dataset=train_dataset,
        evaluator=evaluator,
        multihot_hotnesses_dict=multihot_hotness_dict
    )

    trainer.maybe_restore_checkpoint()

    # Wrap datasets with .epochs(n) method to speed up data loading
    current_epoch = trainer.current_epoch
    trainer.prepare_dataset(current_epoch)
    evaluator.prepare_dataset(current_epoch)

    # Update max_steps to make sure that all workers finish training at the same time
    max_training_steps = len(trainer.train_dataset)
    max_training_steps = min(
        hvd.allgather(tf.constant([max_training_steps], dtype=tf.int32))
    )
    max_training_steps = int(max_training_steps.numpy())
    trainer.max_steps = max_training_steps

    if args.evaluate:
        evaluator.eval(trainer.current_step_var)
    else:
        trainer.run_loop()
