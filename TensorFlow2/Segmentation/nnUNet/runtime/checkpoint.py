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

from pathlib import Path
from time import time

import tensorflow as tf
from models.nn_unet import NNUnet

from runtime.utils import rank_zero_only


class CheckpointManager:
    def __init__(self, ckpt_dir, strategy, variables, step_counter=None, resume_training=False):
        self.dir = Path(ckpt_dir)
        self.strategy = strategy
        self.vars = variables
        self.ckpt = tf.train.Checkpoint(**variables)
        self.creation_time = time()
        self.latest_save_time = time()

        if "last" in strategy:
            self.last_manager = tf.train.CheckpointManager(
                self.ckpt, self.dir, max_to_keep=1, checkpoint_name="ckpt-last", step_counter=step_counter
            )
            if resume_training:
                self.ckpt.restore(self.last_manager.latest_checkpoint)

        if "best" in strategy:
            self.best_manager = tf.train.CheckpointManager(
                self.ckpt, self.dir / "best", max_to_keep=1, checkpoint_name="ckpt-best", step_counter=step_counter
            )
            self.best_metric = None

    @rank_zero_only
    def update(self, metric_value=None):
        if "last" in self.strategy:
            self.last_manager.save()
        if (
            metric_value is not None
            and "best" in self.strategy
            and (self.best_metric is None or self.best_metric < metric_value)
        ):
            self.latest_save_time = time()
            if self.best_metric is not None:
                print(
                    f"({int(self.latest_save_time - self.creation_time)}s)",
                    f"New best metric value achieved ({float(metric_value):.4f} > {float(self.best_metric):.4f}).",
                )
            print("Saving new checkpoint.")
            self.best_metric = metric_value
            self.best_manager.save()

    def load_best(self):
        self.ckpt.restore(self.best_manager.latest_checkpoint)
        return self.best_metric, int(self.latest_save_time - self.creation_time)


def load_model(args):
    if args.saved_model_dir is not None:
        print(f"Loading SavedModel from {str(args.saved_model_dir)}")
        model = tf.saved_model.load(str(args.saved_model_dir))
        model = NNUnet(args, loaded_model=model)
    else:
        if not (Path(args.ckpt_dir).is_dir() and (Path(args.ckpt_dir) / "checkpoint").exists()):
            raise ValueError(f"Could not find checkpoint directory {args.ckpt_dir}")
        model = NNUnet(args)
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(tf.train.latest_checkpoint(args.ckpt_dir)).expect_partial()
    return model
