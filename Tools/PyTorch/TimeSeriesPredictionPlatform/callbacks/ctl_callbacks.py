# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0
import time

import dllogger

from callbacks.callbacks import Callback, CallbackContainer
from distributed_utils import is_main_process
from training.utils import round_dict
from training.checkpoint_utils import save_checkpoint


class CTLCallbackContainer(CallbackContainer):
    """
    Base class for CTLTrainer callbacks storage.
    """

    def __init__(self, trainer, callbacks):
        self.callbacks = callbacks
        self.trainer = trainer
        self._init_trainers()
        self.logs = {}
        super().__init__()

    def _init_trainers(self):
        for callback in self.callbacks:
            callback.trainer = self.trainer

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_valid_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            callback.on_valid_begin(epoch, logs)

    def on_valid_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            callback.on_valid_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_evaluate_end(self, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            callback.on_evaluate_end(logs)

    def on_evaluate_begin(self, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            callback.on_evaluate_begin(logs)


class CTLCallback(Callback):
    """
    Base class for building new CTLTrainer callbacks.
    """

    def __init__(self):
        self.trainer = None
        super().__init__()

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, trainer):
        self._trainer = trainer

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_valid_begin(self, epoch, logs=None):
        pass

    def on_valid_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_evaluate_begin(self, logs=None):
        pass

    def on_evaluate_end(self, logs=None):
        pass


class LoggingCallback(CTLCallback):
    def on_train_begin(self, logs=None):
        self.trainer.logger.log(
            step='event',
            data={"String": "Training with {} epochs".format(self.trainer.config.get("num_epochs", 1))},
            verbosity=dllogger.Verbosity.DEFAULT,
        )

    def on_train_end(self, logs=None):
        self.trainer.logger.log(step='event', data={"String": "Training Stopped"}, verbosity=dllogger.Verbosity.DEFAULT)

    def on_epoch_begin(self, epoch, logs=None):
        self.trainer.logger.log(step='event', data={"String": "Epoch {}".format(epoch)}, verbosity=dllogger.Verbosity.DEFAULT)

    def on_batch_end(self, batch, logs=None):
        if self.trainer.config.log_interval > 0 and self.trainer.global_step % self.trainer.config.log_interval == 0:
            self.trainer.logger.flush()

    def on_valid_begin(self, epoch, logs=None):
        self.trainer.logger.log(
            step='event', data={"String": "Calculating Validation Metrics"}, verbosity=dllogger.Verbosity.DEFAULT
        )

    def on_valid_end(self, epoch, logs=None):
        self.trainer.logger.log(
            step='event',
            data={"String": "Epoch {} Validation Metrics: {}".format(epoch, round_dict(logs))},
            verbosity=dllogger.Verbosity.DEFAULT,
        )

    def on_epoch_end(self, epoch, logs=None):
        self.trainer.logger.flush()

    def on_evaluate_begin(self, logs=None):
        self.trainer.logger.log(
            step='event', data={"String": "Beginning Metric Evaluation"}, verbosity=dllogger.Verbosity.DEFAULT
        )

    def on_evaluate_end(self, logs=None):
        self.trainer.logger.log(
            step='event', data={"String": "Evaluation Metrics: {}".format(round_dict(logs))}, verbosity=dllogger.Verbosity.DEFAULT
        )
        self.trainer.logger.log(step=[], data=logs, verbosity=dllogger.Verbosity.DEFAULT)


class EarlyStopping(CTLCallback):
    def __init__(self, metric="val_loss", min_delta=0, patience=5, max_divergence=None, divergence_patience=1):
        self.metric = metric
        self.min_delta = min_delta
        self.patience = patience
        self.max_divergence = max_divergence
        self.divergence_patience = divergence_patience
        self.divergence_stopped_epochs = 0
        self.stopped_epochs = 0
        self.best_loss = None
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        epoch_loss = logs.get(self.metric, None)
        if epoch_loss is None:
            return

        if self.best_loss is None:
            self.best_loss = epoch_loss
            return

        if self.max_divergence and ((epoch_loss - self.best_loss) > self.max_divergence):
            self.divergence_stopped_epochs += 1
            self.stopped_epochs += 1
            if self.divergence_stopped_epochs >= self.divergence_patience:
                self.trainer._stop_training = True
                self.trainer.logger.log(
                    step='event', data={"String": f"Applying early stopping as divergence threshold reached"}, verbosity=dllogger.Verbosity.DEFAULT
                )
        elif (epoch_loss + self.min_delta) < self.best_loss:
            self.best_loss = epoch_loss
            self.stopped_epochs = 0
            self.divergence_stopped_epochs = 0
        else:
            self.stopped_epochs += 1
            self.divergence_stopped_epochs = 0

        if self.stopped_epochs >= self.patience:
            self.trainer._stop_training = True
            self.trainer.logger.log(
                step='event', data={"String": f"Applying early stopping"}, verbosity=dllogger.Verbosity.DEFAULT
            )


class SaveBestCheckpoint(CTLCallback):
    def __init__(self, metric="val_loss"):
        self.metric = metric
        self.best_loss = None
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        epoch_loss = logs.get(self.metric, None)
        if epoch_loss is None:
            return

        if self.best_loss is None or epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            if is_main_process():
                save_checkpoint(self.trainer, checkpoint_dir=self.trainer.log_path, filename="best_checkpoint.zip")


class SaveCheckpoint(CTLCallback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if is_main_process():
            save_checkpoint(self.trainer, checkpoint_dir=self.trainer.log_path, filename="last_checkpoint.zip")


class MeanAccumulator:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def consume(self, value):
        self.sum += value
        self.count += 1

    @property
    def value(self):
        if self.count == 0:
            return 0
        return self.sum / self.count


class ThroughputBenchmark(CTLCallback):
    def __init__(self, warmup_epochs=0):
        self.warmup_epochs = warmup_epochs
        self.train_throughput = MeanAccumulator()
        self.valid_throughput = MeanAccumulator()
        self.epoch_train_start = None
        self.epoch_train_end = None
        super().__init__()

    def on_train_end(self, logs=None):
        if self.train_throughput.value > 0:
            logs["Train it/s"] = self.train_throughput.value
            logs["Valid it/s"] = self.valid_throughput.value

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_train_start = time.time()

    def on_valid_end(self, epoch, logs=None):
        if epoch >= self.warmup_epochs:
            train_epoch_time = self.epoch_train_end - self.epoch_train_start
            valid_epoch_time = time.time() - self.epoch_train_end
            train_iter_per_sec = self.trainer.train_dataset_len / train_epoch_time
            valid_iter_per_sec = self.trainer.valid_dataset_len / valid_epoch_time

            logs["Train epoch it/s"] = train_iter_per_sec
            logs["Valid epoch it/s"] = valid_iter_per_sec

            self.train_throughput.consume(train_iter_per_sec)
            self.valid_throughput.consume(valid_iter_per_sec)

    def on_valid_begin(self, batch, logs=None):
        self.epoch_train_end = time.time()
