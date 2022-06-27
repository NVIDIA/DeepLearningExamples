import logging
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf

from mrcnn_tf2.utils.keras import KerasCallback

CONFIDENCE_INTERVAL_Z = {
    80.0: 1.282,
    85.0: 1.440,
    90.0: 1.645,
    95.0: 1.960,
    99.0: 2.576,
    99.5: 2.807,
    99.9: 3.291,
}


class DLLoggerMetricsCallback(KerasCallback):
    """
    Keras callback that saves metrics using DLLogger.
    """

    def __init__(self, dllogger, log_every=10, log_learning_rate=False):
        """
        Args:
            dllogger (DLLogger): DLLogger instance.
            log_every (int): Logging interval.
            log_learning_rate (bool): When set to true adds learning rate to metrics.
                Cannot be used with AMP enabled as the used hack fails with AMP.
        """
        super().__init__()
        self._dllogger = dllogger
        self._log_every = log_every
        self._log_learning_rate = log_learning_rate

        if not isinstance(log_every, dict):
            self._log_every = defaultdict(lambda: log_every)

        self._dllogger.metadata('loss', {'unit': None})
        self._dllogger.metadata('AP', {'unit': None})
        self._dllogger.metadata('mask_AP', {'unit': None})

        logging.getLogger('hooks').info('Created metrics logging hook')

    def on_any_batch_end(self, mode, epoch, batch, logs):
        if (batch + 1) % self._log_every[mode] != 0:
            return

        step = (None if epoch is None else epoch + 1, batch + 1)
        self._log_metrics(mode, logs, step=step)

    def on_any_epoch_end(self, mode, epoch, logs):
        step = (None if epoch is None else epoch + 1, )
        self._log_metrics(mode, logs, step=step)

    def on_any_end(self, mode, logs):
        self._log_metrics(mode, logs)

    def _log_metrics(self, mode, logs, step=tuple()):
        logs = logs or {}

        # remove outputs that are not in fact a metric
        logs.pop('outputs', None)

        if mode == 'train' and self._log_learning_rate:
            logs['learning_rate'] = float(self.model.optimizer._decayed_lr(tf.float32))

        # no point in logging with empty data
        if not logs:
            return

        self._dllogger.log(step=step, data=logs)


class DLLoggerPerfCallback(KerasCallback):
    """
    Keras callback that measures performance and logs it using DLLogger.
    """

    def __init__(self, dllogger, batch_sizes, warmup_steps=0, log_every=None):
        super().__init__()
        self._dllogger = dllogger
        self._batch_sizes = batch_sizes
        self._warmup_steps = warmup_steps
        self._log_every = log_every

        if not isinstance(batch_sizes, dict):
            self._batch_sizes = defaultdict(lambda: batch_sizes)
        if not isinstance(warmup_steps, dict):
            self._warmup_steps = defaultdict(lambda: warmup_steps)
        if not isinstance(log_every, dict):
            self._log_every = defaultdict(lambda: log_every)

        self._deltas = {}
        self._batch_timestamps = {}
        self._start_timestamps = {}

        for mode in ['train', 'test', 'predict']:
            self._dllogger.metadata(f'{mode}_throughput', {'unit': 'images/s'})
            self._dllogger.metadata(f'{mode}_latency', {'unit': 's'})
            self._dllogger.metadata(f'{mode}_latency_90', {'unit': 's'})
            self._dllogger.metadata(f'{mode}_latency_95', {'unit': 's'})
            self._dllogger.metadata(f'{mode}_latency_99', {'unit': 's'})
            self._dllogger.metadata(f'{mode}_time', {'unit': 's'})

        self._logger = logging.getLogger('hooks')
        self._logger.info('Created perf logging hooks')

    def on_any_begin(self, mode, logs):
        self._deltas[mode] = []
        self._start_timestamps[mode] = time.time()

    def on_any_batch_begin(self, mode, epoch, batch, logs):
        self._batch_timestamps[mode] = time.time()

    def on_any_batch_end(self, mode, epoch, batch, logs):
        self._deltas[mode].append(time.time() - self._batch_timestamps[mode])

        if self._log_every[mode] and (batch + 1) % self._log_every[mode] != 0:
            return

        step = (None if epoch is None else epoch + 1, batch + 1)
        self._log_perf(self._deltas[mode][-self._log_every[mode]:], mode, step=step)

    def on_any_end(self, mode, logs):
        if len(self._deltas[mode]) > self._warmup_steps[mode]:
            self._log_perf(self._deltas[mode][self._warmup_steps[mode]:], mode)
        else:
            self._logger.warning(
                f'Number of all {mode} steps was smaller then number of warm up steps, '
                f'no stats were collected.'
            )

    def _log_perf(self, deltas, mode, step=tuple()):
        deltas = np.array(deltas)
        self._dllogger.log(
            step=step,
            data={
                f'{mode}_throughput': self._calculate_throughput(deltas, self._batch_sizes[mode]),
                f'{mode}_latency': self._calculate_latency(deltas),
                f'{mode}_latency_90': self._calculate_latency_confidence(deltas, 90.0),
                f'{mode}_latency_95': self._calculate_latency_confidence(deltas, 95.0),
                f'{mode}_latency_99': self._calculate_latency_confidence(deltas, 99.0),
                f'{mode}_time': self._calculate_total_time(self._start_timestamps[mode], time.time())
            }
        )

    @staticmethod
    def _calculate_throughput(deltas, batch_size):
        return batch_size / deltas.mean()

    @staticmethod
    def _calculate_latency(deltas):
        return deltas.mean()

    @staticmethod
    def _calculate_latency_confidence(deltas, confidence_interval):
        mean = deltas.mean()
        std = deltas.std()
        n = len(deltas)
        z = CONFIDENCE_INTERVAL_Z[confidence_interval]
        return mean + (z * std / np.sqrt(n))

    @staticmethod
    def _calculate_total_time(start_time, end_time):
        return end_time - start_time


class PretrainedWeightsLoadingCallback(KerasCallback):
    """
    Loads pretrained weights from given checkpoint after first batch.
    """

    def __init__(self, checkpoint_path, mapping=None):
        """
        Args:
            checkpoint_path: Path to the checkpoint, as accepted by `tf.train.load_checkpoint()`
            mapping: Callable that takes name of a variable and returns name of a corresponding
                entry in the checkpoint.
        """
        super().__init__()
        self._checkpoint_path = checkpoint_path
        self._mapping = mapping or (lambda x: x)

        self._loaded = False

        self._logger = logging.getLogger('hooks')
        self._logger.info(f'Created pretrained backbone weights loading hook that loads from {checkpoint_path}')

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        if not self._loaded:
            self.load_weights()
            self._loaded = True

    def load_weights(self):
        reader = tf.train.load_checkpoint(self._checkpoint_path)

        variable_mapping = {
            self._mapping(var.name): var
            for var in self.model.variables
            if reader.has_tensor(self._mapping(var.name))
        }

        for cp_name, var in variable_mapping.items():
            var.assign(reader.get_tensor(cp_name))
            self._logger.debug(f'Assigned "{cp_name}" from checkpoint to "{var.name}"')

        self._logger.info(f'Loaded {len(variable_mapping)} pretrained backbone variables')
