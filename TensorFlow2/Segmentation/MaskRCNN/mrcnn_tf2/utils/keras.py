import tensorflow as tf


class KerasCallback(tf.keras.callbacks.Callback):
    """ Utility class that simplifies usage of Keras callback across different modes. """

    def __init__(self):
        super().__init__()
        self._current_epoch = None

    def on_any_begin(self, mode, logs):
        pass

    def on_any_end(self, mode, logs):
        pass

    def on_any_epoch_begin(self, mode, epoch, logs):
        pass

    def on_any_epoch_end(self, mode, epoch, logs):
        pass

    def on_any_batch_begin(self, mode, epoch, batch, logs):
        pass

    def on_any_batch_end(self, mode, epoch, batch, logs):
        pass

    def on_train_begin(self, logs=None):
        self.on_any_begin('train', logs)

    def on_test_begin(self, logs=None):
        self.on_any_begin('test', logs)

    def on_predict_begin(self, logs=None):
        self.on_any_begin('predict', logs)

    def on_train_end(self, logs=None):
        self.on_any_end('train', logs)

    def on_test_end(self, logs=None):
        self.on_any_end('test', logs)

    def on_predict_end(self, logs=None):
        self.on_any_end('predict', logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch
        self.on_any_epoch_begin('train', epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.on_any_epoch_end('train', epoch, logs)
        self._current_epoch = None

    def on_train_batch_begin(self, batch, logs=None):
        self.on_any_batch_begin('train', self._current_epoch, batch, logs)

    def on_test_batch_begin(self, batch, logs=None):
        self.on_any_batch_begin('test', None, batch, logs)

    def on_predict_batch_begin(self, batch, logs=None):
        self.on_any_batch_begin('predict', None, batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self.on_any_batch_end('train', self._current_epoch, batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        self.on_any_batch_end('test', None, batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        self.on_any_batch_end('predict', None, batch, logs)
