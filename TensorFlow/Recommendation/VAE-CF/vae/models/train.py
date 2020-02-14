# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
import scipy.sparse as sparse
import tensorflow as tf
import numpy as np
import time
import logging
import dllogger

from sklearn.preprocessing import normalize
from collections import defaultdict

from vae.models.vae import _VAEGraph, TRAINING, QUERY, VALIDATION
from vae.utils.round import round_8

LOG = logging.getLogger("VAE")


class VAE:
    def __init__(self,
                 train_data,
                 encoder_dims,
                 decoder_dims=None,
                 batch_size_train=500,
                 batch_size_validation=2000,
                 lam=3e-2,
                 lr=1e-3,
                 beta1=0.9,
                 beta2=0.999,
                 total_anneal_steps=200000,
                 anneal_cap=0.2,
                 xla=True,
                 activation='tanh',
                 checkpoint_dir=None,
                 trace=False,
                 top_results=100):

        if decoder_dims is None:
            decoder_dims = encoder_dims[::-1]
        for i in encoder_dims + decoder_dims + [batch_size_train, batch_size_validation]:
            if i != round_8(i):
                raise ValueError("all dims and batch sizes should be divisible by 8")

        self.metrics_history = None
        self.batch_size_train = batch_size_train
        self.batch_size_validation = batch_size_validation
        self.lam = lam
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.xla = xla
        self.total_anneal_steps = total_anneal_steps
        self.anneal_cap = anneal_cap
        self.activation = activation
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.trace = trace
        self.top_results = top_results
        self.checkpoint_dir = checkpoint_dir if hvd.rank() == 0 else None
        self._create_dataset(train_data,
                             batch_size_train,
                             encoder_dims)
        self._setup_model()

        self.metrics_history = defaultdict(lambda: [])
        self.time_elapsed_training_history = []
        self.time_elapsed_validation_history = []
        self.training_throughputs = []
        self.inference_throughputs = []


    def _create_dataset(self, train_data, batch_size_train, encoder_dims):
        generator, self.n_batch_per_train = self.batch_iterator(train_data,
                                                                None,
                                                                batch_size_train,
                                                                thread_idx=hvd.rank(),
                                                                thread_num=hvd.size())
        dataset = tf.data.Dataset \
            .from_generator(generator, output_types=(tf.int64, tf.float32)) \
            .map(lambda i, v: tf.SparseTensor(i, v, (batch_size_train, encoder_dims[0]))) \
            .prefetch(10)
        self.iter = dataset.make_initializable_iterator()
        self.inputs_train = self.iter.get_next()

    def _setup_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())

        hooks = [hvd.BroadcastGlobalVariablesHook(0)]
        if self.trace:
            hooks.append(tf.train.ProfilerHook(save_steps=1, output_dir='.'))

        if self.xla:
            LOG.info('Enabling XLA')
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        else:
            LOG.info('XLA disabled')

        self._build_graph()
        self.session = tf.train.MonitoredTrainingSession(config=config,
                                                         checkpoint_dir=self.checkpoint_dir,
                                                         save_checkpoint_secs=10,
                                                         hooks=hooks)

    def _build_optimizer(self, loss):
        optimizer= tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1, beta2=self.beta2)
        return hvd.DistributedOptimizer(optimizer).minimize(
            loss, global_step=tf.train.get_or_create_global_step())

    def close_session(self):
        if self.session is not None:
            self.session.close()

    def batch_iterator(self, data_input, data_true=None, batch_size=500, thread_idx=0, thread_num=1):
        training = data_true is None

        data_input = normalize(data_input)
        indices = np.arange(data_input.shape[0])

        global_batch_size = batch_size * hvd.size()

        if training:
            # crop the data so that each gpu has the same number of batches
            stop = data_input.shape[0] // global_batch_size * global_batch_size
            LOG.info('Cropping each epoch from: {} to {} samples'.format(data_input.shape[0], stop))
        else:
            stop = data_input.shape[0]

        def generator():
            data_in = data_input
            epoch = 0
            while True:
                if training:
                    # deterministic shuffle necessary for multigpu
                    np.random.seed(epoch)
                    np.random.shuffle(indices)
                    data_in = data_in[indices]

                for st_idx in range(thread_idx * batch_size, stop, thread_num * batch_size):
                    batch = data_in[st_idx:st_idx + batch_size].copy()
                    batch = batch.tocoo()
                    idxs = np.stack([batch.row, batch.col], axis=1)
                    vals = batch.data
                    if training:
                        np.random.seed(epoch * thread_num + thread_idx)
                        nnz = vals.shape[0]

                        # dropout with keep_prob=0.5
                        vals *= (2 * np.random.randint(2, size=nnz))
                        yield (idxs, vals)
                    else:
                        yield idxs, vals, data_true[st_idx:st_idx + batch_size]
                if not training:
                    break
                epoch += 1

        be = thread_idx * batch_size
        st = thread_num * batch_size
        return generator, int(np.ceil((stop - be) / st))

    def _build_graph(self):
        self.vae = _VAEGraph(self.encoder_dims, self.decoder_dims, self.activation)

        self.inputs_validation = tf.sparse.placeholder(
            dtype=tf.float32,
            shape=np.array([self.batch_size_validation, self.vae.input_dim], dtype=np.int32))
        self.inputs_query = tf.sparse.placeholder(
            dtype=tf.float32,
            shape=np.array([1, self.vae.input_dim], dtype=np.int32))

        self.top_k_validation = self._gen_handlers(mode=VALIDATION)
        self.logits_train, self.loss_train, self.optimizer = self._gen_handlers(mode=TRAINING)
        self.top_k_query = self._gen_handlers(mode=QUERY)

        global_step = tf.train.get_or_create_global_step()
        self.increment_global_step = tf.assign(global_step, global_step + 1)

    def _gen_handlers(self, mode):
        # model input
        if mode is TRAINING:
            inputs = self.inputs_train
        elif mode is VALIDATION:
            inputs = self.inputs_validation
        elif mode is QUERY:
            inputs = self.inputs_query
        else:
            assert False

        if mode is TRAINING:
            batch_size = self.batch_size_train
        elif mode is VALIDATION:
            batch_size = self.batch_size_validation
        elif mode is QUERY:
            batch_size = 1
        else:
            assert False

        # model output
        logits, latent_mean, latent_log_var = self.vae(inputs, mode=mode)
        if mode in [VALIDATION, QUERY]:
            mask = tf.ones_like(inputs.values) * (-np.inf)
            logits = tf.tensor_scatter_nd_update(logits, inputs.indices, mask)
            top_k_values, top_k_indices = tf.math.top_k(logits, sorted=True, k=self.top_results)
            return top_k_indices

        softmax = tf.nn.log_softmax(logits)

        anneal = tf.math.minimum(
            tf.cast(tf.train.get_or_create_global_step(), tf.float32) /
            self.total_anneal_steps, self.anneal_cap)

        # KL divergence
        KL = tf.reduce_mean(
            tf.reduce_sum(
                (-latent_log_var + tf.exp(latent_log_var) + latent_mean ** 2 - 1)
                / 2,
                axis=1))

        # per-user average negative log-likelihood part of loss
        ll_loss = -tf.reduce_sum(tf.gather_nd(softmax, inputs.indices)) / batch_size

        # regularization part of loss
        reg_loss = 2 * tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        loss = ll_loss + self.lam * reg_loss + anneal * KL

        train_op = self._build_optimizer(loss)
        return logits, ll_loss, train_op

    def train(
            self,
            n_epochs: int,
            validation_data_input: sparse.csr_matrix,
            validation_data_true: sparse.csr_matrix,
            metrics: dict,  # Dict[str, matrix -> matrix -> float]
            validation_step: 10,
    ):
        """
        Train the model
        :param n_epochs: number of epochs
        :param train_data:  train matrix of shape users count x items count
        :param metrics: Dictionary of metric names to metric functions
        :param validation_step: If it's set to n then validation is run once every n epochs
        """

        self.total_time_start = time.time()
        self.session.run(self.iter.initializer)

        num_workers = hvd.size()
        for epoch in range(1, n_epochs + 1):

            init_time = time.time()

            for i in range(self.n_batch_per_train):
                self.session.run(self.optimizer)
            batches_per_epoch = i + 1

            training_duration = time.time() - init_time
            self.time_elapsed_training_history.append(training_duration)
            training_throughput = num_workers * batches_per_epoch * self.batch_size_train / training_duration
            self.training_throughputs.append(training_throughput)

            dllogger.log(data={"train_epoch_time" : training_duration,
                               "train_throughput" : training_throughput},
                         step=(epoch,))

            if (epoch % validation_step == 0 or epoch == n_epochs) and hvd.rank() == 0:
                init_time = time.time()
                metrics_scores = self.test(validation_data_input,
                                           validation_data_true,
                                           metrics,
                                           epoch=epoch)

                for name, score in metrics_scores.items():
                    self.metrics_history[name].append(score)

                validation_duration = time.time() - init_time
                self.time_elapsed_validation_history.append(validation_duration)

                dllogger.log(data={"valid_time" : validation_duration},
                             step=(epoch,))

                self.log_metrics(epoch, metrics_scores, n_epochs)
        self.total_time = time.time() - self.total_time_start
        if hvd.rank() == 0:
            self.log_final_stats()

    def test(
            self,
            test_data_input,
            test_data_true,
            metrics,
            epoch=0,
    ):
        """
        Test the performance of the model
        :param metrics: Dictionary of metric names to metric functions
        """
        metrics_scores = defaultdict(lambda: [])
        gen = self.batch_iterator_val(test_data_input, test_data_true)
        for idxs, vals, X_true in gen():
            inference_begin = time.time()

            if self.trace:
                pred_val, _ = self.session.run([self.top_k_validation, self.increment_global_step],
                                            feed_dict={self.inputs_validation: (idxs, vals)})
            else:
                pred_val = self.session.run(self.top_k_validation,
                                            feed_dict={self.inputs_validation: (idxs, vals)})
            elapsed = time.time() - inference_begin
            pred_val = np.copy(pred_val)

            inference_throughput = self.batch_size_validation / elapsed
            self.inference_throughputs.append(inference_throughput)
            dllogger.log(data={"inference_throughput" : inference_throughput},
                         step=(epoch,))

            for name, metric in metrics.items():
                metrics_scores[name].append(metric(X_true, pred_val))

        # For some random seeds passed to the data preprocessing script
        # the test set might contain samples that have no true items to be predicted.
        # At least one such sample is present in about 7% of all possible test sets.
        # We decided not to change the preprocessing to remain comparable to the original implementation.
        # Therefore we're using the nan-aware mean from numpy to ignore users with no items to be predicted. 
        return {name: np.nanmean(scores) for name, scores in metrics_scores.items()}

    def query(self, input_data: np.ndarray):
        """
        inference for batch size 1

        :param input_data:
        :return:
        """
        query_start = time.time()
        indices = np.stack([np.zeros(len(input_data)), input_data], axis=1)
        values = np.ones(shape=(1, len(input_data)))
        values = normalize(values)
        values = values.reshape(-1)

        sess_run_start = time.time()
        res = self.session.run(
            self.top_k_query,
            feed_dict={self.inputs_query: (indices,
                                           values)})
        query_end_time = time.time()
        LOG.info('query time: {}'.format(query_end_time - query_start))
        LOG.info('sess run time: {}'.format(query_end_time - sess_run_start))
        return res

    def _increment_global_step(self):
        res = self.session.run(self.increment_global_step)
        print('increment global step result: ', res)

    def batch_iterator_train(self, data_input):
        """
        :return: iterator of consecutive batches and its length
        """
        data_input = normalize(data_input)

        indices = np.arange(data_input.shape[0])
        np.random.shuffle(indices)
        data_input = data_input[list(indices)]

        nsize, _ = data_input.shape
        csize = nsize // self.batch_size_train * self.batch_size_train

        def generator():
            while True:
                for st_idx in range(0, csize, self.batch_size_train):
                    idxs, vals = self.next_batch(data_input,st_idx, self.batch_size_train)

                    nnz = vals.shape[0]
                    vals *= (2 * np.random.randint(2, size=nnz))
                    yield (idxs, vals)

        return generator, int(np.ceil(csize / self.batch_size_train))

    def batch_iterator_val(self, data_input, data_true):
        """
        :return: iterator of consecutive batches and its length
        """

        data_input = normalize(data_input)

        nsize, _ = data_input.shape
        csize = nsize // self.batch_size_validation * self.batch_size_validation

        def generator():
            for st_idx in range(0, csize, self.batch_size_validation):
                idxs, vals = self.next_batch(data_input, st_idx, self.batch_size_validation)
                yield idxs, vals, data_true[st_idx:st_idx + self.batch_size_validation]

        return generator

    def next_batch(self, data_input, st_idx, batch_size):
        batch = data_input[st_idx:st_idx + batch_size].copy()
        batch = batch.tocoo()
        idxs = np.stack([batch.row, batch.col], axis=1)
        vals = batch.data
        return idxs,vals

    def log_metrics(self, epoch, metrics_scores, n_epochs):
        dllogger.log(data=metrics_scores, step=(epoch,))

    def log_final_stats(self):
        data = {"total_train_time": np.sum(self.time_elapsed_training_history),
                "total_valid_time": np.sum(self.time_elapsed_validation_history),
                "average_train_epoch time": np.mean(self.time_elapsed_training_history),
                "average_validation_time": np.mean(self.time_elapsed_validation_history),
                "total_elapsed_time" : self.total_time,
                "mean_training_throughput": np.mean(self.training_throughputs[10:]),
                "mean_inference_throughput": np.mean(self.inference_throughputs),
                "max_training_throughput": np.max(self.training_throughputs[10:]),
                "max_inference_throughput": np.max(self.inference_throughputs)}

        for metric_name, metric_values in self.metrics_history.items():
            data["final_" + metric_name] = metric_values[-1]

        dllogger.log(data=data, step=tuple())
