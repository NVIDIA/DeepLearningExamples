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
#
# author: Tomasz Grel (tgrel@nvidia.com)


import tensorflow as tf
from embedding import Embedding, SplitEmbedding
import interaction
import tensorflow.keras.initializers as initializers
import math
import horovod.tensorflow as hvd
from distributed_utils import BroadcastingInitializer
import numpy as np
import time
from utils import dist_print

try:
    from tensorflow_dot_based_interact.python.ops import dot_based_interact_ops
except ImportError:
    print('WARNING: Could not import the custom dot-interaction kernels')


def scale_grad(grad, factor):
    if isinstance(grad, tf.IndexedSlices):
        # sparse gradient
        grad._values = grad._values * factor
        return grad
    else:
        # dense gradient
        return grad * factor


class DataParallelSplitter:
    def __init__(self, batch_size):
        local_batch_size = (batch_size // hvd.size())
        if local_batch_size % 1 != 0:
            raise ValueError("Global batch size must be divisible by the number of workers!")
        local_batch_size = int(local_batch_size)

        batch_sizes_per_gpu = [local_batch_size] * hvd.size()
        indices = tuple(np.cumsum([0] + list(batch_sizes_per_gpu)))
        self.begin_idx = indices[hvd.rank()]
        self.end_idx = indices[hvd.rank() + 1]

    def __call__(self, x):
        x = x[self.begin_idx:self.end_idx]
        x = tf.cast(x, dtype=tf.float32)
        return x


class DlrmTrainer:
    def __init__(self, dlrm, embedding_optimizer, mlp_optimizer, amp, lr_scheduler):
        self.dlrm = dlrm
        self.embedding_optimizer = embedding_optimizer
        self.mlp_optimizer = mlp_optimizer
        self.amp = amp
        self.lr_scheduler = lr_scheduler
        self.bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                      from_logits=True)

    def _bottom_part_weight_update(self, unscaled_gradients):
        bottom_gradients = self.dlrm.extract_bottom_gradients(unscaled_gradients)

        if hvd.size() > 1:
            # need to correct for allreduced gradients being averaged and model-parallel ones not
            bottom_gradients = [scale_grad(g, 1 / hvd.size()) for g in bottom_gradients]

        if self.mlp_optimizer is self.embedding_optimizer:
            self.mlp_optimizer.apply_gradients(zip(bottom_gradients, self.dlrm.bottom_variables))
        else:
            bottom_grads_and_vars = list(zip(bottom_gradients, self.dlrm.bottom_variables))

            embedding_grads_and_vars = [(g,v) for g,v in bottom_grads_and_vars if 'embedding' in v.name]
            bottom_mlp_grads_and_vars = [(g,v) for g,v in bottom_grads_and_vars if 'embedding' not in v.name]

            self.mlp_optimizer.apply_gradients(bottom_mlp_grads_and_vars)
            self.embedding_optimizer.apply_gradients(embedding_grads_and_vars)

    def _top_part_weight_update(self, unscaled_gradients):
        top_gradients = self.dlrm.extract_top_gradients(unscaled_gradients)

        if hvd.size() > 1:
            top_gradients = [hvd.allreduce(g, name="top_gradient_{}".format(i), op=hvd.Average,
                                           compression=hvd.compression.NoneCompressor) for i, g in
                             enumerate(top_gradients)]

        self.mlp_optimizer.apply_gradients(zip(top_gradients, self.dlrm.top_variables))

    @tf.function
    def train_step(self, numerical_features, categorical_features, labels):
        self.lr_scheduler()

        with tf.GradientTape() as tape:
            predictions = self.dlrm(inputs=(numerical_features, categorical_features),
                                    training=True)

            unscaled_loss = self.bce(labels, predictions)
            # tf keras doesn't reduce the loss when using a Custom Training Loop
            unscaled_loss = tf.math.reduce_mean(unscaled_loss)
            scaled_loss = self.mlp_optimizer.get_scaled_loss(unscaled_loss) if self.amp else unscaled_loss

        scaled_gradients = tape.gradient(scaled_loss, self.dlrm.trainable_variables)

        if self.amp:
            unscaled_gradients = self.mlp_optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            unscaled_gradients = scaled_gradients

        self._bottom_part_weight_update(unscaled_gradients)
        self._top_part_weight_update(unscaled_gradients)

        if hvd.size() > 1:
            # compute mean loss for all workers for reporting
            mean_loss = hvd.allreduce(unscaled_loss, name="mean_loss", op=hvd.Average)
        else:
            mean_loss = unscaled_loss

        return mean_loss


def evaluate(validation_pipeline, dlrm, timer, auc_thresholds,
             data_parallel_splitter, max_steps=None):

    if auc_thresholds is not None:
        auc_metric = tf.keras.metrics.AUC(num_thresholds=auc_thresholds,
                                          curve='ROC', summation_method='interpolation',
                                          name='my_auc')
    else:
        auc_metric = None

    bce_op = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                from_logits=False)
    auc, test_loss = 0, 0
    latencies, all_test_losses = [], []
    distributed = hvd.size() != 1
    iterator = enumerate(validation_pipeline)
    while True:
        begin = time.time()

        try:
            eval_step, ((numerical_features, categorical_features), labels) = next(iterator)
        except StopIteration:
            break

        if dlrm.data_parallel_bottom_mlp:
            numerical_features = data_parallel_splitter(numerical_features)

        if max_steps is not None and eval_step >= max_steps:
            break

        y_pred = dlrm((numerical_features, categorical_features), sigmoid=True)
        end = time.time()
        latency = end - begin
        latencies.append(latency)

        if distributed:
            y_pred = hvd.allgather(y_pred)

        timer.step_test()
        if hvd.rank() == 0 and auc_metric is not None:
            auc_metric.update_state(labels, y_pred)
            test_loss = bce_op(labels, y_pred)
            all_test_losses.append(test_loss)

    if hvd.rank() == 0 and auc_metric is not None:
        auc = auc_metric.result().numpy().item()
        test_loss = tf.reduce_mean(all_test_losses).numpy().item()

    return auc, test_loss, latencies


class Dlrm(tf.keras.Model):
    def __init__(self, FLAGS, dataset_metadata, multi_gpu_metadata):
        super(Dlrm, self).__init__()
        local_table_ids = multi_gpu_metadata.rank_to_categorical_ids[hvd.rank()]
        self.table_sizes = [dataset_metadata.categorical_cardinalities[i] for i in local_table_ids]
        self.rank_to_feature_count = multi_gpu_metadata.rank_to_feature_count
        self.distributed = hvd.size() > 1
        self.batch_size = FLAGS.batch_size
        self.num_all_categorical_features = len(dataset_metadata.categorical_cardinalities)

        self.amp = FLAGS.amp
        self.dataset_metadata = dataset_metadata

        self.embedding_dim = FLAGS.embedding_dim

        if FLAGS.dot_interaction == 'custom_cuda':
            self.interact_op = dot_based_interact_ops.dot_based_interact
        elif FLAGS.dot_interaction == 'tensorflow':
            self.interact_op =  interaction.dot_interact
        elif FLAGS.dot_interaction == 'dummy':
            self.interact_op = interaction.dummy_dot_interact
        else:
            raise ValueError(f'Unknown dot-interaction implementation {FLAGS.dot_interaction}')

        self.dummy_embedding = FLAGS.dummy_embedding

        self.experimental_columnwise_split = FLAGS.experimental_columnwise_split
        self.data_parallel_bottom_mlp = FLAGS.data_parallel_bottom_mlp

        if self.experimental_columnwise_split:
            self.local_embedding_dim = self.embedding_dim // hvd.size()
        else:
            self.local_embedding_dim = self.embedding_dim

        self.embedding_type = FLAGS.embedding_type
        self.embedding_trainable = FLAGS.embedding_trainable

        self.bottom_mlp_dims = [int(d) for d in FLAGS.bottom_mlp_dims]
        self.top_mlp_dims = [int(d) for d in FLAGS.top_mlp_dims]

        self.top_mlp_padding = None
        self.bottom_mlp_padding = None

        self.variables_partitioned = False
        self.running_bottom_mlp = (not self.distributed) or (hvd.rank() == 0) or self.data_parallel_bottom_mlp

        self.num_numerical_features = FLAGS.num_numerical_features
        # override in case there's no numerical features in the dataset
        if self.num_numerical_features == 0:
            self.running_bottom_mlp = False

        if self.running_bottom_mlp:
            self._create_bottom_mlp()
        self._create_embeddings()
        self._create_top_mlp()

        # write embedding checkpoints of 1M rows at a time
        self.embedding_checkpoint_batch = 1024 * 1024

    def _create_bottom_mlp_padding(self, multiple=8):
        num_features = self.dataset_metadata.num_numerical_features
        pad_to = tf.math.ceil(num_features / multiple) * multiple
        pad_to = tf.cast(pad_to, dtype=tf.int32)
        padding_features = pad_to - num_features

        batch_size = self.batch_size // hvd.size() if self.data_parallel_bottom_mlp else self.batch_size

        padding_shape = [batch_size, padding_features]
        dtype=tf.float16 if self.amp else tf.float32
        self.bottom_mlp_padding = self.add_weight("bottom_mlp_padding", shape=padding_shape, dtype=dtype,
                                                   initializer=initializers.Zeros(), trainable=False)

    def _create_top_mlp_padding(self, multiple=8):
        num_features = self.num_all_categorical_features
        if self.num_numerical_features != 0:
            num_features += 1
        num_features = num_features * (num_features - 1)
        num_features = num_features // 2
        num_features = num_features + self.embedding_dim

        pad_to = tf.math.ceil(num_features / multiple) * multiple
        pad_to = tf.cast(pad_to, dtype=tf.int32)
        padding_features = pad_to - num_features

        padding_shape = [self.batch_size // hvd.size(), padding_features]
        dtype=tf.float16 if self.amp else tf.float32
        self.top_mlp_padding = self.add_weight("top_mlp_padding", shape=padding_shape, dtype=dtype,
                                                initializer=initializers.Zeros(), trainable=False)

    def _create_bottom_mlp(self):
        self._create_bottom_mlp_padding()
        self.bottom_mlp_layers = []
        for dim in self.bottom_mlp_dims:
            kernel_initializer = initializers.GlorotNormal()
            bias_initializer = initializers.RandomNormal(stddev=math.sqrt(1. / dim))

            if self.data_parallel_bottom_mlp:
                kernel_initializer = BroadcastingInitializer(kernel_initializer)
                bias_initializer = BroadcastingInitializer(bias_initializer)

            l = tf.keras.layers.Dense(dim, activation='relu',
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer)
            self.bottom_mlp_layers.append(l)

    def _create_top_mlp(self):
        self._create_top_mlp_padding()
        self.top_mlp = []
        for i, dim in enumerate(self.top_mlp_dims):
            if i == len(self.top_mlp_dims) - 1:
                # final layer
                activation = 'linear'
            else:
                activation = 'relu'

            kernel_initializer = BroadcastingInitializer(initializers.GlorotNormal())
            bias_initializer = BroadcastingInitializer(initializers.RandomNormal(stddev=math.sqrt(1. / dim)))

            l = tf.keras.layers.Dense(dim, activation=activation,
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer)
            self.top_mlp.append(l)

    def _create_embeddings(self):
        self.embedding_layers = []
        if self.embedding_type == 'embedding':
            for i, table_size in enumerate(self.table_sizes):
                l = Embedding(input_dim=table_size,
                              output_dim=self.local_embedding_dim,
                              trainable=self.embedding_trainable)
                self.embedding_layers.append(l)

        elif self.embedding_type == 'split_embedding':
            for i, table_size in enumerate(self.table_sizes):
                l = SplitEmbedding(input_dim=table_size,
                                   output_dim=self.local_embedding_dim,
                                   trainable=self.embedding_trainable)
                self.embedding_layers.append(l)
        else:
            raise ValueError(f'Unknown embedding type {self.embedding_type}')

    def _partition_variables(self):
        self.bottom_variables = [v for v in self.trainable_variables if 'bottom_model' in v.name]
        self.bottom_variable_indices = [i for i,v in enumerate(self.trainable_variables) if 'bottom_model' in v.name]

        self.top_variables = [v for v in self.trainable_variables if 'bottom_model' not in v.name]
        self.top_variable_indices = [i for i, v in enumerate(self.trainable_variables) if 'bottom_model' not in v.name]
        self.variables_partitioned = True

    def extract_bottom_gradients(self, all_gradients):
        if not self.variables_partitioned:
            self._partition_variables()
        return [all_gradients[i] for i in self.bottom_variable_indices]

    def extract_top_gradients(self, all_gradients):
        if not self.variables_partitioned:
            self._partition_variables()
        return [all_gradients[i] for i in self.top_variable_indices]

    def force_initialization(self):
        if self.running_bottom_mlp:
            if self.data_parallel_bottom_mlp:
                numerical_features = tf.zeros(shape=[self.batch_size / hvd.size(),
                                                     self.dataset_metadata.num_numerical_features])
            else:
                numerical_features = tf.zeros(shape=[self.batch_size,
                                                     self.dataset_metadata.num_numerical_features])
        else:
            numerical_features = None

        categorical_features = [tf.zeros(shape=[self.batch_size, 1], dtype=tf.int32)] * len(self.table_sizes)
        self((numerical_features, categorical_features))

    @tf.function
    def call(self, inputs, sigmoid=False):
        numerical_features, cat_features = inputs
        embedding_outputs = self._call_embeddings(cat_features)

        if self.running_bottom_mlp:
            bottom_mlp_out = self._call_bottom_mlp(numerical_features)
        else:
            bottom_mlp_out = None

        if self.distributed:
            if self.experimental_columnwise_split:
                interaction_input = self._call_alltoall_experimental_columnwise(
                                             embedding_outputs,
                                             bottom_mlp_out)
            else:
                interaction_input = self._call_alltoall(embedding_outputs, bottom_mlp_out)
        else:
            if bottom_mlp_out is not None:
                bottom_part_output = tf.concat([bottom_mlp_out] + embedding_outputs,
                                               axis=1)
            else:
                bottom_part_output = tf.concat(embedding_outputs, axis=1)

            num_categorical_features = len(self.dataset_metadata.categorical_cardinalities)
            interaction_input = tf.reshape(bottom_part_output,
                                           [-1, num_categorical_features + 1,
                                            self.embedding_dim])

        if not self.data_parallel_bottom_mlp:
            bottom_mlp_out = interaction_input[:, 0, :]

        x = self.interact_op(interaction_input, tf.squeeze(bottom_mlp_out))
        x = self._call_top_mlp(x)

        if sigmoid:
            x = tf.math.sigmoid(x)
        return x

    def _call_bottom_mlp(self, numerical_features):
        if self.amp:
            numerical_features = tf.cast(numerical_features, dtype=tf.float16)
        x = tf.concat([numerical_features, self.bottom_mlp_padding], axis=1)

        name_scope = "bottom_mlp" if self.data_parallel_bottom_mlp else "bottom_model"
        with tf.name_scope(name_scope):
            for l in self.bottom_mlp_layers:
                x = l(x)
            x = tf.expand_dims(x, axis=1)
            bottom_mlp_out = x
        return bottom_mlp_out

    def _call_dummy_embeddings(self, cat_features):
        batch_size = tf.shape(cat_features)[0]
        num_embeddings = tf.shape(cat_features)[1]
        dtype = tf.float16 if self.amp else tf.float32
        return [tf.zeros(shape=[batch_size, num_embeddings, self.embedding_dim], dtype=dtype)]

    def _call_embeddings(self, cat_features):
        if self.dummy_embedding:
            return self._call_dummy_embeddings(cat_features)

        with tf.name_scope("bottom_model"):
            embedding_outputs = []
            if self.table_sizes:
                for i, l in enumerate(self.embedding_layers):
                    indices = tf.cast(cat_features[i], tf.int32)
                    out = l(indices)
                    embedding_outputs.append(out)
        if self.amp:
            embedding_outputs = [tf.cast(e, dtype=tf.float16) for e in embedding_outputs]
        return embedding_outputs

    def _call_alltoall(self, embedding_outputs, bottom_mlp_out=None):
        num_tables = len(self.table_sizes)
        if bottom_mlp_out is not None and not self.data_parallel_bottom_mlp:
            bottom_part_output = tf.concat([bottom_mlp_out] + embedding_outputs,
                                           axis=1)
            num_tables += 1
        else:
            bottom_part_output = tf.concat(embedding_outputs, axis=1)

        global_batch = tf.shape(bottom_part_output)[0]
        world_size = hvd.size()
        local_batch = global_batch // world_size
        embedding_dim = self.embedding_dim

        alltoall_input = tf.reshape(bottom_part_output,
                                    shape=[global_batch * num_tables,
                                           embedding_dim])

        splits = [tf.shape(alltoall_input)[0] // world_size] * world_size

        alltoall_output = hvd.alltoall(tensor=alltoall_input, splits=splits, ignore_name_scope=True)

        vectors_per_worker = [x * local_batch for x in self.rank_to_feature_count]
        alltoall_output = tf.split(alltoall_output,
                                   num_or_size_splits=vectors_per_worker,
                                   axis=0)
        interaction_input = [tf.reshape(x, shape=[local_batch, -1, embedding_dim]) for x in alltoall_output]

        if self.data_parallel_bottom_mlp:
            interaction_input = [bottom_mlp_out] + interaction_input

        interaction_input = tf.concat(interaction_input, axis=1)  # shape=[local_batch, num_vectors, vector_dim]
        return interaction_input

    def _call_alltoall_experimental_columnwise(self, embedding_outputs, bottom_mlp_out):
        bottom_part_output = tf.concat(embedding_outputs, axis=1)

        global_batch = tf.shape(bottom_part_output)[0]
        world_size = hvd.size()
        local_batch = global_batch // world_size
        num_tables = len(self.table_sizes)

        alltoall_input = tf.transpose(bottom_part_output, perm=[0, 2, 1])
        alltoall_input = tf.reshape(alltoall_input, shape=[global_batch * self.local_embedding_dim,
                                                           num_tables])

        splits = [tf.shape(alltoall_input)[0] // world_size] * world_size

        alltoall_output = hvd.alltoall(tensor=alltoall_input, splits=splits, ignore_name_scope=True)

        alltoall_output = tf.split(alltoall_output,
                                   num_or_size_splits=hvd.size(),
                                   axis=0)
        interaction_input = [tf.reshape(x, shape=[local_batch,
                                                  self.local_embedding_dim, num_tables]) for x in alltoall_output]

        interaction_input = tf.concat(interaction_input, axis=1)  # shape=[local_batch, vector_dim, num_tables]
        interaction_input = tf.transpose(interaction_input,
                                         perm=[0, 2, 1])  # shape=[local_batch, num_tables, vector_dim]

        if self.running_bottom_mlp:
            interaction_input = tf.concat([bottom_mlp_out,
                                           interaction_input],
                                          axis=1)  # shape=[local_batch, num_tables + 1, vector_dim]
        return interaction_input

    def _call_top_mlp(self, x):
        if self.interact_op != 'custom_cuda':
            x = tf.concat([x, self.top_mlp_padding], axis=1)

        with tf.name_scope("top_model"):
            for i, l in enumerate(self.top_mlp):
                x = l(x)
            x = tf.cast(x, dtype=tf.float32)
        return x

    @staticmethod
    def _get_variable_path(checkpoint_path, v, i=0):
        checkpoint_path = checkpoint_path + f'_rank_{hvd.rank()}'
        name = v.name.replace('/', '_').replace(':', '_')
        return checkpoint_path + '_' + name + f'_{i}' + '.npy'

    def maybe_save_checkpoint(self, checkpoint_path):
        if checkpoint_path is None:
            return

        dist_print('Saving a checkpoint...')
        for v in self.trainable_variables:
            filename = self._get_variable_path(checkpoint_path, v)
            if 'embedding' not in v.name:
                np.save(arr=v.numpy(), file=filename)
                continue
            print(f'saving embedding {v.name}')
            chunks = math.ceil(v.shape[0] / self.embedding_checkpoint_batch)
            for i in range(chunks):
                filename = self._get_variable_path(checkpoint_path, v, i)
                end = min((i + 1) * self.embedding_checkpoint_batch, v.shape[0])

                indices = tf.range(start=i * self.embedding_checkpoint_batch,
                                   limit=end,
                                   dtype=tf.int32)

                arr = tf.gather(params=v, indices=indices, axis=0)
                arr = arr.numpy()
                np.save(arr=arr, file=filename)

        dist_print('Saved a checkpoint to ', checkpoint_path)

    def maybe_restore_checkpoint(self, checkpoint_path):
        if checkpoint_path is None:
            return

        dist_print('Restoring a checkpoint...')
        self.force_initialization()

        for v in self.trainable_variables:
            filename = self._get_variable_path(checkpoint_path, v)
            if 'embedding' not in v.name:
                numpy_var = np.load(file=filename)
                v.assign(numpy_var)
                continue

            chunks = math.ceil(v.shape[0] / self.embedding_checkpoint_batch)
            for i in range(chunks):
                filename = self._get_variable_path(checkpoint_path, v, i)
                start = i * self.embedding_checkpoint_batch
                numpy_arr = np.load(file=filename)
                indices = tf.range(start=start,
                                   limit=start + numpy_arr.shape[0],
                                   dtype=tf.int32)
                update = tf.IndexedSlices(values=numpy_arr, indices=indices, dense_shape=v.shape)
                v.scatter_update(sparse_delta=update)

        dist_print('Restored a checkpoint from', checkpoint_path)


# dummy model for profiling and debugging
class DummyDlrm(tf.keras.Model):
    def __init__(self, FLAGS, dataset_metadata):
        super(DummyDlrm, self).__init__()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid',
                                           kernel_initializer='glorot_normal',
                                           bias_initializer=initializers.RandomNormal(stddev=math.sqrt(1. / 1))
                                           )
        self.dataset_metadata = dataset_metadata
        self.top_variables = [v for v in self.trainable_variables if 'bottom_model' not in v.name]
        self.variables_partitioned = False
        self.batch_size = FLAGS.batch_size
        self.data_parallel_bottom_mlp = FLAGS.data_parallel_bottom_mlp

    def call(self, inputs, sigmoid=False):
        x = tf.zeros(shape=[self.batch_size // hvd.size(),
                            self.dataset_metadata.num_numerical_features],
                     dtype=tf.float32)
        x = self.dense(x)
        x = tf.cast(x, dtype=tf.float32)
        if sigmoid:
            x = tf.math.sigmoid(x)
        return x

    def _partition_variables(self):
        self.bottom_variables = [v for v in self.trainable_variables if 'bottom_model' in v.name]
        self.bottom_variable_indices = [i for i,v in enumerate(self.trainable_variables) if 'bottom_model' in v.name]

        self.top_variables = [v for v in self.trainable_variables if 'bottom_model' not in v.name]
        self.top_variable_indices = [i for i, v in enumerate(self.trainable_variables) if 'bottom_model' not in v.name]
        self.variables_partitioned = True

    def extract_bottom_gradients(self, all_gradients):
        if not self.variables_partitioned:
            self._partition_variables()
        return [all_gradients[i] for i in self.bottom_variable_indices]

    def extract_top_gradients(self, all_gradients):
        if not self.variables_partitioned:
            self._partition_variables()
        return [all_gradients[i] for i in self.top_variable_indices]
