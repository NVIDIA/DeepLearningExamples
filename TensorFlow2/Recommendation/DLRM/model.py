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
from embedding import DualEmbeddingGroup
import interaction
import tensorflow.keras.initializers as initializers
import math
import horovod.tensorflow as hvd
from distributed_utils import BroadcastingInitializer, is_multinode
import numpy as np
import time
import os
from utils import dist_print, get_variable_path
from tensorflow.python.keras.saving.saving_utils import model_input_signature
from collections import OrderedDict


try:
    from tensorflow_dot_based_interact.python.ops import dot_based_interact_ops
except ImportError:
    print('WARNING: Could not import the custom dot-interaction kernels')

# wrap metric computations in tf.function
@tf.function
def update_auc_metric(auc_metric, labels, y_pred):
    auc_metric.update_state(labels, y_pred)

@tf.function
def compute_bce_loss(bce_op, labels, y_pred):
    return bce_op(labels, y_pred)

def scale_grad(grad, factor):
    if isinstance(grad, tf.IndexedSlices):
        # sparse gradient
        grad._values = grad._values * factor
        return grad
    else:
        # dense gradient
        return grad * factor

def _create_inputs_dict(numerical_features, categorical_features):
    # Passing inputs as (numerical_features, categorical_features) changes the model
    # input signature to (<tensor, [list of tensors]>).
    # This leads to errors while loading the saved model.
    # TF flattens the inputs while loading the model,
    # so the inputs are converted from (<tensor, [list of tensors]>) -> [list of tensors]
    # see _set_inputs function in training_v1.py:
    # https://github.com/tensorflow/tensorflow/blob/7628750678786f1b65e8905fb9406d8fbffef0db/tensorflow/python/keras/engine/training_v1.py#L2588)
    inputs = OrderedDict()
    inputs['numerical_features'] = numerical_features
    inputs['categorical_features'] = categorical_features
    return inputs

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
    def __init__(self, dlrm, embedding_optimizer, mlp_optimizer, amp, lr_scheduler, dp_splitter,
                 data_parallel_bottom_mlp, pipe, cpu):

        self.dlrm = dlrm
        self.embedding_optimizer = embedding_optimizer
        self.mlp_optimizer = mlp_optimizer
        self.amp = amp
        self.lr_scheduler = lr_scheduler
        self.bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                      from_logits=True)
        self.dp_splitter = dp_splitter
        self.data_parallel_bottom_mlp = data_parallel_bottom_mlp
        self.cpu = cpu
        self.pipe = iter(pipe.op())

    def _bottom_part_weight_update(self, unscaled_gradients):
        bottom_gradients = self.dlrm.extract_bottom_gradients(unscaled_gradients)

        if hvd.size() > 1:
            # need to correct for allreduced gradients being averaged and model-parallel ones not
            bottom_gradients = [scale_grad(g, 1 / hvd.size()) for g in bottom_gradients]

        if self.mlp_optimizer is self.embedding_optimizer:
            self.mlp_optimizer.apply_gradients(zip(bottom_gradients, self.dlrm.model_parallel_variables))
        else:
            bottom_grads_and_vars = list(zip(bottom_gradients, self.dlrm.model_parallel_variables))

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

        self.mlp_optimizer.apply_gradients(zip(top_gradients, self.dlrm.data_parallel_variables))

    def broadcasting_dataloader_wrapper(self):
        if hvd.rank() == 0:
            (numerical_features, categorical_features), labels = self.pipe.get_next()

            # Bitcasting to float32 before broadcast and back to int32 right afterwards is necessary
            # otherwise tensorflow performs a spurious D2H and H2D transfer on this tensor.
            # Without this call, the columnwise-split mode gets about 2x slower.
            categorical_features = tf.bitcast(categorical_features, type=tf.float32)
        else:
            # using random uniform instead of e.g., tf.zeros is necessary here
            # tf.zeros would be placed on CPU causing a device clash in the broadcast
            numerical_features = tf.random.uniform(shape=[self.dlrm.batch_size, self.dlrm.num_numerical_features],
                                                   dtype=tf.float16)
            categorical_features = tf.random.uniform(maxval=1, dtype=tf.float32,
                                                     shape=[self.dlrm.batch_size, len(self.dlrm.table_sizes)])
            labels = tf.random.uniform(maxval=1, shape=[self.dlrm.batch_size], dtype=tf.int32)
            labels = tf.cast(labels, dtype=tf.int8)

        numerical_features = hvd.broadcast(numerical_features, root_rank=0,
                                           name='numerical_broadcast')

        categorical_features = hvd.broadcast(categorical_features, root_rank=0,
                                             name='cat_broadcast')

        labels = hvd.broadcast(labels, root_rank=0,
                               name='labels_broadcast')

        categorical_features = tf.bitcast(categorical_features, type=tf.int32)
        return (numerical_features, categorical_features), labels

    def _load_data(self):
        if hvd.size() > 1 and self.dlrm.columnwise_split and not is_multinode():
            (numerical_features, categorical_features), labels = self.broadcasting_dataloader_wrapper()
        else:
            (numerical_features, categorical_features), labels = self.pipe.get_next()
        labels = self.dp_splitter(labels)
        if self.data_parallel_bottom_mlp:
            numerical_features = self.dp_splitter(numerical_features)
        return (numerical_features, categorical_features), labels

    @tf.function
    def train_step(self):
        device = '/CPU:0' if self.cpu else '/GPU:0'
        with tf.device(device):
            self.lr_scheduler()
            (numerical_features, categorical_features), labels = self._load_data()

            inputs = _create_inputs_dict(numerical_features, categorical_features)
            with tf.GradientTape() as tape:
                predictions = self.dlrm(inputs=inputs, training=True)

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
             data_parallel_splitter, max_steps=None, cast_dtype=None):

    auc, test_loss = 0, 0
    latencies, all_test_losses = [], []
    distributed = hvd.size() != 1

    pipe = iter(validation_pipeline.op())


    auc_metric = tf.keras.metrics.AUC(num_thresholds=auc_thresholds,
                                      curve='ROC', summation_method='interpolation',
                                      from_logits=True)
    bce_op = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                from_logits=True)
    for eval_step in range(len(validation_pipeline)):
        begin = time.time()

        (numerical_features, categorical_features), labels = pipe.get_next()

        if hasattr(dlrm, 'data_parallel_bottom_mlp') and dlrm.data_parallel_bottom_mlp:
            numerical_features = data_parallel_splitter(numerical_features)

        if cast_dtype is not None:
            numerical_features = tf.cast(numerical_features, cast_dtype)

        if max_steps is not None and eval_step >= max_steps:
            break

        inputs = _create_inputs_dict(numerical_features, categorical_features)
        y_pred = dlrm(inputs, sigmoid=False, training=False)
        end = time.time()
        latency = end - begin
        latencies.append(latency)

        if distributed:
            y_pred = hvd.allgather(y_pred)

        timer.step_test()
        if hvd.rank() == 0 and auc_metric is not None:
            update_auc_metric(auc_metric, labels, y_pred)
            test_loss = compute_bce_loss(bce_op, labels, y_pred)
            all_test_losses.append(test_loss)

    if hvd.rank() == 0 and dlrm.auc_metric is not None:
        auc = auc_metric.result().numpy().item()
        test_loss = tf.reduce_mean(all_test_losses).numpy().item()

    auc_metric.reset_state()
    return auc, test_loss, latencies


class Dlrm(tf.keras.Model):
    def __init__(self, FLAGS, dataset_metadata, multi_gpu_metadata):
        super(Dlrm, self).__init__()
        self.local_table_ids = multi_gpu_metadata.rank_to_categorical_ids[hvd.rank()]
        self.table_sizes = [dataset_metadata.categorical_cardinalities[i] for i in self.local_table_ids]
        self.rank_to_feature_count = multi_gpu_metadata.rank_to_feature_count

        if multi_gpu_metadata.sort_order == sorted(multi_gpu_metadata.sort_order):
            self.sort_order = None
        else:
            self.sort_order = multi_gpu_metadata.sort_order

        self.distributed = hvd.size() > 1
        self.batch_size = FLAGS.batch_size
        self.num_all_categorical_features = len(dataset_metadata.categorical_cardinalities)

        self.memory_limit = FLAGS.tf_gpu_memory_limit_gb

        self.amp = FLAGS.amp
        self.fp16 = FLAGS.fp16

        self.dataset_metadata = dataset_metadata

        self.embedding_dim = FLAGS.embedding_dim

        self.dot_interaction = FLAGS.dot_interaction
        if FLAGS.dot_interaction == 'custom_cuda':
            self.interact_op = dot_based_interact_ops.dot_based_interact
        elif FLAGS.dot_interaction == 'tensorflow':
            self.interact_op =  interaction.dot_interact
        elif FLAGS.dot_interaction == 'dummy':
            self.interact_op = interaction.dummy_dot_interact
        else:
            raise ValueError(f'Unknown dot-interaction implementation {FLAGS.dot_interaction}')

        self.cpu_embedding_type = FLAGS.cpu_embedding_type
        self.gpu_embedding_type = FLAGS.gpu_embedding_type

        self.columnwise_split = FLAGS.columnwise_split
        self.data_parallel_bottom_mlp = FLAGS.data_parallel_bottom_mlp

        if self.columnwise_split:
            self.local_embedding_dim = self.embedding_dim // hvd.size()
        else:
            self.local_embedding_dim = self.embedding_dim

        self.embedding_trainable = FLAGS.embedding_trainable

        self.bottom_mlp_dims = [int(d) for d in FLAGS.bottom_mlp_dims]
        self.top_mlp_dims = [int(d) for d in FLAGS.top_mlp_dims]

        self.variables_partitioned = False
        self.running_bottom_mlp = (not self.distributed) or (hvd.rank() == 0) or self.data_parallel_bottom_mlp

        self.num_numerical_features = dataset_metadata.num_numerical_features
        # override in case there's no numerical features in the dataset
        if self.num_numerical_features == 0:
            self.running_bottom_mlp = False

        if self.running_bottom_mlp:
            self._create_bottom_mlp()
        self._create_embeddings()
        self._create_top_mlp()

        # create once to avoid tf.function recompilation at each eval
        self.create_eval_metrics(FLAGS)

    def _get_bottom_mlp_padding(self, batch_size, multiple=8):
        num_features = self.dataset_metadata.num_numerical_features
        pad_to = tf.math.ceil(num_features / multiple) * multiple
        pad_to = tf.cast(pad_to, dtype=tf.int32)
        padding_features = pad_to - num_features

        padding_shape = [batch_size, padding_features]
        dtype=tf.float16 if self.amp or self.fp16 else tf.float32
        return tf.zeros(shape=padding_shape, dtype=dtype)

    def _get_top_mlp_padding(self, batch_size, multiple=8):
        num_features = self.num_all_categorical_features
        if self.num_numerical_features != 0:
            num_features += 1
        num_features = num_features * (num_features - 1)
        num_features = num_features // 2
        num_features = num_features + self.embedding_dim

        pad_to = tf.math.ceil(num_features / multiple) * multiple
        pad_to = tf.cast(pad_to, dtype=tf.int32)
        padding_features = pad_to - num_features

        padding_shape = [batch_size, padding_features]
        dtype=tf.float16 if self.amp or self.fp16 else tf.float32
        return tf.zeros(shape=padding_shape, dtype=dtype)

    def _create_bottom_mlp(self):
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
        feature_names = [f'feature_{i}_{nrows}' for i, nrows in zip(self.local_table_ids, self.table_sizes)]

        self.embedding = DualEmbeddingGroup(cardinalities=self.table_sizes, output_dim=self.local_embedding_dim,
                                            memory_threshold=self.memory_limit,
                                            cpu_embedding=self.cpu_embedding_type,
                                            gpu_embedding=self.gpu_embedding_type,
                                            dtype=tf.float16 if self.fp16 else tf.float32,
                                            feature_names=feature_names,
                                            trainable=self.embedding_trainable)

    def create_eval_metrics(self, FLAGS):
        if FLAGS.auc_thresholds is not None:
            self.auc_metric = tf.keras.metrics.AUC(num_thresholds=FLAGS.auc_thresholds,
                                                   curve='ROC', summation_method='interpolation',
                                                   from_logits=True)
        else:
            self.auc_metric = None

        self.bce_op = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                         from_logits=True)


    def _partition_variables(self):
        self.model_parallel_variables = [v for v in self.trainable_variables if 'model_parallel' in v.name]
        self.model_parallel_variable_indices = [i for i, v in enumerate(self.trainable_variables) if 'model_parallel' in v.name]

        self.data_parallel_variables = [v for v in self.trainable_variables if 'model_parallel' not in v.name]
        self.data_parallel_variable_indices = [i for i, v in enumerate(self.trainable_variables) if 'model_parallel' not in v.name]
        self.variables_partitioned = True

    def extract_bottom_gradients(self, all_gradients):
        if not self.variables_partitioned:
            self._partition_variables()
        return [all_gradients[i] for i in self.model_parallel_variable_indices]

    def extract_top_gradients(self, all_gradients):
        if not self.variables_partitioned:
            self._partition_variables()
        return [all_gradients[i] for i in self.data_parallel_variable_indices]

    def force_initialization(self):
        if self.running_bottom_mlp:
            if self.data_parallel_bottom_mlp:
                numerical_features = tf.zeros(shape=[self.batch_size // hvd.size(),
                                                     self.dataset_metadata.num_numerical_features])
            else:
                numerical_features = tf.zeros(shape=[self.batch_size,
                                                     self.dataset_metadata.num_numerical_features])
        else:
            numerical_features = None

        categorical_features = tf.zeros(shape=[self.batch_size, len(self.table_sizes)], dtype=tf.int32)
        inputs = _create_inputs_dict(numerical_features, categorical_features)
        self(inputs=inputs, training=True)

    @tf.function
    def call(self, inputs, sigmoid=False, training=False):
        vals = list(inputs.values())
        numerical_features, cat_features = vals[0], vals[1]
        embedding_outputs = self._call_embeddings(cat_features, training=training)

        if self.running_bottom_mlp:
            bottom_mlp_out = self._call_bottom_mlp(numerical_features, training=training)
        else:
            bottom_mlp_out = None

        if self.distributed:
            if self.columnwise_split:
                interaction_input = self._call_alltoall_columnwise(
                                             embedding_outputs,
                                             bottom_mlp_out)
            else:
                interaction_input = self._call_alltoall_tablewise(embedding_outputs, bottom_mlp_out)
        else:
            if bottom_mlp_out is not None:
                bottom_part_output = tf.concat([bottom_mlp_out, embedding_outputs], axis=1)
            else:
                bottom_part_output = tf.concat(embedding_outputs, axis=1)
            bottom_part_output = self._maybe_reorder_features(bottom_part_output)

            num_categorical_features = len(self.dataset_metadata.categorical_cardinalities)
            interaction_input = tf.reshape(bottom_part_output,
                                           [-1, num_categorical_features + 1,
                                            self.embedding_dim])

        if not self.data_parallel_bottom_mlp:
            bottom_mlp_out = interaction_input[:, 0, :]

        x = self.interact_op(interaction_input, tf.squeeze(bottom_mlp_out))
        x = self._call_top_mlp(x, training=training)

        if sigmoid:
            x = tf.math.sigmoid(x)

        x = tf.cast(x, tf.float32)
        return x

    def _call_bottom_mlp(self, numerical_features, training=False):
        if self.amp:
            numerical_features = tf.cast(numerical_features, dtype=tf.float16)

        if training and self.data_parallel_bottom_mlp:
            batch_size = self.batch_size // hvd.size()
        elif training:
            batch_size = self.batch_size
        else:
            batch_size = tf.shape(numerical_features)[0]

        padding = self._get_bottom_mlp_padding(batch_size=batch_size)
        x = tf.concat([numerical_features, padding], axis=1)

        name_scope = "data_parallel" if self.data_parallel_bottom_mlp else "model_parallel"
        with tf.name_scope(name_scope):
            with tf.name_scope('bottom_mlp'):
                for l in self.bottom_mlp_layers:
                    x = l(x)
                x = tf.expand_dims(x, axis=1)
                bottom_mlp_out = x
        return bottom_mlp_out

    def _call_embeddings(self, cat_features, training=False):
        if not self.table_sizes:
            return []

        batch_size = self.batch_size if training else -1

        with tf.name_scope("model_parallel"):
            x = self.embedding(cat_features)
            shape = [batch_size, len(self.table_sizes), self.local_embedding_dim]
            x = tf.reshape(x, shape)
        if self.amp:
            x = tf.cast(x, dtype=tf.float16)
        return x

    def _maybe_reorder_features(self, interaction_input):
        """
        Sort the features to always maintain the same order no matter the number of GPUs.
        Necessary for checkpoint interoperability.
        """
        if self.sort_order is None:
            return interaction_input
        else:
            features = tf.split(interaction_input,
                                num_or_size_splits=len(self.sort_order),
                                axis=1)

            reordered_features = [features[i] for i in self.sort_order]
            reordered_tensor = tf.concat(reordered_features, axis=1)
            return reordered_tensor

    def _call_alltoall_tablewise(self, embedding_outputs, bottom_mlp_out=None):
        num_tables = len(self.table_sizes)
        if bottom_mlp_out is not None and not self.data_parallel_bottom_mlp:
            features = [bottom_mlp_out]
            if len(embedding_outputs) > 0:
                features.append(embedding_outputs)
            bottom_part_output = tf.concat(features, axis=1)
            num_tables += 1
        else:
            bottom_part_output = tf.concat(embedding_outputs, axis=1)

        global_batch = tf.shape(bottom_part_output)[0]
        world_size = hvd.size()
        local_batch = global_batch // world_size
        embedding_dim = self.embedding_dim

        alltoall_input = tf.reshape(bottom_part_output,
                                    shape=[global_batch * num_tables,
                                           embedding_dim],
                                    name='alltoall_input_reshape')

        splits = [tf.shape(alltoall_input)[0] // world_size] * world_size

        alltoall_output = hvd.alltoall(tensor=alltoall_input, splits=splits, ignore_name_scope=True)[0]

        vectors_per_worker = [x * local_batch for x in self.rank_to_feature_count]
        alltoall_output = tf.split(alltoall_output,
                                   num_or_size_splits=vectors_per_worker,
                                   axis=0)
        interaction_input = [tf.reshape(x, shape=[local_batch, -1, embedding_dim]) for x in alltoall_output]
        interaction_input = tf.concat(interaction_input, axis=1)  # shape=[local_batch, num_vectors, vector_dim]
        interaction_input = self._maybe_reorder_features(interaction_input)

        if self.data_parallel_bottom_mlp:
            interaction_input = [bottom_mlp_out, interaction_input]
            interaction_input = tf.concat(interaction_input, axis=1)  # shape=[local_batch, num_vectors, vector_dim]

        return interaction_input

    def _call_alltoall_columnwise(self, embedding_outputs, bottom_mlp_out):
        bottom_part_output = tf.concat(embedding_outputs, axis=1)

        global_batch = tf.shape(bottom_part_output)[0]
        world_size = hvd.size()
        local_batch = global_batch // world_size
        num_tables = len(self.table_sizes)

        alltoall_input = tf.transpose(bottom_part_output, perm=[0, 2, 1])
        alltoall_input = tf.reshape(alltoall_input, shape=[global_batch * self.local_embedding_dim,
                                                           num_tables])

        splits = [tf.shape(alltoall_input)[0] // world_size] * world_size

        alltoall_output = hvd.alltoall(tensor=alltoall_input, splits=splits, ignore_name_scope=True)[0]

        alltoall_output = tf.split(alltoall_output,
                                   num_or_size_splits=hvd.size(),
                                   axis=0)
        interaction_input = [tf.reshape(x, shape=[local_batch,
                                                  self.local_embedding_dim, num_tables]) for x in alltoall_output]

        interaction_input = tf.concat(interaction_input, axis=1)  # shape=[local_batch, vector_dim, num_tables]
        interaction_input = tf.transpose(interaction_input,
                                         perm=[0, 2, 1])  # shape=[local_batch, num_tables, vector_dim]

        interaction_input = self._maybe_reorder_features(interaction_input)

        if self.running_bottom_mlp:
            interaction_input = tf.concat([bottom_mlp_out,
                                           interaction_input],
                                          axis=1)  # shape=[local_batch, num_tables + 1, vector_dim]
        return interaction_input

    def _call_top_mlp(self, x, training=False):
        if self.dot_interaction != 'custom_cuda':
             batch_size = self.batch_size // hvd.size() if training else tf.shape(x)[0]
             padding = self._get_top_mlp_padding(batch_size=batch_size)
             x = tf.concat([x, padding], axis=1)

        with tf.name_scope('data_parallel'):
            with tf.name_scope('top_mlp'):
                for i, l in enumerate(self.top_mlp):
                    x = l(x)
                x = tf.cast(x, dtype=tf.float32)
        return x

    @staticmethod
    def _save_mlp_checkpoint(checkpoint_path, layers, prefix):
        for i, layer in enumerate(layers):
            for varname in ['kernel', 'bias']:
                filename = get_variable_path(checkpoint_path, name=f'{prefix}/layer_{i}/{varname}')
                print(f'saving: {varname} to {filename}')
                variable = layer.__dict__[varname]
                np.save(arr=variable.numpy(), file=filename)

    @staticmethod
    def _restore_mlp_checkpoint(checkpoint_path, layers, prefix):
        for i, layer in enumerate(layers):
            for varname in ['kernel', 'bias']:
                filename = get_variable_path(checkpoint_path, name=f'{prefix}/layer_{i}/{varname}')
                print(f'loading: {varname} from {filename}')
                variable = layer.__dict__[varname]

                numpy_var = np.load(file=filename)
                variable.assign(numpy_var)

    def save_checkpoint_if_path_exists(self, checkpoint_path):
        if checkpoint_path is None:
            return

        os.makedirs(checkpoint_path, exist_ok=True)

        dist_print('Saving a checkpoint...')

        if hvd.rank() == 0:
            self._save_mlp_checkpoint(checkpoint_path, self.bottom_mlp_layers, prefix='bottom_mlp')
            self._save_mlp_checkpoint(checkpoint_path, self.top_mlp, prefix='top_mlp')

        distributed_embedding = self.distributed and self.columnwise_split
        self.embedding.save_checkpoint(checkpoint_path,
                                       distributed=distributed_embedding)

        dist_print('Saved a checkpoint to ', checkpoint_path)

    def restore_checkpoint_if_path_exists(self, checkpoint_path):
        if checkpoint_path is None:
            return self

        dist_print('Restoring a checkpoint...')
        self.force_initialization()

        if self.running_bottom_mlp:
            self._restore_mlp_checkpoint(checkpoint_path, self.bottom_mlp_layers, prefix='bottom_mlp')
        self._restore_mlp_checkpoint(checkpoint_path, self.top_mlp, prefix='top_mlp')

        distributed_embedding = self.distributed and self.columnwise_split
        self.embedding.restore_checkpoint(checkpoint_path, distributed=distributed_embedding)

        dist_print('Restored a checkpoint from', checkpoint_path)
        return self

    def save_model_if_path_exists(self, path, save_input_signature=False):
        if not path:
            return

        if hvd.size() > 1:
            raise ValueError('SavedModel conversion not supported in HybridParallel mode')

        if save_input_signature:
            input_sig = model_input_signature(self, keep_original_batch_size=True)
            call_graph = tf.function(self)
            signatures = call_graph.get_concrete_function(input_sig[0])
        else:
            signatures = None

        options = tf.saved_model.SaveOptions(
            experimental_variable_policy=tf.saved_model.experimental.VariablePolicy.SAVE_VARIABLE_DEVICES)

        tf.keras.models.save_model(
            model=self,
            filepath=path,
            overwrite=True,
            signatures=signatures,
            options=options)

    @staticmethod
    def load_model_if_path_exists(path):
        if not path:
            return None

        if hvd.size() > 1:
            raise ValueError('Loading a SavedModel not supported in HybridParallel mode')

        print('Loading a saved model from', path)

        loaded = tf.keras.models.load_model(path)
        return loaded


# dummy model for profiling and debugging
class DummyDlrm(tf.keras.Model):
    def __init__(self, FLAGS, dataset_metadata):
        super(DummyDlrm, self).__init__()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid',
                                           kernel_initializer='glorot_normal',
                                           bias_initializer=initializers.RandomNormal(stddev=math.sqrt(1. / 1))
                                           )
        self.dataset_metadata = dataset_metadata
        self.top_variables = [v for v in self.trainable_variables if 'model_parallel' not in v.name]
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
        self.bottom_variables = [v for v in self.trainable_variables if 'model_parallel' in v.name]
        self.bottom_variable_indices = [i for i,v in enumerate(self.trainable_variables) if 'model_parallel' in v.name]

        self.top_variables = [v for v in self.trainable_variables if 'model_parallel' not in v.name]
        self.top_variable_indices = [i for i, v in enumerate(self.trainable_variables) if 'model_parallel' not in v.name]
        self.variables_partitioned = True

    def extract_bottom_gradients(self, all_gradients):
        if not self.variables_partitioned:
            self._partition_variables()
        return [all_gradients[i] for i in self.bottom_variable_indices]

    def extract_top_gradients(self, all_gradients):
        if not self.variables_partitioned:
            self._partition_variables()
        return [all_gradients[i] for i in self.top_variable_indices]
