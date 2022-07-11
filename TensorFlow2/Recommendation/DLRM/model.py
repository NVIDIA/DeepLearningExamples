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
from embedding import DualEmbeddingGroup, EmbeddingInitializer
from distributed_embeddings.python.layers import embedding
import interaction
import tensorflow.keras.initializers as initializers
import math
import horovod.tensorflow as hvd
import numpy as np
import time
import os
from utils import dist_print, get_variable_path
from tensorflow.python.keras.saving.saving_utils import model_input_signature
from collections import OrderedDict
from distributed_embeddings.python.layers import dist_model_parallel as dmp

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


class DlrmTrainer:
    def __init__(self, dlrm, embedding_optimizer, mlp_optimizer, amp, lr_scheduler, pipe, cpu):
        self.dlrm = dlrm
        self.embedding_optimizer = embedding_optimizer
        self.mlp_optimizer = mlp_optimizer
        self.amp = amp
        self.lr_scheduler = lr_scheduler
        self.bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                      from_logits=True)
        self.cpu = cpu
        self.pipe = iter(pipe.op())

    @tf.function
    def train_step(self):
        device = '/CPU:0' if self.cpu else '/GPU:0'
        with tf.device(device):
            self.lr_scheduler()
            with tf.name_scope("dataloading"):
                (numerical_features, categorical_features), labels = self.pipe.get_next()

            inputs = _create_inputs_dict(numerical_features, categorical_features)
            with tf.GradientTape() as tape:
                predictions = self.dlrm(inputs=inputs, training=True)
                unscaled_loss = self.bce(labels, predictions)
                # tf keras doesn't reduce the loss when using a Custom Training Loop
                unscaled_loss = tf.math.reduce_mean(unscaled_loss)
                scaled_loss = self.mlp_optimizer.get_scaled_loss(unscaled_loss) if self.amp else unscaled_loss

            if hvd.size() > 1:
                tape = dmp.DistributedGradientTape(tape)
            gradients = tape.gradient(scaled_loss, self.dlrm.trainable_variables)
            
            if self.amp:
                gradients = self.mlp_optimizer.get_unscaled_gradients(gradients)

            self.mlp_optimizer.apply_gradients(zip(gradients, self.dlrm.trainable_variables))
            if hvd.size() > 1:
                # compute mean loss for all workers for reporting
                mean_loss = hvd.allreduce(unscaled_loss, name="mean_loss", op=hvd.Average)
            else:
                mean_loss = unscaled_loss

            return mean_loss


def evaluate(validation_pipeline, dlrm, timer, auc_thresholds, max_steps=None, cast_dtype=None):

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
            labels = hvd.allgather(labels)

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
    def __init__(self, FLAGS, dataset_metadata):
        super(Dlrm, self).__init__()

        self.global_table_sizes = dataset_metadata.categorical_cardinalities

        self.distributed = hvd.size() > 1
        self.batch_size = FLAGS.batch_size
        self.num_all_categorical_features = len(dataset_metadata.categorical_cardinalities)

        self.amp = FLAGS.amp
        self.fp16 = FLAGS.fp16

        self.use_merlin_de_embeddings = FLAGS.use_merlin_de_embeddings

        self.dataset_metadata = dataset_metadata

        self.embedding_dim = FLAGS.embedding_dim
        self.column_slice_threshold = FLAGS.column_slice_threshold

        self.dot_interaction = FLAGS.dot_interaction
        if FLAGS.dot_interaction == 'custom_cuda':
            self.interact_op = dot_based_interact_ops.dot_based_interact
        elif FLAGS.dot_interaction == 'tensorflow':
            self.interact_op =  interaction.dot_interact
        elif FLAGS.dot_interaction == 'dummy':
            self.interact_op = interaction.dummy_dot_interact
        else:
            raise ValueError(f'Unknown dot-interaction implementation {FLAGS.dot_interaction}')

        self.embedding_trainable = FLAGS.embedding_trainable

        self.bottom_mlp_dims = [int(d) for d in FLAGS.bottom_mlp_dims]
        self.top_mlp_dims = [int(d) for d in FLAGS.top_mlp_dims]

        self.num_numerical_features = dataset_metadata.num_numerical_features

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

            kernel_initializer = initializers.GlorotNormal()
            bias_initializer = initializers.RandomNormal(stddev=math.sqrt(1. / dim))

            l = tf.keras.layers.Dense(dim, activation=activation,
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer)
            self.top_mlp.append(l)

    def _create_embeddings(self):
        if self.distributed:
            self.embedding_layers = []
            for table_size in self.global_table_sizes:
                if self.use_merlin_de_embeddings:
                    e = embedding.Embedding(input_dim=table_size,
                                            output_dim=self.embedding_dim,
                                            embeddings_initializer=EmbeddingInitializer())
                else:
                    e = tf.keras.layers.Embedding(input_dim=table_size,
                                                  output_dim=self.embedding_dim,
                                                  embeddings_initializer=EmbeddingInitializer())
                self.embedding_layers.append(e)

            self.embedding = dmp.DistributedEmbedding(self.embedding_layers,
                                                      strategy='memory_balanced',
                                                      dp_input=False,
                                                      column_slice_threshold=self.column_slice_threshold)

            self.local_table_ids = self.embedding.strategy.input_ids_list[hvd.rank()]
        else:
            self.local_table_ids = list(range(len(self.global_table_sizes)))
            feature_names = [f'feature_{i}' for i in self.local_table_ids]

            self.embedding = DualEmbeddingGroup(cardinalities=self.global_table_sizes, output_dim=self.embedding_dim,
                                                memory_threshold=70,
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

    def force_initialization(self):
        numerical_features = tf.zeros(shape=[self.batch_size // hvd.size(),
                                      self.dataset_metadata.num_numerical_features])

        categorical_features = tf.zeros(shape=[self.batch_size, len(self.local_table_ids)], dtype=tf.int32)
        inputs = _create_inputs_dict(numerical_features, categorical_features)
        self(inputs=inputs, training=True)

    @tf.function
    def call(self, inputs, sigmoid=False, training=False):
        vals = list(inputs.values())
        numerical_features, cat_features = vals[0], vals[1]

        cat_features = tf.split(cat_features, num_or_size_splits=len(self.local_table_ids), axis=1)
        cat_features = [tf.reshape(f, shape=[self.batch_size]) for f in cat_features]

        embedding_outputs = self._call_embeddings(cat_features, training=training)

        bottom_mlp_out = self._call_bottom_mlp(numerical_features, training=training)
        bottom_part_output = tf.concat([bottom_mlp_out, embedding_outputs], axis=1)

        num_categorical_features = len(self.dataset_metadata.categorical_cardinalities)
        interaction_input = tf.reshape(bottom_part_output,
                                       [-1, num_categorical_features + 1,
                                        self.embedding_dim])

        x = self.interact_op(interaction_input, tf.squeeze(bottom_mlp_out))
        x = self._call_top_mlp(x, training=training)

        if sigmoid:
            x = tf.math.sigmoid(x)

        x = tf.cast(x, tf.float32)
        return x

    def _call_bottom_mlp(self, numerical_features, training=False):
        if self.amp:
            numerical_features = tf.cast(numerical_features, dtype=tf.float16)

        if training:
            batch_size = self.batch_size // hvd.size()
        else:
            batch_size = tf.shape(numerical_features)[0]

        padding = self._get_bottom_mlp_padding(batch_size=batch_size)
        x = tf.concat([numerical_features, padding], axis=1)

        with tf.name_scope('bottom_mlp'):
            for l in self.bottom_mlp_layers:
                x = l(x)
            x = tf.expand_dims(x, axis=1)
            bottom_mlp_out = x
        return bottom_mlp_out

    def _call_embeddings(self, cat_features, training=False):
        x = self.embedding(cat_features)
        if self.distributed:
            x = tf.concat([tf.expand_dims(z, axis=1) for z in x], axis=1)

        if self.amp:
            x = tf.cast(x, dtype=tf.float16)
        return x

    def _call_top_mlp(self, x, training=False):
        if self.dot_interaction != 'custom_cuda':
             batch_size = self.batch_size // hvd.size() if training else tf.shape(x)[0]
             padding = self._get_top_mlp_padding(batch_size=batch_size)
             x = tf.concat([x, padding], axis=1)

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

    @staticmethod
    def _save_embeddings_checkpoint(checkpoint_path, embedding_weights):
        for i, weight in enumerate(embedding_weights):
            filename = get_variable_path(checkpoint_path, f'feature_{i}')
            np.save(file=filename, arr=weight)

    @staticmethod
    def _restore_weights_checkpoint(checkpoint_path, num_weights, name):
        result = []
        for i in range(num_weights):
            filename = os.path.join(checkpoint_path, f'{name}_{i}.npy')
            print(f'loading: {name}_{i} from {filename}')
            result.append(np.load(file=filename))
        return result


    def save_checkpoint_if_path_exists(self, checkpoint_path):
        if checkpoint_path is None:
            return

        begin_save = time.time()
        os.makedirs(checkpoint_path, exist_ok=True)

        dist_print('Saving a checkpoint...')

        if hvd.rank() == 0:
            self._save_mlp_checkpoint(checkpoint_path, self.bottom_mlp_layers, prefix='bottom_mlp')
            self._save_mlp_checkpoint(checkpoint_path, self.top_mlp, prefix='top_mlp')

        begin = time.time()
        full_embedding_weights = self.embedding.get_weights()
        end = time.time()
        print(f'get weights took: {end - begin:.3f} seconds')

        if hvd.rank() == 0 and self.distributed:
            self._save_embeddings_checkpoint(checkpoint_path, full_embedding_weights)
        elif not self.distributed:
            self.embedding.save_checkpoint(checkpoint_path=checkpoint_path)

        end_save = time.time()
        dist_print('Saved a checkpoint to ', checkpoint_path)
        dist_print(f'Saving a checkpoint took {end_save - begin_save:.3f}')

    def restore_checkpoint_if_path_exists(self, checkpoint_path):
        begin = time.time()
        if checkpoint_path is None:
            return self

        dist_print('Restoring a checkpoint...')
        self.force_initialization()

        self._restore_mlp_checkpoint(checkpoint_path, self.bottom_mlp_layers, prefix='bottom_mlp')
        self._restore_mlp_checkpoint(checkpoint_path, self.top_mlp, prefix='top_mlp')

        paths = []
        for i in range(self.num_all_categorical_features):
            path = get_variable_path(checkpoint_path, f'feature_{i}')
            paths.append(path)

        if self.distributed:
            self.embedding.set_weights(weights=paths)
        else:
            self.embedding.restore_checkpoint(checkpoint_path=checkpoint_path)

        end = time.time()
        print('Restored a checkpoint from', checkpoint_path)
        print(f'Restoring a checkpoint took: {end-begin:.3f} seconds')
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
