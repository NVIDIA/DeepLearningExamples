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

from __future__ import print_function

import os
import math

import tensorflow as tf
import horovod.tensorflow as hvd

from model import efficientnet_model

from utils import dataset_factory, hvd_utils, callbacks, preprocessing

__all__ = ['get_optimizer_params', 'get_metrics', 'get_learning_rate_params', 'build_model_params', 'get_models', 'build_augmenter_params', \
            'get_image_size_from_model', 'get_dataset_builders', 'build_stats', 'parse_inference_input', 'preprocess_image_files']

def get_optimizer_params(name,
        decay,
        epsilon,
        momentum,
        moving_average_decay,
        nesterov,
        beta_1,
        beta_2):
    return {
        'name': name,
        'decay': decay,
        'epsilon': epsilon,
        'momentum': momentum,
        'moving_average_decay': moving_average_decay,
        'nesterov': nesterov,
        'beta_1': beta_1,
        'beta_2': beta_2
    }


def get_metrics(one_hot: bool):
    """Get a dict of available metrics to track."""
    if one_hot:
        return {
            # (name, metric_fn)
            'acc': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            'accuracy': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            'top_1': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            'top_5': tf.keras.metrics.TopKCategoricalAccuracy(
                k=5,
                name='top_5_accuracy'),
        }
    else:
        return {
            # (name, metric_fn)
            'acc': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            'top_1': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            'top_5': tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=5,
                name='top_5_accuracy'),
        }

def get_learning_rate_params(name,
        initial_lr,
        decay_epochs,
        decay_rate,
        warmup_epochs):
    return {
        'name':name,
        'initial_lr': initial_lr,
        'decay_epochs': decay_epochs,
        'decay_rate': decay_rate,
        'warmup_epochs': warmup_epochs,
        'examples_per_epoch': None,
        'boundaries': None,
        'multipliers': None,
        'scale_by_batch_size': 1./128.,
        'staircase': True
    }


def build_model_params(model_name, is_training, batch_norm, num_classes, activation, dtype, weight_decay, weight_init):
    return {
        'model_name': model_name,
        'model_weights_path': '',
        'weights_format': 'saved_model',
        'overrides': {
            'is_training': is_training,
            'batch_norm': batch_norm,
            'rescale_input': True,
            'num_classes': num_classes,
            'weight_decay': weight_decay,
            'activation': activation,
            'dtype': dtype,
            'weight_init': weight_init
        }
    }

def get_models():
    """Returns the mapping from model type name to Keras model."""
    return  {
        'efficientnet': efficientnet_model.EfficientNet.from_name,
    }

def build_augmenter_params(augmenter_name, cutout_const, translate_const, num_layers, magnitude, autoaugmentation_name):
    if augmenter_name is None or augmenter_name not in ['randaugment', 'autoaugment']:
        return {}
    augmenter_params = {}
    if cutout_const is not None:
        augmenter_params['cutout_const'] = cutout_const
    if translate_const is not None:
        augmenter_params['translate_const'] = translate_const
    if augmenter_name == 'randaugment':
        if num_layers is not None:
            augmenter_params['num_layers'] = num_layers
        if magnitude is not None:
            augmenter_params['magnitude'] = magnitude
    if augmenter_name == 'autoaugment':
        if autoaugmentation_name is not None:
            augmenter_params['autoaugmentation_name'] = autoaugmentation_name

    return augmenter_params

def get_image_size_from_model(arch):
    """If the given model has a preferred image size, return it."""
    if 'efficientnet' in arch:
        efficientnet_name = arch
        if efficientnet_name in efficientnet_model.MODEL_CONFIGS:
            return efficientnet_model.MODEL_CONFIGS[efficientnet_name]['resolution']
    return None

def get_dataset_builders(params, one_hot):
    """Create and return train and validation dataset builders."""
    if hvd.size() > 1:
        num_gpus = hvd.size()
    else:
        num_devices = 1

    image_size = get_image_size_from_model(params.arch)
    print("Image size {}".format(image_size))
    print("Train batch size {}".format(params.train_batch_size))
    builders = []
    validation_dataset_builder = None
    train_dataset_builder = None
    if "train" in params.mode:
        train_dataset_builder = dataset_factory.Dataset(data_dir=params.data_dir,
        index_file_dir=params.index_file,
        split='train',
        num_classes=params.num_classes,
        image_size=image_size,
        batch_size=params.train_batch_size,
        one_hot=one_hot,
        use_dali=params.use_dali,
        augmenter=params.augmenter_name,
        augmenter_params=build_augmenter_params(params.augmenter_name, 
            params.cutout_const, 
            params.translate_const, 
            params.num_layers, 
            params.magnitude, 
            params.autoaugmentation_name),
        mixup_alpha=params.mixup_alpha
        )
    if "eval" in params.mode:
        validation_dataset_builder = dataset_factory.Dataset(data_dir=params.data_dir,
        index_file_dir=params.index_file,
        split='validation',
        num_classes=params.num_classes,
        image_size=image_size,
        batch_size=params.eval_batch_size,
        one_hot=one_hot,
        use_dali=params.use_dali_eval)

    builders.append(train_dataset_builder)
    builders.append(validation_dataset_builder)

    return builders

def build_stats(history, validation_output, train_callbacks, eval_callback, logger):
    stats = {}
    if validation_output:
        stats['eval_loss'] = float(validation_output[0])
        stats['eval_accuracy_top_1'] = float(validation_output[1])
        stats['eval_accuracy_top_5'] = float(validation_output[2])
    #This part is train loss on GPU_0
    if history and history.history:
        train_hist = history.history
        #Gets final loss from training.
        stats['training_loss'] = float(hvd.allreduce(tf.constant(train_hist['loss'][-1], dtype=tf.float32), average=True))
        # Gets top_1 training accuracy.
        if 'categorical_accuracy' in train_hist:
            stats['training_accuracy_top_1'] = float(hvd.allreduce(tf.constant(train_hist['categorical_accuracy'][-1], dtype=tf.float32), average=True))
        elif 'sparse_categorical_accuracy' in train_hist:
            stats['training_accuracy_top_1'] = float(hvd.allreduce(tf.constant(train_hist['sparse_categorical_accuracy'][-1], dtype=tf.float32), average=True))
        elif 'accuracy' in train_hist:
            stats['training_accuracy_top_1'] = float(hvd.allreduce(tf.constant(train_hist['accuracy'][-1], dtype=tf.float32), average=True))
            stats['training_accuracy_top_5'] = float(hvd.allreduce(tf.constant(train_hist['top_5_accuracy'][-1], dtype=tf.float32), average=True))

    # Look for the time history callback which was used during keras.fit
    if train_callbacks:
        for callback in train_callbacks:
            if isinstance(callback, callbacks.TimeHistory):
                if callback.epoch_runtime_log:
                    stats['avg_exp_per_second_training'] = callback.average_examples_per_second
                    stats['avg_exp_per_second_training_per_GPU'] = callback.average_examples_per_second / hvd.size()

    if eval_callback:
        stats['avg_exp_per_second_eval'] = float(eval_callback.average_examples_per_second) * hvd.size()
        stats['avg_exp_per_second_eval_per_GPU'] = float(eval_callback.average_examples_per_second)
        stats['avg_time_per_exp_eval'] = 1000./stats['avg_exp_per_second_eval']
        batch_time = eval_callback.batch_time
        batch_time.sort()
        latency_pct_per_batch = sum( batch_time[:-1] ) / int( len(batch_time) - 1 )
        stats['latency_pct'] = 1000.0 * latency_pct_per_batch
        latency_90pct_per_batch = sum( batch_time[:int( 0.9 * len(batch_time) )] ) / int( 0.9 * len(batch_time) )
        stats['latency_90pct'] = 1000.0 * latency_90pct_per_batch
        latency_95pct_per_batch = sum( batch_time[:int( 0.95 * len(batch_time) )] ) / int( 0.95 * len(batch_time) )
        stats['latency_95pct'] = 1000.0 * latency_95pct_per_batch
        latency_99pct_per_batch = sum( batch_time[:int( 0.99 * len(batch_time) )] ) / int( 0.99 * len(batch_time) )
        stats['latency_99pct'] = 1000.0 * latency_99pct_per_batch

    if not hvd_utils.is_using_hvd() or hvd.rank() == 0:
        logger.log(step=(), data=stats)


def preprocess_image_files(directory_name, arch, batch_size, num_channels=3, dtype=tf.float32):
    image_size = get_image_size_from_model(arch)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(data_format="channels_last")
    images = datagen.flow_from_directory(directory_name, class_mode=None, batch_size=batch_size, target_size=(image_size, image_size), shuffle=False)
    return images


def parse_inference_input(to_predict):
    
    filenames = []
    
    image_formats = ['.jpg', '.jpeg', '.JPEG', '.JPG', '.png', '.PNG']
    
    if os.path.isdir(to_predict):
        filenames = [f for f in os.listdir(to_predict) 
                     if os.path.isfile(os.path.join(to_predict, f)) 
                     and os.path.splitext(f)[1] in image_formats]
        
    elif os.path.isfile(to_predict):
        filenames.append(to_predict)
      
    return filenames
