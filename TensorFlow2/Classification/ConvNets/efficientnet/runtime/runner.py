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
import multiprocessing
import warnings
import yaml
import time

import tensorflow as tf
import numpy as np

import horovod.tensorflow.keras as hvd

from utils import hvd_utils, optimizer_factory
from utils import callbacks as custom_callbacks

from runtime.runner_utils import get_optimizer_params, get_metrics, get_learning_rate_params, \
                        build_model_params, get_models, get_dataset_builders, build_stats, \
                        parse_inference_input, preprocess_image_files

__all__ = [
    'Runner',
]

DTYPE_MAP = {
    'float32': tf.float32,
    'bfloat16': tf.bfloat16,
    'float16': tf.float16,
    'fp32': tf.float32,
    'bf16': tf.bfloat16,
}

class Runner(object):

    def __init__(self, flags, logger):

        self.params = flags
        self.logger = logger

        if hvd.rank() == 0:
            self.serialize_config(model_dir=self.params.model_dir)

        # =================================================
        # Define Datasets
        # =================================================
        label_smoothing = flags.label_smoothing
        self.one_hot = label_smoothing and label_smoothing > 0

        builders = get_dataset_builders(self.params, self.one_hot)
        datasets = [builder.build() if builder else None for builder in builders]

        self.train_dataset, self.validation_dataset = datasets
        self.train_builder, self.validation_builder = builders

        self.initialize()

        # =================================================
        # Define Model
        # =================================================
        model_params = build_model_params(model_name=self.params.arch,
            is_training="predict" not in self.params.mode,
            batch_norm=self.params.batch_norm,
            num_classes=self.params.num_classes,
            activation=self.params.activation,
            dtype=DTYPE_MAP[self.params.dtype],
            weight_decay=self.params.weight_decay,
            weight_init=self.params.weight_init
            )
        models_dict = get_models()
        self.model = [model for model_name, model in models_dict.items() if model_name in self.params.arch][0](**model_params)
        
        self.metrics = ['accuracy', 'top_5']

        if self.params.dataset == 'ImageNet':
            self.train_num_examples = 1281167
            self.eval_num_examples = 50000

    def initialize(self):
        """Initializes backend related initializations."""
        if tf.config.list_physical_devices('GPU'):
            data_format = 'channels_first'
        else:
            data_format = 'channels_last'
        tf.keras.backend.set_image_data_format(data_format)
        if self.params.run_eagerly:
            # Enable eager execution to allow step-by-step debugging
            tf.config.experimental_run_functions_eagerly(True)


    def load_model_weights(self, model_dir):
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        if not latest_checkpoint:
            return 0

        self.model.load_weights(latest_checkpoint)
        return self.model.optimizer.iterations

    def resume_from_checkpoint(self,
                           model_dir: str,
                           train_steps: int) -> int:
        """Resumes from the latest checkpoint, if possible.

        Loads the model weights and optimizer settings from a checkpoint.
        This function should be used in case of preemption recovery.

        Args:
        model: The model whose weights should be restored.
        model_dir: The directory where model weights were saved.
        train_steps: The number of steps to train.

        Returns:
        The epoch of the latest checkpoint, or 0 if not restoring.

        """
        last_iteration = self.load_model_weights(model_dir)
        initial_epoch = last_iteration // train_steps
        return int(initial_epoch)


    def serialize_config(self, model_dir: str):
        """Serializes and saves the experiment config."""
        params_save_path = os.path.join(model_dir, 'params.yaml')
        with open(params_save_path, 'w') as outfile:
            yaml.dump(vars(self.params), outfile, default_flow_style=False)


    def train(self):
        train_epochs = self.params.max_epochs
        train_steps = self.params.steps_per_epoch if self.params.steps_per_epoch is not None else self.train_num_examples // self.train_builder.global_batch_size
        if self.validation_builder is not None:
            validation_steps = self.eval_num_examples // self.validation_builder.global_batch_size
        else:
            validation_steps = None

        learning_rate = optimizer_factory.build_learning_rate(
            params=get_learning_rate_params(name=self.params.lr_decay,
                initial_lr=self.params.lr_init,
                decay_epochs=self.params.lr_decay_epochs,
                decay_rate=self.params.lr_decay_rate,
                warmup_epochs=self.params.lr_warmup_epochs),
            batch_size=self.train_builder.global_batch_size,
            train_steps=train_steps,
            max_epochs=train_epochs)
        optimizer = optimizer_factory.build_optimizer(
            optimizer_name=self.params.optimizer,
            base_learning_rate=learning_rate,
            params=get_optimizer_params(name=self.params.optimizer,
                decay=self.params.decay,
                epsilon=self.params.epsilon,
                momentum=self.params.momentum,
                moving_average_decay=self.params.moving_average_decay,
                nesterov=self.params.nesterov,
                beta_1=self.params.beta_1,
                beta_2=self.params.beta_2)
            )

        metrics_map = get_metrics(self.one_hot)
        metrics = [metrics_map[metric] for metric in self.metrics]

        optimizer = hvd.DistributedOptimizer(optimizer, compression=hvd.Compression.fp16)
        
        if self.one_hot:
            loss_obj = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=self.params.label_smoothing)
        else:
            loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

        # Training 
        self.model.compile(optimizer=optimizer, 
        loss=loss_obj,
        metrics=metrics,
        experimental_run_tf_function=False)

        initial_epoch = 0
        if self.params.resume_checkpoint:
            initial_epoch = self.resume_from_checkpoint(model_dir=self.params.model_dir,
                                                train_steps=train_steps)
        
        #Define Callbacks (TODO)
        callbacks=[hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
        callbacks += custom_callbacks.get_callbacks(
            model_checkpoint=self.params.enable_checkpoint_and_export,
            include_tensorboard=self.params.enable_tensorboard,
            time_history=self.params.time_history,
            track_lr=True,
            write_model_weights=self.params.write_model_weights,
            initial_step=initial_epoch * train_steps,
            batch_size=self.train_builder.global_batch_size,
            log_steps=self.params.log_steps,
            model_dir=self.params.model_dir,
            save_checkpoint_freq=train_steps * self.params.save_checkpoint_freq,
            logger=self.logger)

        if "eval" not in self.params.mode:
            validation_kwargs = {}
        else:
            validation_kwargs = {
                'validation_data': self.validation_dataset,
                'validation_steps': validation_steps,
                'validation_freq': self.params.num_epochs_between_eval,
                }

        history = self.model.fit(
            self.train_dataset,
            epochs=train_epochs,
            steps_per_epoch=train_steps,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=2,
            **validation_kwargs)

        validation_output = None
        eval_callback = None
        if not self.params.skip_eval and self.validation_builder is not None:
            eval_callback = custom_callbacks.EvalTimeHistory(batch_size=self.params.eval_batch_size, logger=self.logger)
            worker_validation_output = self.model.evaluate(
                self.validation_dataset, steps=validation_steps, callbacks=eval_callback, verbose=2)
            validation_output = list(hvd.allreduce(worker_validation_output,average=True))

        build_stats(history, validation_output, callbacks, eval_callback, self.logger)


    def evaluate(self):

        if self.validation_builder is not None:
            validation_steps = self.eval_num_examples // self.validation_builder.global_batch_size
        else:
            validation_steps = None

        metrics_map = get_metrics(self.one_hot)
        metrics = [metrics_map[metric] for metric in self.metrics]
        
        if self.one_hot:
            loss_obj = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=self.params.label_smoothing)
        else:
            loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

        # Training 
        self.model.compile(optimizer="rmsprop", 
        loss=loss_obj,
        metrics=metrics,
        experimental_run_tf_function=False)

        _ = self.load_model_weights(self.params.model_dir)
        eval_callback = custom_callbacks.EvalTimeHistory(batch_size=self.params.eval_batch_size, logger=self.logger)
        results = self.model.evaluate(self.validation_dataset, steps=validation_steps, callbacks=eval_callback, verbose=1)
        build_stats(None, results, None, eval_callback, self.logger)


    def predict(self, to_predict, checkpoint_name=None, print_results=True):

        images = preprocess_image_files(directory_name=to_predict, arch=self.params.arch, batch_size=self.params.predict_batch_size, dtype=DTYPE_MAP[self.params.dtype])
        nb_samples = len(images)
        if checkpoint_name is not None:
            self.model.load_weights(checkpoint_name)
        try:
            file_names = images.filenames
            num_files = len(file_names)
            if self.params.benchmark:
                nb_samples *= 50
                print_results = False
                num_files *= 50
            start_time = time.time()
            inference_results = self.model.predict(images, verbose=1, steps=nb_samples)
            total_time = time.time() - start_time
            score = tf.nn.softmax(inference_results, axis=1)
            
            if print_results:
                for i, name in enumerate(file_names):
                    print(
                        "This {} image most likely belongs to {} class with a {} percent confidence."
                        .format(name, tf.math.argmax(score[i]), 100 * tf.math.reduce_max(score[i]))
                    )
            print("Total time to infer {} images :: {}".format(num_files, total_time))
            print("Inference Throughput {}".format(num_files/total_time))
            print("Inference Latency {}".format(total_time/num_files))

        except KeyboardInterrupt:
            print("Keyboard interrupt")

        print('Ending Inference ...')
