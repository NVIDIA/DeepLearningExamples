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
from tensorflow.python.ops.gen_array_ops import deep_copy
import yaml
import time
import tensorflow as tf
import numpy as np
import horovod.tensorflow.keras as hvd
import tensorflow_addons as tfa

from tensorflow.python.platform import gfile
from importlib import import_module
from utils import hvd_utils, optimizer_factory
from utils import callbacks as custom_callbacks
from runtime.runner_utils import get_optimizer_params
from runtime.runner_utils import get_metrics
from runtime.runner_utils import get_learning_rate_params
from runtime.runner_utils import get_dataset_builders
from runtime.runner_utils import build_stats
from runtime.runner_utils import preprocess_image_files
from utils.tf_utils import get_num_flops
from utils.tf_utils import get_num_params
from runtime.runner_utils import train_step        

__all__ = [
    'Runner',
]


class Runner(object):

    def __init__(self, flags, logger):

        self.params = flags
        self.logger = logger

        if hvd.rank() == 0:
            self.serialize_config()
            
                

        self.one_hot = self.params.label_smoothing and self.params.label_smoothing > 0
        self.initialize()
        self.metrics = self.params.metrics
        
        # =================================================
        # Define Model
        # =================================================
        Model = import_module(self.params.mparams.path_to_impl).Model
        # we use custom train_step if gradient accumulation is requested
        if self.params.grad_accum_steps > 1:
            Model.train_step = train_step # monkey patching
        self.model = Model(self.params)

        try:
            # complexity analysis
            img_size = 171
            params = get_num_params(self.model)
            flops = get_num_flops(self.model,(img_size, img_size, 3))
            print('''
                # ======================================================#
                # #params: {:.4f}M    #Flops: {:.4f}G [{}x{}x3]
                # ======================================================#
                  '''.format(params,flops, img_size,img_size))
        except:
            print('''
                # ======================================================#
                # Skip complexity analysis
                # ======================================================#
                  ''')
        

    def initialize(self):
        """Initializes backend related initializations."""
        if self.params.data_format:
            tf.keras.backend.set_image_data_format(self.params.data_format)
        if self.params.run_eagerly:
            # Enable eager execution to allow step-by-step debugging
            tf.config.experimental_run_functions_eagerly(True)


    def load_model_weights(self, model_dir_or_fn, expect_partial=False):
        """
         Resumes from the latest checkpoint, if possible.

        Loads the model weights and optimizer settings from a checkpoint.
        This function should be used in case of preemption recovery.

        Args:
        model_dir: The directory where model weights were saved.

        Returns:
        The iteration of the latest checkpoint, or 0 if not restoring.

        """

        if gfile.IsDirectory(model_dir_or_fn):
            latest_checkpoint = tf.train.latest_checkpoint(model_dir_or_fn)
            if not latest_checkpoint:
                return 0
        else:
            latest_checkpoint = model_dir_or_fn
        
        if expect_partial:
            self.model.load_weights(latest_checkpoint).expect_partial()
        else:
            self.model.load_weights(latest_checkpoint)

        if self.model.optimizer:
            return int(self.model.optimizer.iterations)
        else:
            # optimizer has not been compiled (predict mode)
            return 0
        
        
    def serialize_config(self):
        """Serializes and saves the experiment config."""
        save_dir=self.params.log_dir if self.params.log_dir is not None else self.params.model_dir
        mode=self.params.mode
        if mode in ["train", "train_and_eval", "training_benchmark"]:
            params_save_path = os.path.join(save_dir, 'params.yaml')
        else:
            # to avoid overwriting the training config file that may exist in the same dir
            params_save_path = os.path.join(save_dir, 'eval_params.yaml')
        self.params.save_to_yaml(params_save_path)


    def train(self):
        
        # get global batch size, #epochs, and equivalent #steps
        global_batch_size_tr = self.params.train_batch_size * hvd.size() * self.params.grad_accum_steps
        train_epochs = self.params.max_epochs
        assert train_epochs >= self.params.n_stages, "each training stage requires at least 1 training epoch"
        train_steps = self.params.steps_per_epoch if self.params.steps_per_epoch  else self.params.train_num_examples // global_batch_size_tr * self.params.grad_accum_steps
        train_iterations = train_steps // self.params.grad_accum_steps
        
        global_batch_size_eval = self.params.eval_batch_size * hvd.size()
        validation_steps = self.params.eval_num_examples //global_batch_size_eval if "eval" in self.params.mode else None
        
        # set up lr schedule
        learning_rate = optimizer_factory.build_learning_rate(
            params=get_learning_rate_params(name=self.params.lr_decay,
                initial_lr=self.params.lr_init,
                decay_epochs=self.params.lr_decay_epochs,
                decay_rate=self.params.lr_decay_rate,
                warmup_epochs=self.params.lr_warmup_epochs),
            batch_size=global_batch_size_tr, # updates are iteration based not batch-index based
            train_steps=train_iterations,
            max_epochs=train_epochs)
        
        # set up optimizer
        optimizer = optimizer_factory.build_optimizer(
            optimizer_name=self.params.optimizer,
            base_learning_rate=learning_rate,
            params=get_optimizer_params(name=self.params.optimizer,
                decay=self.params.decay,
                epsilon=self.params.opt_epsilon,
                momentum=self.params.momentum,
                moving_average_decay=self.params.moving_average_decay,
                nesterov=self.params.nesterov,
                beta_1=self.params.beta_1,
                beta_2=self.params.beta_2)
            )
        
        if self.params.grad_accum_steps > 1:
            # we use custom train_step when self.params.grad_accum_steps > 1
            if self.params.use_amp:
                # in which case we must manually wrap the optimizer with LossScaleOptimizer
                optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")
            
            # Horovod allreduce across devices takes place in the custom train_step
        else:
            # Without custom train_step, AMP optimizer is automatically taken care of. 
            # Without custom train_step, we need to wrap the optimizer to enable Horovod
            optimizer = hvd.DistributedOptimizer(optimizer, compression=hvd.Compression.fp16 if self.params.hvd_fp16_compression else hvd.Compression.none)
        
        # define metrics depending on target labels (1-hot vs. sparse)
        metrics_map = get_metrics(self.one_hot)
        metrics = [metrics_map[metric] for metric in self.metrics]

        
        # define loss functions depending on target labels (1-hot vs. sparse)
        if self.one_hot:
            loss_obj = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=self.params.label_smoothing)
        else:
            loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

        # model compilation 
        self.model.compile(optimizer=optimizer, 
        loss=loss_obj,
        metrics=metrics,
        run_eagerly=self.params.run_eagerly,
        )
       
      
        resumed_iterations = resumed_epoch = 0
        if self.params.resume_checkpoint:
            print('# ==================MODEL LOADING BEGINS=====================#')
            resumed_iterations = self.load_model_weights(self.params.model_dir)
            resumed_epoch = resumed_iterations // train_iterations
            if resumed_iterations > 0:
                print('''
                # =======================================
                    ckpt at iteration {} loaded!
                # ======================================='''.format(resumed_iterations))
        
        #Define Callbacks (TODO)
        callbacks=[hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
        callbacks += custom_callbacks.get_callbacks(
            model_checkpoint=self.params.enable_checkpoint_saving,
            include_tensorboard=self.params.enable_tensorboard,
            time_history=self.params.time_history,
            track_lr=True,
            write_model_weights=self.params.tb_write_model_weights,
            initial_step=resumed_epoch * train_steps,
            batch_size=global_batch_size_tr / self.params.grad_accum_steps, # throughput calc: 1 batch is 1 step
            log_steps=self.params.log_steps,
            model_dir=self.params.model_dir,
            save_checkpoint_freq=train_steps * self.params.save_checkpoint_freq, # conditioned on batch index
            ema_decay=self.params.moving_average_decay,
            intratrain_eval_using_ema=self.params.intratrain_eval_using_ema,
            logger=self.logger)


        n_stages = self.params.n_stages
        if not n_stages or n_stages == 1:
            # =================================================
            # Define Datasets
            # =================================================
            builders = get_dataset_builders(self.params, self.one_hot)
            datasets = [builder.build() if builder else None for builder in builders]
            self.train_dataset, self.validation_dataset = datasets
            self.train_builder, self.validation_builder = builders
            
            # set model validation args
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
                initial_epoch=resumed_epoch,
                callbacks=callbacks,
                verbose=2,
                **validation_kwargs)
        else: # progressive training
            
            # determine regularization schedule 
            base_img_size=self.params.base_img_size
            base_mixup=self.params.base_mixup
            base_cutmix=self.params.base_cutmix
            base_randaug_mag=self.params.base_randaug_mag
            ram_list = np.linspace(base_randaug_mag, self.params.raug_magnitude, n_stages)
            mixup_list = np.linspace(base_mixup, self.params.mixup_alpha, n_stages)
            cutmix_list = np.linspace(base_cutmix, self.params.cutmix_alpha, n_stages)


            target_img_size = self.params.train_img_size
            epochs_per_stage = train_epochs // n_stages 
            resumed_stage = min(resumed_epoch // epochs_per_stage, n_stages-1)
            for stage in range(resumed_stage, n_stages):
                epoch_st = stage * epochs_per_stage 
                epoch_end = (epoch_st + epochs_per_stage) if stage < n_stages-1 else train_epochs
                epoch_curr = epoch_st if epoch_st >= resumed_epoch else resumed_epoch
                ratio = float(stage + 1) / float(n_stages)
                image_size = int(base_img_size + (target_img_size - base_img_size) * ratio)
                    
                
                
                # reassign new param vals
                self.params.raug_magnitude = ram_list[stage]
                self.params.mixup_alpha = mixup_list[stage]
                self.params.cutmix_alpha  = cutmix_list[stage]
                self.params.train_img_size = image_size
                
                # =================================================
                # Define Datasets
                # =================================================
                
                builders = get_dataset_builders(self.params, self.one_hot)
                datasets = [builder.build() if builder else None for builder in builders]
                self.train_dataset, self.validation_dataset = datasets
                self.train_builder, self.validation_builder = builders
                
                # set model validation args
                if "eval" not in self.params.mode:
                    validation_kwargs = {}
                else:
                    validation_kwargs = {
                        'validation_data': self.validation_dataset,
                        'validation_steps': validation_steps,
                        'validation_freq': self.params.num_epochs_between_eval,
                        }



                print('''
                # ===============================================
                    Training stage: {}
                    Epochs: {}-{}: starting at {}
                    batch size: {}
                    grad accum steps: {}
                    image size: {}
                    cutmix_alpha: {}
                    mixup_alpha:{}
                    raug_magnitude: {}
                # ==============================================='''.format(stage,
                                                                            epoch_st,
                                                                            epoch_end,
                                                                            epoch_curr,
                                                                            self.params.train_batch_size,
                                                                            self.params.grad_accum_steps,
                                                                            self.params.train_img_size,
                                                                            self.params.cutmix_alpha,
                                                                            self.params.mixup_alpha,
                                                                            self.params.raug_magnitude,
                                                                            ))

                history = self.model.fit(
                    self.train_dataset,
                    epochs=epoch_end,
                    steps_per_epoch=train_steps,
                    initial_epoch=epoch_curr,
                    callbacks=callbacks,
                    verbose=2,
                    **validation_kwargs)

        # we perform final evaluation using 1 GPU (hvd_size=1)  
        builders = get_dataset_builders(self.params, self.one_hot, hvd_size=1)
        datasets = [builder.build() if builder else None for builder in builders]
        _, self.validation_dataset = datasets
        _, self.validation_builder = builders
              
        validation_output = None
        eval_callbacks = []
        if not self.params.skip_eval and self.validation_builder is not None:
            eval_callbacks.append(custom_callbacks.EvalTimeHistory(batch_size=self.params.eval_batch_size, logger=self.logger))
            validation_output = self.model.evaluate(
                self.validation_dataset, callbacks=eval_callbacks, verbose=2)
            # stats are printed regardless of whether eval is requested or not
            build_stats(history, validation_output, callbacks, eval_callbacks, self.logger, comment="eval using original weights")
        else:
            build_stats(history, validation_output, callbacks, eval_callbacks, self.logger, comment="eval not requested")
                

        if self.params.moving_average_decay > 0:
            ema_validation_output = None
            eval_callbacks = []
            if not self.params.skip_eval and self.validation_builder is not None:
                eval_callbacks.append(custom_callbacks.EvalTimeHistory(batch_size=self.params.eval_batch_size, logger=self.logger))
                eval_callbacks.append(custom_callbacks.MovingAverageCallback(intratrain_eval_using_ema=True))
                ema_validation_output = self.model.evaluate(
                    self.validation_dataset, callbacks=eval_callbacks, verbose=2)
                # we print stats again if eval using EMA weights requested
                build_stats(history, ema_validation_output, callbacks, eval_callbacks, self.logger, comment="eval using EMA weights")

        if hvd.rank() == 0 and self.params.export_SavedModel:
            if not self.params.skip_eval and self.validation_builder is not None:
                # with the availability of eval stats and EMA weights, we choose the better weights for saving
                if self.params.moving_average_decay > 0 and float(ema_validation_output[1]) > float(validation_output[1]):
                    self.ema_opt = optimizer_factory.fetch_optimizer(self.model, optimizer_factory.MovingAverage)
                    self.ema_opt.swap_weights()
            self.model.save(self.params.model_dir + '/savedmodel', include_optimizer=True, save_format='tf')


    def evaluate(self):
                
        metrics_map = get_metrics(self.one_hot)
        metrics = [metrics_map[metric] for metric in self.metrics]
        
        if self.one_hot:
            loss_obj = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=self.params.label_smoothing)
        else:
            loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

        # set up optimizer
        optimizer = optimizer_factory.build_optimizer(
            optimizer_name=self.params.optimizer,
            base_learning_rate=0.1,
            params=get_optimizer_params(name=self.params.optimizer,
                decay=self.params.decay,
                epsilon=self.params.opt_epsilon,
                momentum=self.params.momentum,
                moving_average_decay=self.params.moving_average_decay,
                nesterov=self.params.nesterov,
                beta_1=self.params.beta_1,
                beta_2=self.params.beta_2)
            )
        
        self.model.compile(optimizer=optimizer, 
                    loss=loss_obj,
                    metrics=metrics,
                    run_eagerly=self.params.run_eagerly)

        if self.params.weights_format == 'saved_model':
            self.model = tf.keras.models.load_model(self.params.model_dir + '/savedmodel', custom_objects = {"HvdMovingAverage":optimizer_factory.HvdMovingAverage})
            #self.model.set_weights(loaded_model.get_weights())
            print('''
                  # =======================================
                        Saved_model loaded successfully!
                  # =======================================''')
        else:
            # we allow for partial loading
            resumed_step = self.load_model_weights(self.params.model_dir, expect_partial=True)
            if resumed_step > 0:
                print('''
                # =======================================
                    ckpt at iteration {} loaded!
                # ======================================='''.format(resumed_step))

        # Ckpt format contains both original weights and EMA weights. However, saved_model format only stores the better performing
        # weights between the original and EMA weights. As such, saved_model format doesn't allow for evaluation using EMA weights,
        # because we simply don't know which weights have ended up being saved in this format.
        if self.params.moving_average_decay > 0  and self.params.weights_format != 'saved_model':
            # =================================================
            # Define Datasets
            # =================================================
            builders = get_dataset_builders(self.params, self.one_hot)
            datasets = [builder.build() if builder else None for builder in builders]
            _, self.validation_dataset = datasets
            _, self.validation_builder = builders
            eval_callbacks = []
            if not self.params.skip_eval and self.validation_builder is not None:
                eval_callbacks.append(custom_callbacks.EvalTimeHistory(batch_size=self.params.eval_batch_size, logger=self.logger))
                eval_callbacks.append(custom_callbacks.MovingAverageCallback(intratrain_eval_using_ema=True))
                ema_worker_validation_output = self.model.evaluate(
                    self.validation_dataset, callbacks=eval_callbacks, verbose=1)

                build_stats(None, ema_worker_validation_output, None, eval_callbacks, self.logger, comment="eval using EMA weights")

        for round_num in range(self.params.n_repeat_eval):
            # =================================================
                # Define Datasets
            # =================================================
            builders = get_dataset_builders(self.params, self.one_hot)
            datasets = [builder.build() if builder else None for builder in builders]
            _, self.validation_dataset = datasets
            _, self.validation_builder = builders
            eval_callbacks = []
            if not self.params.skip_eval and self.validation_builder is not None:
                eval_callbacks.append(custom_callbacks.EvalTimeHistory(batch_size=self.params.eval_batch_size, logger=self.logger))
                worker_validation_output = self.model.evaluate(
                    self.validation_dataset, callbacks=eval_callbacks, verbose=1)

                build_stats(None, worker_validation_output, None, eval_callbacks, self.logger, comment="eval using original weights: Round {} ".format(round_num))


        if self.params.export_SavedModel and self.params.weights_format != 'saved_model':
            if self.params.moving_average_decay > 0 and float(ema_worker_validation_output[1]) > float(worker_validation_output[1]):
                self.ema_opt = optimizer_factory.fetch_optimizer(self.model, optimizer_factory.MovingAverage)
                self.ema_opt.swap_weights()
            self.model.save(self.params.model_dir + '/savedmodel' , include_optimizer=True, save_format='tf')
  
        



    def predict(self, img_dir, checkpoint_path=None, print_results=True):
        # verify checkpoint_name validity
            # if not, we use rnd weights
            # if so, load the model conditioned on the format


        # load the weights if ckpt exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            if self.params.weights_format == 'saved_model':
                loaded_model = tf.keras.models.load_model(checkpoint_path, custom_objects = {"HvdMovingAverage":optimizer_factory.HvdMovingAverage})
                self.model.set_weights(loaded_model.get_weights())
            elif self.params.weights_format == 'ckpt':
                self.load_model_weights(checkpoint_path, expect_partial=True)
        else:
            print('***ckpt not found! predicting using random weights...')

        try:
            tf.keras.backend.set_learning_phase(0)

            dtype = self.params.mparams.dtype
            images = preprocess_image_files(img_dir, self.params.predict_img_size, self.params.predict_batch_size, dtype)
            nb_samples = len(images)
            file_names = images.filenames
            num_files = len(file_names)
            REPEAT=50 if self.params.benchmark else 1
            print_results = not self.params.benchmark

            # start_time = time.time()
            # inference_results = self.model.predict(images, verbose=1, steps=nb_samples)
            # total_time = time.time() - start_time
            # score = tf.nn.softmax(inference_results, axis=1)

            num_files = num_files * REPEAT
            batch_times = []
            for i in range(nb_samples*REPEAT):
                start_time = time.time()
                image = images.next()
                batch_result = np.asarray(self.model(image),dtype='float32')
                batch_times.append(time.time() - start_time)
                if not i:
                    inference_results = batch_result
                else:
                    inference_results = np.vstack((inference_results,batch_result))
            total_time = np.sum(batch_times)
            score = tf.nn.softmax(tf.convert_to_tensor(inference_results, dtype = tf.float32),  axis=1)

            #
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

