# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

import tensorflow as tf

import horovod.tensorflow as hvd

from model import resnet_v1_5

from utils import hooks
from utils import data_utils
from utils import hvd_utils

from runtime import runner_utils

from dllogger.logger import LOGGER

__all__ = [
    'Runner',
]

class Runner(object):

    def __init__(
        self,
        # ========= Model HParams ========= #
        n_classes=1001,
        input_format='NHWC',    # NCHW or NHWC
        compute_format='NCHW',  # NCHW or NHWC
        dtype=tf.float32,       # tf.float32 or tf.float16
        n_channels=3,
        height=224,
        width=224,
        distort_colors=False,
        model_dir=None,
        log_dir=None,
        data_dir=None,
        data_idx_dir=None,

        # ======= Optimization HParams ======== #
        use_xla=False,
        use_tf_amp=False,
        use_dali=False,
        gpu_memory_fraction=1.0,
        gpu_id=0,
        
        # ======== Debug Flags ======== #
        debug_verbosity=0,
        seed=None
    ):

        if dtype not in [tf.float32, tf.float16]:
            raise ValueError("Unknown dtype received: %s (allowed: `tf.float32` and `tf.float16`)" % dtype)

        if compute_format not in ["NHWC", 'NCHW']:
            raise ValueError("Unknown `compute_format` received: %s (allowed: ['NHWC', 'NCHW'])" % compute_format)

        if input_format not in ["NHWC", 'NCHW']:
            raise ValueError("Unknown `input_format` received: %s (allowed: ['NHWC', 'NCHW'])" % input_format)

        if n_channels not in [1, 3]:
            raise ValueError("Unsupported number of channels: %d (allowed: 1 (grayscale) and 3 (color))" % n_channels)

        if data_dir is not None and not os.path.exists(data_dir):
            raise ValueError("The `data_dir` received does not exists: %s" % data_dir)

        hvd.init()
        tf_seed = 2 * (seed + hvd.rank()) if seed is not None else None

        # ============================================
        # Optimsation Flags - Do not remove
        # ============================================

        os.environ['CUDA_CACHE_DISABLE'] = '0'

        os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'

        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = '1' if not hvd_utils.is_using_hvd() else str(hvd.size())

        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

        os.environ['TF_ADJUST_HUE_FUSED'] = '1'
        os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        os.environ['TF_SYNC_ON_FINISH'] = '0'
        os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
        os.environ['TF_DISABLE_NVTX_RANGES'] = '1'

        # ============================================
        # TF-AMP Setup - Do not remove
        # ============================================

        if dtype == tf.float16:

            if use_tf_amp:
                raise RuntimeError("TF AMP can not be activated for FP16 precision")

        elif use_tf_amp:
            
            if hvd.rank() == 0:
                LOGGER.log("TF AMP is activated - Experimental Feature")
            os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"

        # =================================================

        model_hparams = tf.contrib.training.HParams(
            width=height,
            height=width,
            n_channels=n_channels,
            n_classes=n_classes,
            dtype=dtype,
            input_format=input_format,
            compute_format=compute_format,
            distort_colors=distort_colors,
            seed=tf_seed
        )
        
        if use_dali:
            num_preprocessing_threads=4
        else:
            num_preprocessing_threads=10

        run_config_performance = tf.contrib.training.HParams(
            num_preprocessing_threads=num_preprocessing_threads,
            use_tf_amp=use_tf_amp,
            use_xla=use_xla,
            use_dali=use_dali,
            gpu_memory_fraction=gpu_memory_fraction,
            gpu_id=gpu_id
        )

        run_config_additional = tf.contrib.training.HParams(
            model_dir=model_dir if not hvd_utils.is_using_hvd() or hvd.rank() == 0 else None,
            log_dir=log_dir if not hvd_utils.is_using_hvd() or hvd.rank() == 0 else None,
            data_dir=data_dir,
            data_idx_dir=data_idx_dir,
            num_preprocessing_threads=num_preprocessing_threads
        )

        self.run_hparams = Runner._build_hparams(model_hparams, run_config_additional, run_config_performance)

        self._model = resnet_v1_5.ResnetModel(
            model_name="resnet50_v1.5",
            n_classes=model_hparams.n_classes,
            input_format=model_hparams.input_format,
            compute_format=model_hparams.compute_format,
            dtype=model_hparams.dtype,
            use_dali=use_dali
        )

        if self.run_hparams.seed is not None:
            if hvd.rank() == 0:
                LOGGER.log("Deterministic Run - Seed: %d" % seed)
            tf.set_random_seed(self.run_hparams.seed)
            

    @staticmethod
    def _build_hparams(*args):

        hparams = tf.contrib.training.HParams()

        for _hparams in args:
            if not isinstance(_hparams, tf.contrib.training.HParams):
                raise ValueError("Non valid HParams argument object detected:", _hparams)

            for key, val in _hparams.values().items():
                try:
                    hparams.add_hparam(name=key, value=val)

                except ValueError:
                    warnings.warn(
                        "the parameter `{}` already exists - existing value: {} and duplicated value: {}".format(
                            key, hparams.get(key), val
                        )
                    )

        return hparams

    @staticmethod
    def _get_global_batch_size(worker_batch_size):

        if hvd_utils.is_using_hvd():
            return worker_batch_size * hvd.size()
        else:
            return worker_batch_size

    @staticmethod
    def _get_session_config(mode, use_xla, use_dali, gpu_memory_fraction, gpu_id=0):

        if mode not in ["train", 'validation', 'benchmark', 'inference']:
            raise ValueError("Unknown mode received: %s (allowed: 'train', 'validation', 'benchmark', 'inference')" % mode)

        # Limit available GPU memory (tune the size)
        if use_dali:
            LOGGER.log("DALI is activated, GPU memory fraction used for training is limited to", gpu_memory_fraction)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)  
            config = tf.ConfigProto(gpu_options=gpu_options)
            config.gpu_options.allow_growth = False

        
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

        config.allow_soft_placement = True
        config.log_device_placement = False
    
        config.gpu_options.visible_device_list = str(gpu_id)

        if hvd_utils.is_using_hvd():
            config.gpu_options.visible_device_list = str(hvd.local_rank())

        if use_xla: 
            LOGGER.log("XLA is activated - Experimental Feature")
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        config.gpu_options.force_gpu_compatible = True  # Force pinned memory

        if mode == 'train':
            config.intra_op_parallelism_threads = 1  # Avoid pool of Eigen threads

            if hvd_utils.is_using_hvd():
                config.inter_op_parallelism_threads = max(2, (multiprocessing.cpu_count() // hvd.size()) - 2)
            else:
                config.inter_op_parallelism_threads = 4

        return config

    @staticmethod
    def _get_run_config(mode, model_dir, use_xla, use_dali, gpu_memory_fraction, gpu_id=0, seed=None):

        if mode not in ["train", 'validation', 'benchmark', 'inference']:
            raise ValueError("Unknown mode received: %s (allowed: 'train', 'validation', 'benchmark', 'inference')" % mode)

        if seed is not None:
            if hvd_utils.is_using_hvd():
                tf_random_seed = 2 * (seed + hvd.rank())
            else:
                tf_random_seed = 2 * seed
        else:
            tf_random_seed = None

        config = tf.estimator.RunConfig(
            model_dir=model_dir,
            tf_random_seed=tf_random_seed,
            save_summary_steps=100 if mode in ['train', 'validation'] else 1e9,  # disabled in benchmark mode
            save_checkpoints_steps=None,
            save_checkpoints_secs=None,
            session_config=Runner._get_session_config(mode=mode, use_xla=use_xla, use_dali=use_dali, gpu_memory_fraction=gpu_memory_fraction, gpu_id=gpu_id),
            keep_checkpoint_max=5,
            keep_checkpoint_every_n_hours=1e6,  # disabled
            log_step_count_steps=1e9,
            train_distribute=None,
            device_fn=None,
            protocol=None,
            eval_distribute=None,
            experimental_distribute=None
        )

        if mode == 'train':
            if hvd_utils.is_using_hvd():
                config = config.replace(
                    save_checkpoints_steps=1000 if hvd.rank() == 0 else None, keep_checkpoint_every_n_hours=3
                )
            else:
                config = config.replace(save_checkpoints_steps=1000, keep_checkpoint_every_n_hours=3)

        return config

    def _get_estimator(self, mode, run_params, use_xla, use_dali, gpu_memory_fraction, gpu_id=0):

        if mode not in ["train", 'validation', 'benchmark', 'inference']:
            raise ValueError("Unknown mode received: %s (allowed: 'train', 'validation', 'benchmark', 'inference')" % mode)

        run_config = Runner._get_run_config(
            mode=mode,
            model_dir=self.run_hparams.model_dir,
            use_xla=use_xla,
            use_dali=use_dali,
            gpu_memory_fraction=gpu_memory_fraction,
            gpu_id=gpu_id,
            seed=self.run_hparams.seed
            
        )

        return tf.estimator.Estimator(
            model_fn=self._model,
            model_dir=self.run_hparams.model_dir,
            config=run_config,
            params=run_params
        )

    def train(
        self,
        iter_unit,
        num_iter,
        batch_size,
        warmup_steps=50,
        weight_decay=1e-4,
        lr_init=0.1,
        lr_warmup_epochs=5,
        momentum=0.9,
        log_every_n_steps=1,
        loss_scale=256,
        label_smoothing=0.0,
        mixup=0.0,
        use_cosine_lr=False,
        use_static_loss_scaling=False,
        is_benchmark=False
    ):

        if iter_unit not in ["epoch", "batch"]:
            raise ValueError('`iter_unit` value is unknown: %s (allowed: ["epoch", "batch"])' % iter_unit)

        if self.run_hparams.data_dir is None and not is_benchmark:
            raise ValueError('`data_dir` must be specified for training!')

        if self.run_hparams.use_tf_amp or self.run_hparams.dtype == tf.float16:
            if use_static_loss_scaling:
                os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING"] = "0"
            else:
                LOGGER.log("TF Loss Auto Scaling is activated")
                os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING"] = "1"
        else:
            use_static_loss_scaling = False  # Make sure it hasn't been set to True on FP32 training

        num_gpus = 1 if not hvd_utils.is_using_hvd() else hvd.size()
        global_batch_size = batch_size * num_gpus

        if self.run_hparams.data_dir is not None:
            filenames,num_samples, num_steps, num_epochs, num_decay_steps = runner_utils.parse_tfrecords_dataset(
                data_dir=self.run_hparams.data_dir,
                mode="train",
                iter_unit=iter_unit,
                num_iter=num_iter,
                global_batch_size=global_batch_size,
            )

            steps_per_epoch = num_steps / num_epochs

        else:
            num_epochs = 1
            num_steps = num_iter
            steps_per_epoch = num_steps
            num_decay_steps = num_steps
            num_samples = num_steps * batch_size

            
        if self.run_hparams.data_idx_dir is not None:
            idx_filenames = runner_utils.parse_dali_idx_dataset(
                data_idx_dir=self.run_hparams.data_idx_dir,
                mode="train"
            )
            
        training_hooks = []
      
        if hvd.rank() == 0:
            LOGGER.log('Starting Model Training...')
            LOGGER.log("Training Epochs", num_epochs)
            LOGGER.log("Total Steps", num_steps)
            LOGGER.log("Steps per Epoch", steps_per_epoch)
            LOGGER.log("Decay Steps", num_decay_steps)
            LOGGER.log("Weight Decay Factor", weight_decay)
            LOGGER.log("Init Learning Rate", lr_init)
            LOGGER.log("Momentum", momentum)
            LOGGER.log("Num GPUs", num_gpus)
            LOGGER.log("Per-GPU Batch Size", batch_size)

            
            if is_benchmark:

                benchmark_logging_hook = hooks.BenchmarkLoggingHook(
                    log_file_path=os.path.join(self.run_hparams.log_dir, "training_benchmark.json"),
                    global_batch_size=global_batch_size,
                    log_every=log_every_n_steps,
                    warmup_steps=warmup_steps
                )

                training_hooks.append(benchmark_logging_hook)

            else:

                training_logging_hook = hooks.TrainingLoggingHook(
                    log_file_path=os.path.join(self.run_hparams.log_dir, "training.json"),
                    global_batch_size=global_batch_size,
                    num_steps=num_steps,
                    num_samples=num_samples,
                    num_epochs=num_epochs,
                    log_every=log_every_n_steps
                )

                training_hooks.append(training_logging_hook)

        if hvd_utils.is_using_hvd():
            bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
            training_hooks.append(bcast_hook)

        training_hooks.append(hooks.PrefillStagingAreasHook())

      
        estimator_params = {
            'batch_size': batch_size,
            'steps_per_epoch': steps_per_epoch,
            'num_gpus': num_gpus,
            'momentum': momentum,
            'lr_init': lr_init,
            'lr_warmup_epochs': lr_warmup_epochs,
            'weight_decay': weight_decay,
            'loss_scale': loss_scale,
            'apply_loss_scaling': use_static_loss_scaling,
            'label_smoothing': label_smoothing,
            'mixup': mixup,
            'num_decay_steps': num_decay_steps,
            'use_cosine_lr': use_cosine_lr
        }

        image_classifier = self._get_estimator(
            mode='train',
            run_params=estimator_params,
            use_xla=self.run_hparams.use_xla,
            use_dali=self.run_hparams.use_dali,
            gpu_memory_fraction=self.run_hparams.gpu_memory_fraction,
            gpu_id=self.run_hparams.gpu_id
        )

        def training_data_fn():
            
            if self.run_hparams.use_dali and self.run_hparams.data_idx_dir is not None:
                if hvd.rank() == 0:
                    LOGGER.log("Using DALI input... ")
                    
                return data_utils.get_dali_input_fn(
                    filenames=filenames,
                    idx_filenames=idx_filenames,
                    batch_size=batch_size,
                    height=self.run_hparams.height,
                    width=self.run_hparams.width,
                    training=True,
                    distort_color=self.run_hparams.distort_colors,
                    num_threads=self.run_hparams.num_preprocessing_threads,
                    deterministic=False if self.run_hparams.seed is None else True
                )
            
            elif self.run_hparams.data_dir is not None:

                return data_utils.get_tfrecords_input_fn(
                    filenames=filenames,
                    batch_size=batch_size,
                    height=self.run_hparams.height,
                    width=self.run_hparams.width,
                    training=True,
                    distort_color=self.run_hparams.distort_colors,
                    num_threads=self.run_hparams.num_preprocessing_threads,
                    deterministic=False if self.run_hparams.seed is None else True
                )

            else:
                if hvd.rank() == 0:
                    LOGGER.log("Using Synthetic Data ...")
                return data_utils.get_synth_input_fn(
                    batch_size=batch_size,
                    height=self.run_hparams.height,
                    width=self.run_hparams.width,
                    num_channels=self.run_hparams.n_channels,
                    data_format=self.run_hparams.input_format,
                    num_classes=self.run_hparams.n_classes,
                    dtype=self.run_hparams.dtype,
                )


        try:
            image_classifier.train(
                input_fn=training_data_fn,
                steps=num_steps,
                hooks=training_hooks,
            )
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            
        if hvd.rank() == 0:
            LOGGER.log('Ending Model Training ...')

    def evaluate(
        self,
        iter_unit,
        num_iter,
        batch_size,
        warmup_steps=50,
        log_every_n_steps=1,
        is_benchmark=False,
        export_dir=None,
    ):

        if iter_unit not in ["epoch", "batch"]:
            raise ValueError('`iter_unit` value is unknown: %s (allowed: ["epoch", "batch"])' % iter_unit)

        if self.run_hparams.data_dir is None and not is_benchmark:
            raise ValueError('`data_dir` must be specified for evaluation!')

        if hvd_utils.is_using_hvd() and hvd.rank() != 0:
            raise RuntimeError('Multi-GPU inference is not supported')

        estimator_params = {}
            
        image_classifier = self._get_estimator(
            mode='validation',
            run_params=estimator_params,
            use_xla=self.run_hparams.use_xla,
            use_dali=self.run_hparams.use_dali,
            gpu_memory_fraction=self.run_hparams.gpu_memory_fraction,         
            gpu_id=self.run_hparams.gpu_id
        )

        if self.run_hparams.data_dir is not None:
            filenames, num_samples, num_steps, num_epochs, num_decay_steps = runner_utils.parse_tfrecords_dataset(
                data_dir=self.run_hparams.data_dir,
                mode="validation",
                iter_unit=iter_unit,
                num_iter=num_iter,
                global_batch_size=batch_size,
            )
        
        else:
            num_epochs = 1
            num_decay_steps = -1
            num_steps = num_iter
       
    
        if self.run_hparams.data_idx_dir is not None:
            idx_filenames = runner_utils.parse_dali_idx_dataset(
                data_idx_dir=self.run_hparams.data_idx_dir,
                mode="validation"
            )
    
        eval_hooks = []
        
        if hvd.rank() == 0:
            if is_benchmark:
                
                benchmark_logging_hook = hooks.BenchmarkLoggingHook(
                    log_file_path=os.path.join(self.run_hparams.log_dir, "eval_benchmark.json"),
                    global_batch_size=batch_size,
                    log_every=log_every_n_steps,
                    warmup_steps=warmup_steps
                )
                eval_hooks.append(benchmark_logging_hook)

            LOGGER.log('Starting Model Evaluation...')
            LOGGER.log("Evaluation Epochs", num_epochs)
            LOGGER.log("Evaluation Steps", num_steps)
            LOGGER.log("Decay Steps", num_decay_steps)
            LOGGER.log("Global Batch Size", batch_size)

        def evaluation_data_fn():
    
            if self.run_hparams.use_dali and self.run_hparams.data_idx_dir is not None:
                if hvd.rank() == 0:
                    LOGGER.log("Using DALI input... ")
                    
                return data_utils.get_dali_input_fn(
                    filenames=filenames,
                    idx_filenames=idx_filenames,
                    batch_size=batch_size,
                    height=self.run_hparams.height,
                    width=self.run_hparams.width,
                    training=False,
                    distort_color=self.run_hparams.distort_colors,
                    num_threads=self.run_hparams.num_preprocessing_threads,
                    deterministic=False if self.run_hparams.seed is None else True
                )
            
            
            elif self.run_hparams.data_dir is not None:
                return data_utils.get_tfrecords_input_fn(
                        filenames=filenames,
                        batch_size=batch_size,
                        height=self.run_hparams.height,
                        width=self.run_hparams.width,
                        training=False,
                        distort_color=self.run_hparams.distort_colors,
                        num_threads=self.run_hparams.num_preprocessing_threads,
                        deterministic=False if self.run_hparams.seed is None else True
                    )

            else:
                LOGGER.log("Using Synthetic Data ...\n")
                return data_utils.get_synth_input_fn(
                    batch_size=batch_size,
                    height=self.run_hparams.height,
                    width=self.run_hparams.width,
                    num_channels=self.run_hparams.n_channels,
                    data_format=self.run_hparams.input_format,
                    num_classes=self.run_hparams.n_classes,
                    dtype=self.run_hparams.dtype,
                )

        try:
            eval_results = image_classifier.evaluate(
                input_fn=evaluation_data_fn,
                steps=num_steps,
                hooks=eval_hooks,
            )
            LOGGER.log('Top-1 Accuracy: %.3f' % float(eval_results['top1_accuracy'] * 100))
            LOGGER.log('Top-5 Accuracy: %.3f' % float(eval_results['top5_accuracy'] * 100))

            #def get_serving_input_receiver_fn(batch_size, height, width, num_channels, data_format, dtype=tf.float32):   
            
            if export_dir is not None:
                LOGGER.log('Exporting to', export_dir)
                input_receiver_fn = data_utils.get_serving_input_receiver_fn(
                    batch_size=None,                        
                    height=self.run_hparams.height,
                    width=self.run_hparams.width,
                    num_channels=self.run_hparams.n_channels,
                    data_format=self.run_hparams.input_format,
                    dtype=self.run_hparams.dtype)
                
                image_classifier.export_savedmodel(export_dir, input_receiver_fn)
            
        except KeyboardInterrupt:
            print("Keyboard interrupt")

        LOGGER.log('Ending Model Evaluation ...')
        
    def predict(self, to_predict):

        estimator_params = {}
            
        if to_predict is not None:
            filenames = runner_utils.parse_inference_input(to_predict)
            
        image_classifier = self._get_estimator(
            mode='inference',
            run_params=estimator_params,
            use_xla=self.run_hparams.use_xla,
            use_dali=self.run_hparams.use_dali,
            gpu_memory_fraction=self.run_hparams.gpu_memory_fraction
        )
      
        inference_hooks = []
        
        def inference_data_fn():
            return data_utils.get_inference_input_fn(
                        filenames=filenames,
                        height=self.run_hparams.height,
                        width=self.run_hparams.width,
                        num_threads=self.run_hparams.num_preprocessing_threads
                )
        try:
            inference_results = image_classifier.predict(
                input_fn=inference_data_fn,
                predict_keys=None,
                hooks=inference_hooks,
                yield_single_examples=True
            )
            
            for result in inference_results:
                LOGGER.log(result['classes'], str(result['probabilities'][result['classes']]))

        except KeyboardInterrupt:
            print("Keyboard interrupt")

        LOGGER.log('Ending Inference ...')
