#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
# ==============================================================================

import os
import json
import multiprocessing
import operator
import random

import numpy as np

import tensorflow as tf
import horovod.tensorflow as hvd

from datasets import known_datasets

from model.unet import UNet_v1

from utils import hvd_utils

from utils.hooks import ProfilerHook

import dllogger as Logger

__all__ = [
    'Runner',
]


class Runner(object):

    def __init__(
        self,

        # Model Params
        input_format,  # NCHW or NHWC
        compute_format,  # NCHW or NHWC
        n_channels,
        activation_fn,
        weight_init_method,
        model_variant,
        input_shape,
        mask_shape,
        input_normalization_method,

        # Training HParams
        augment_data,
        loss_fn_name,

        #  Runtime HParams
        amp,
        xla,

        # Directory Params
        model_dir=None,
        log_dir=None,
        sample_dir=None,
        data_dir=None,
        dataset_name=None,
        dataset_hparams=None,

        # Debug Params
        log_every_n_steps=1,
        debug_verbosity=0,
        seed=None
    ):

        if dataset_hparams is None:
            dataset_hparams = dict()

        if compute_format not in ["NHWC", 'NCHW']:
            raise ValueError("Unknown `compute_format` received: %s (allowed: ['NHWC', 'NCHW'])" % compute_format)

        if input_format not in ["NHWC", 'NCHW']:
            raise ValueError("Unknown `input_format` received: %s (allowed: ['NHWC', 'NCHW'])" % input_format)

        if n_channels not in [1, 3]:
            raise ValueError("Unsupported number of channels: %d (allowed: 1 (grayscale) and 3 (color))" % n_channels)

        if data_dir is not None and not os.path.exists(data_dir):
            raise ValueError("The `data_dir` received does not exists: %s" % data_dir)

        if hvd_utils.is_using_hvd():
            hvd.init()

            if hvd.rank() == 0:
                print("Horovod successfully initialized ...")

            tf_seed = 2 * (seed + hvd.rank()) if seed is not None else None

        else:
            tf_seed = 2 * seed if seed is not None else None

        # ============================================
        # Optimisation Flags - Do not remove
        # ============================================

        os.environ['CUDA_CACHE_DISABLE'] = '0'

        os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'

        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = '1' if not hvd_utils.is_using_hvd() else str(hvd.size())
        print("WORLD_SIZE", hvd.size())

        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

        os.environ['TF_ADJUST_HUE_FUSED'] = '1'
        os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        os.environ['TF_SYNC_ON_FINISH'] = '0'
        os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'

        # =================================================

        self.xla = xla

        if amp:

            if not hvd_utils.is_using_hvd() or hvd.rank() == 0:
                print("TF AMP is activated - Experimental Feature")

            os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"

        # =================================================

        model_hparams = tf.contrib.training.HParams(
            # Model Params
            input_format=input_format,
            compute_format=compute_format,
            input_shape=input_shape,
            mask_shape=mask_shape,
            n_channels=n_channels,
            activation_fn=activation_fn,
            weight_init_method=weight_init_method,
            model_variant=model_variant,
            input_normalization_method=input_normalization_method,

            # Training HParams
            augment_data=augment_data,
            loss_fn_name=loss_fn_name,

            # Runtime Params
            amp=amp,

            # Debug Params
            log_every_n_steps=log_every_n_steps,
            debug_verbosity=debug_verbosity,
            seed=tf_seed
        )

        run_config_additional = tf.contrib.training.HParams(
            dataset_hparams=dataset_hparams,
            model_dir=model_dir if not hvd_utils.is_using_hvd() or hvd.rank() == 0 else None,
            log_dir=log_dir if not hvd_utils.is_using_hvd() or hvd.rank() == 0 else None,
            sample_dir=sample_dir if not hvd_utils.is_using_hvd() or hvd.rank() == 0 else None,
            data_dir=data_dir,
            num_preprocessing_threads=32,
        )

        if not hvd_utils.is_using_hvd() or hvd.rank() == 0:
            try:
                os.makedirs(sample_dir)
            except FileExistsError:
                pass

        self.run_hparams = Runner._build_hparams(model_hparams, run_config_additional)

        if not hvd_utils.is_using_hvd() or hvd.rank() == 0:
            print('Defining Model Estimator ...\n')

        self._model = UNet_v1(
            model_name="UNet_v1",
            input_format=self.run_hparams.input_format,
            compute_format=self.run_hparams.compute_format,
            n_output_channels=1,
            unet_variant=self.run_hparams.model_variant,
            weight_init_method=self.run_hparams.weight_init_method,
            activation_fn=self.run_hparams.activation_fn
        )

        if self.run_hparams.seed is not None:

            if not hvd_utils.is_using_hvd() or hvd.rank() == 0:
                print("Deterministic Run - Seed: %d\n" % seed)

            tf.set_random_seed(self.run_hparams.seed)
            np.random.seed(self.run_hparams.seed)
            random.seed(self.run_hparams.seed)

        if dataset_name not in known_datasets.keys():
            raise RuntimeError(
                "The dataset `%s` is unknown, allowed values: %s ..." % (dataset_name, list(known_datasets.keys()))
            )

        self.dataset = known_datasets[dataset_name](data_dir=data_dir, **self.run_hparams.dataset_hparams)

        self.num_gpus = 1 if not hvd_utils.is_using_hvd() else hvd.size()

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
                    print(
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
    def _get_session_config(mode, xla):

        if mode not in ["train", 'validation', 'benchmark']:
            raise ValueError("Unknown mode received: %s (allowed: 'train', 'validation', 'benchmark')" % mode)

        config = tf.ConfigProto()

        config.allow_soft_placement = True
        config.log_device_placement = False

        config.gpu_options.allow_growth = True

        if hvd_utils.is_using_hvd():
            config.gpu_options.visible_device_list = str(hvd.rank())

        if xla:
            if not hvd_utils.is_using_hvd() or hvd.rank() == 0:
                print("XLA is activated - Experimental Feature")
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        config.gpu_options.force_gpu_compatible = True  # Force pinned memory

        # TODO: Provide correct session configuration for both
        # variations with comments explaining why specific options were used

        if mode == 'train':
            config.intra_op_parallelism_threads = 1  # Avoid pool of Eigen threads

            if hvd_utils.is_using_hvd():
                config.inter_op_parallelism_threads = max(2, (multiprocessing.cpu_count() // hvd.size()) - 2)
            else:
                config.inter_op_parallelism_threads = 4

        return config

    @staticmethod
    def _get_run_config(mode, model_dir, xla, seed=None):

        if mode not in ["train", 'validation', 'benchmark']:
            raise ValueError("Unknown mode received: %s (allowed: 'train', 'validation', 'benchmark')" % mode)

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
            save_summary_steps=10 if mode == "train" else 1e9,  # disabled
            save_checkpoints_steps=None,
            save_checkpoints_secs=None,
            session_config=Runner._get_session_config(mode=mode, xla=xla),
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

    def _get_estimator(self, mode, run_params, xla):

        if mode not in ["train", 'validation', 'benchmark']:
            raise ValueError("Unknown mode received: %s (allowed: 'train', 'validation', 'benchmark')" % mode)

        run_config = Runner._get_run_config(
            mode=mode, model_dir=self.run_hparams.model_dir, xla=xla, seed=self.run_hparams.seed
        )

        return tf.estimator.Estimator(
            model_fn=self._model, model_dir=self.run_hparams.model_dir, config=run_config, params=run_params
        )

    def train(
        self,
        iter_unit,
        num_iter,
        batch_size,
        weight_decay,
        learning_rate,
        learning_rate_decay_factor,
        learning_rate_decay_steps,
        rmsprop_decay,
        rmsprop_momentum,
        use_auto_loss_scaling,
        augment_data,
        warmup_steps=50,
        is_benchmark=False
    ):

        if iter_unit not in ["epoch", "batch"]:
            raise ValueError('`iter_unit` value is unknown: %s (allowed: ["epoch", "batch"])' % iter_unit)

        if self.run_hparams.data_dir is None and not is_benchmark:
            raise ValueError('`data_dir` must be specified for training!')

        if self.run_hparams.amp:
            if use_auto_loss_scaling:

                if not hvd_utils.is_using_hvd() or hvd.rank() == 0:
                    print("TF Loss Auto Scaling is activated - Experimental Feature")

                os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING"] = "1"
                apply_manual_loss_scaling = False

            else:
                os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING"] = "0"
                apply_manual_loss_scaling = True
        else:
            apply_manual_loss_scaling = False

        global_batch_size = batch_size * self.num_gpus

        if self.run_hparams.data_dir is not None:
            filenames, num_samples, num_steps, num_epochs = self.dataset.get_dataset_runtime_specs(
                training=True, iter_unit=iter_unit, num_iter=num_iter, global_batch_size=global_batch_size
            )

            steps_per_epoch = int(num_steps / num_epochs)

        else:
            num_epochs = 1
            num_steps = num_iter
            steps_per_epoch = 625

        training_hooks = []

        if hvd_utils.is_using_hvd():
            training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

        if not hvd_utils.is_using_hvd() or hvd.rank() == 0:
            training_hooks.append(
                ProfilerHook(
                    global_batch_size=global_batch_size,
                    log_every=self.run_hparams.log_every_n_steps,
                    warmup_steps=warmup_steps,
                    is_training=True,
                    sample_dir=self.run_hparams.sample_dir
                )
            )

            print("Starting Model Training ...")

            Logger.log(step=('PARAMETER'), data={"Epochs": num_epochs}, verbosity=Logger.Verbosity.DEFAULT)
            Logger.log(step=('PARAMETER'), data={"Total Steps": num_steps}, verbosity=Logger.Verbosity.DEFAULT)
            Logger.log(step=('PARAMETER'), data={"Steps per Epoch": steps_per_epoch}, verbosity=Logger.Verbosity.DEFAULT)
            Logger.log(step=('PARAMETER'), data={"Weight Decay Factor": weight_decay}, verbosity=Logger.Verbosity.DEFAULT)
            Logger.log(step=('PARAMETER'), data={"Learning Rate": learning_rate}, verbosity=Logger.Verbosity.DEFAULT)
            Logger.log(step=('PARAMETER'), data={"Learning Rate Decay Factor": learning_rate_decay_factor}, verbosity=Logger.Verbosity.DEFAULT)
            Logger.log(step=('PARAMETER'), data={"Learning Rate Decay Steps": learning_rate_decay_steps}, verbosity=Logger.Verbosity.DEFAULT)
            Logger.log(step=('PARAMETER'), data={"RMSProp - Decay": rmsprop_decay}, verbosity=Logger.Verbosity.DEFAULT)
            Logger.log(step=('PARAMETER'), data={"RMSProp - Momentum": rmsprop_momentum}, verbosity=Logger.Verbosity.DEFAULT)
            Logger.log(step=('PARAMETER'), data={"Loss Function Name": self.run_hparams.loss_fn_name}, verbosity=Logger.Verbosity.DEFAULT)

            if self.run_hparams.amp:
                Logger.log(step=('PARAMETER'), data={"Use Auto Loss Scaling": use_auto_loss_scaling}, verbosity=Logger.Verbosity.DEFAULT)

            Logger.log(step=('PARAMETER'), data={"# GPUs": self.num_gpus}, verbosity=Logger.Verbosity.DEFAULT)
            Logger.log(step=('PARAMETER'), data={"GPU Batch Size": batch_size}, verbosity=Logger.Verbosity.DEFAULT)
            Logger.log(step=('PARAMETER'), data={"Global Batch Size": global_batch_size}, verbosity=Logger.Verbosity.DEFAULT)
            Logger.log(step=('PARAMETER'), data={"Total Files to be Processed": num_steps * global_batch_size}, verbosity=Logger.Verbosity.DEFAULT)

            print()  # visual spacing

        estimator_params = {
            'batch_size': batch_size,
            'steps_per_epoch': steps_per_epoch,
            'learning_rate': learning_rate,
            'learning_rate_decay_steps': learning_rate_decay_steps,
            'learning_rate_decay_factor': learning_rate_decay_factor,
            'rmsprop_decay': rmsprop_decay,
            'rmsprop_momentum': rmsprop_momentum,
            'weight_decay': weight_decay,
            'apply_manual_loss_scaling': apply_manual_loss_scaling,
            'loss_fn_name': self.run_hparams.loss_fn_name,
            'debug_verbosity': self.run_hparams.debug_verbosity,
        }

        def training_data_fn():

            if not is_benchmark or self.run_hparams.data_dir is not None:

                return self.dataset.dataset_fn(
                    batch_size=batch_size,
                    training=True,
                    only_defective_images=True,
                    augment_data=augment_data,
                    input_shape=list(self.run_hparams.input_shape) + [self.run_hparams.n_channels],
                    mask_shape=list(self.run_hparams.mask_shape) + [self.run_hparams.n_channels],
                    num_threads=64,
                    use_gpu_prefetch=True,
                    normalize_data_method="zero_centered",
                    seed=self.run_hparams.seed
                )

            else:
                if not hvd_utils.is_using_hvd() or hvd.rank() == 0:
                    print("Using Synthetic Data ...")

                return self.dataset.synth_dataset_fn(
                    batch_size=batch_size,
                    training=True,
                    input_shape=list(self.run_hparams.input_shape) + [self.run_hparams.n_channels],
                    mask_shape=list(self.run_hparams.mask_shape) + [self.run_hparams.n_channels],
                    num_threads=64,
                    use_gpu_prefetch=True,
                    normalize_data_method="zero_centered",
                    only_defective_images=True,
                    augment_data=augment_data,
                    seed=self.run_hparams.seed
                )

        model = self._get_estimator(mode='train', run_params=estimator_params, xla=self.xla)

        try:
            model.train(
                input_fn=training_data_fn,
                steps=num_steps,
                hooks=training_hooks,
            )
        except KeyboardInterrupt:
            print("Keyboard interrupt")

        if not hvd_utils.is_using_hvd() or hvd.rank() == 0:
            print('Ending Model Training ...')

    def evaluate(self, iter_unit, num_iter, batch_size, warmup_steps=50, is_benchmark=False, save_eval_results_to_json=False):

        if iter_unit not in ["epoch", "batch"]:
            raise ValueError('`iter_unit` value is unknown: %s (allowed: ["epoch", "batch"])' % iter_unit)

        if self.run_hparams.data_dir is None and not is_benchmark:
            raise ValueError('`data_dir` must be specified for evaluation!')

        # if hvd_utils.is_using_hvd() and hvd.rank() != 0:
        #     raise RuntimeError('Multi-GPU inference is not supported')

        print('Defining Model Estimator ...\n')

        if self.run_hparams.data_dir is not None:
            filenames, num_samples, num_steps, num_epochs = self.dataset.get_dataset_runtime_specs(
                training=False, iter_unit=iter_unit, num_iter=num_iter, global_batch_size=batch_size
            )

            steps_per_epoch = num_steps / num_epochs

        else:
            num_epochs = 1
            num_steps = num_iter
            steps_per_epoch = num_steps

        evaluation_hooks = [
            ProfilerHook(
                global_batch_size=batch_size,
                log_every=self.run_hparams.log_every_n_steps,
                warmup_steps=warmup_steps,
                is_training=False,
                sample_dir=self.run_hparams.sample_dir
            )
        ]

        print('Starting Model Evaluation ...\n')

        Logger.log(step=('PARAMETER'), data={"Epochs": num_epochs}, verbosity=Logger.Verbosity.DEFAULT)
        Logger.log(step=('PARAMETER'), data={"Total Steps": num_steps}, verbosity=Logger.Verbosity.DEFAULT)
        Logger.log(step=('PARAMETER'), data={"Steps per Epoch": steps_per_epoch}, verbosity=Logger.Verbosity.DEFAULT)
        Logger.log(step=('PARAMETER'), data={"GPU Batch Size": batch_size}, verbosity=Logger.Verbosity.DEFAULT)
        Logger.log(step=('PARAMETER'), data={"Total Files to Processed": num_steps * batch_size}, verbosity=Logger.Verbosity.DEFAULT)

        print()  # visual spacing

        estimator_params = {
            'batch_size': batch_size,
            'steps_per_epoch': steps_per_epoch,
            'loss_fn_name': self.run_hparams.loss_fn_name,
            'debug_verbosity': self.run_hparams.debug_verbosity,
        }

        def evaluation_data_fn():

            if not is_benchmark or self.run_hparams.data_dir is not None:

                return self.dataset.dataset_fn(
                    batch_size=batch_size,
                    training=False,
                    input_shape=list(self.run_hparams.input_shape) + [self.run_hparams.n_channels],
                    mask_shape=list(self.run_hparams.mask_shape) + [self.run_hparams.n_channels],
                    num_threads=64,
                    use_gpu_prefetch=True,
                    normalize_data_method="zero_centered",
                    only_defective_images=False,
                    augment_data=False,
                    seed=self.run_hparams.seed
                )

            else:
                print("Using Synthetic Data ...")

                return self.dataset.synth_dataset_fn(
                    batch_size=batch_size,
                    training=False,
                    input_shape=list(self.run_hparams.input_shape) + [self.run_hparams.n_channels],
                    mask_shape=list(self.run_hparams.mask_shape) + [self.run_hparams.n_channels],
                    num_threads=64,
                    use_gpu_prefetch=True,
                    normalize_data_method="zero_centered",
                    only_defective_images=False,
                    augment_data=False,
                    seed=self.run_hparams.seed
                )

        model = self._get_estimator(mode='validation', run_params=estimator_params, xla=self.xla)

        try:
            eval_results = model.evaluate(
                input_fn=evaluation_data_fn,
                steps=num_steps,
                hooks=evaluation_hooks,
            )

            print('Ending Model Evaluation ...')

            print('###################################\n\nEvaluation Results:\n')

            data_to_log = {"{prefix}.{key}".format(prefix=Logger._stage, key=key): float(val)
                           for key, val in sorted(eval_results.items(), key=operator.itemgetter(0))
                           if not any(val in key for val in ["loss", "global_step", "Confusion_Matrix"])}
            Logger.log(step=(), data=data_to_log, verbosity=Logger.Verbosity.DEFAULT)

            fns = eval_results["Confusion_Matrix_FN"]
            fps = eval_results["Confusion_Matrix_FP"]
            tns = eval_results["Confusion_Matrix_TN"]
            tps = eval_results["Confusion_Matrix_TP"]

            positives = np.add(tps, fns)
            negatives = np.add(tns, fps)

            tpr = np.divide(tps, positives)
            tnr = np.divide(tns, negatives)

            Logger.log(
                step=(num_steps,),
                data={"{prefix}.true_positives".format(prefix=Logger._stage): str(tps)},
                verbosity=Logger.Verbosity.DEFAULT
            )

            Logger.log(
                step=(num_steps,),
                data={"{prefix}.true_negatives".format(prefix=Logger._stage): str(tns)},
                verbosity=Logger.Verbosity.DEFAULT
            )

            Logger.log(
                step=(num_steps,),
                data={"{prefix}.false_positives".format(prefix=Logger._stage): str(fps)},
                verbosity=Logger.Verbosity.DEFAULT
            )

            Logger.log(
                step=(num_steps,),
                data={"{prefix}.false_negatives".format(prefix=Logger._stage): str(fns)},
                verbosity=Logger.Verbosity.DEFAULT
            )

            Logger.log(
                step=(num_steps,),
                data={"{prefix}.true_positive_rate".format(prefix=Logger._stage): str(["%.3f" % x for x in tpr])},
                verbosity=Logger.Verbosity.DEFAULT
            )

            Logger.log(
                step=(num_steps,),
                data={"{prefix}.true_negative_rate".format(prefix=Logger._stage): str(["%.3f" % x for x in tnr])},
                verbosity=Logger.Verbosity.DEFAULT
            )

            if save_eval_results_to_json:

                results_dict = {
                    'IoU': {
                        '0.75': str(eval_results["IoU_THS_0.75"]),
                        '0.85': str(eval_results["IoU_THS_0.85"]),
                        '0.95': str(eval_results["IoU_THS_0.95"]),
                        '0.99': str(eval_results["IoU_THS_0.99"]),
                    },
                    'TPR': {
                        '0.75': str(tpr[-4]),
                        '0.85': str(tpr[-3]),
                        '0.95': str(tpr[-2]),
                        '0.99': str(tpr[-1]),
                    },
                    'TNR': {
                        '0.75': str(tnr[-4]),
                        '0.85': str(tnr[-3]),
                        '0.95': str(tnr[-2]),
                        '0.99': str(tnr[-1]),
                    }
                }

                with open(os.path.join(self.run_hparams.model_dir, "..", "results.json"), 'w') as f:
                    json.dump(results_dict, f)

        except KeyboardInterrupt:
            print("Keyboard interrupt")
