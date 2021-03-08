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
import time
import operator

import numpy as np
import tensorflow as tf

import dllogger as Logger

__all__ = ["ProfilerHook"]


class ProfilerHook(tf.train.SessionRunHook):

    def __init__(self, global_batch_size, sample_dir, log_every=10, warmup_steps=20, is_training=True):

        self._warmup_steps = warmup_steps
        self._global_batch_size = global_batch_size
        self._current_step = 0

        self._log_every = log_every

        self._t0 = None
        self._start_training_time = None

        self._is_training = is_training

        self._sample_dir = sample_dir

        self._processing_speed_arr = list()

    @staticmethod
    def moving_average(a, n=4):
        if len(a) < n:
            return [np.mean(a)]

        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def after_create_session(self, session, coord):

        params_count = tf.get_default_graph().get_tensor_by_name("trainable_parameters_count_ref:0")
        _params_count = session.run(params_count)

        Logger._stage = "train" if self._is_training else "eval"

        Logger.log(
            step=('PARAMETER'),
            data={"# Total Trainable Parameters": int(_params_count)}, verbosity=Logger.Verbosity.DEFAULT
        )

        Logger.metadata(
            metric="{prefix}.avg_ips".format(prefix=Logger._stage),
            metadata={"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": Logger._stage.upper()}
        )

        for ths in [0.05, 0.125, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99]:
            Logger.metadata(
                metric="{prefix}.IoU_THS_{ths}".format(prefix=Logger._stage, ths=ths),
                metadata={"format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": Logger._stage.upper()}
            )

        if self._is_training:
            Logger.metadata(
                metric="{prefix}.learning_rate".format(prefix=Logger._stage),
                metadata={"format": ":.3e", "GOAL": "NONE", "STAGE": Logger._stage.upper()}
            )

            Logger.metadata(
                metric="{prefix}.weight_decay".format(prefix=Logger._stage),
                metadata={"format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": Logger._stage.upper()}
            )

            Logger.metadata(
                metric="{prefix}.reconstruction_loss".format(prefix=Logger._stage),
                metadata={"format": ":.3f", "GOAL": "MINIMIZE", "STAGE": Logger._stage.upper()}
            )

            Logger.metadata(
                metric="{prefix}.total_loss".format(prefix=Logger._stage),
                metadata={"format": ":.3f", "GOAL": "MINIMIZE", "STAGE": Logger._stage.upper()}
            )

        Logger.metadata(
            metric="{prefix}.true_positives".format(prefix=Logger._stage),
            metadata={"STAGE": Logger._stage.upper()}
        )

        Logger.metadata(
            metric="{prefix}.true_negatives".format(prefix=Logger._stage),
            metadata={"STAGE": Logger._stage.upper()}
        )

        Logger.metadata(
            metric="{prefix}.false_positives".format(prefix=Logger._stage),
            metadata={"STAGE": Logger._stage.upper()}
        )

        Logger.metadata(
            metric="{prefix}.false_negatives".format(prefix=Logger._stage),
            metadata={"STAGE": Logger._stage.upper()}
        )

        Logger.metadata(
            metric="{prefix}.true_positive_rate".format(prefix=Logger._stage),
            metadata={"STAGE": Logger._stage.upper()}
        )

        Logger.metadata(
            metric="{prefix}.true_negative_rate".format(prefix=Logger._stage),
            metadata={"STAGE": Logger._stage.upper()}
        )

        self._start_training_time = time.time()

    def before_run(self, run_context):

        self._current_step += 1

        request_fetches = dict()

        if self._current_step % self._log_every == 0:

            additional_fetches = {
                'total_loss': tf.get_default_graph().get_tensor_by_name("losses/total_loss_ref:0"),
                'iou_scores': dict(),
                'confusion_matrix': dict()
            }

            if self._is_training:

                additional_fetches["weight_decay"] = tf.get_default_graph().get_tensor_by_name("losses/l2_loss_ref:0")

                additional_fetches["reconstruction_loss"] = tf.get_default_graph(
                ).get_tensor_by_name("losses/reconstruction_loss_ref:0")

                additional_fetches["learning_rate"] = tf.get_default_graph(
                ).get_tensor_by_name("optimizers/learning_rate_ref:0")

            # ==================== Samples ==================== #

            if self._sample_dir is not None and self._is_training:
                additional_fetches["samples"] = {}

                additional_fetches["samples"]["input_image"] = tf.get_default_graph(
                ).get_tensor_by_name("input_image_jpeg_ref:0")

                additional_fetches["samples"]["mask"] = tf.get_default_graph().get_tensor_by_name("mask_sample_ref:0")

                for threshold in [None, 0.05, 0.125, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99]:

                    additional_fetches["samples"][str(threshold)] = tf.get_default_graph().get_tensor_by_name(
                        "output_sample_ths_%s_ref:0" % threshold
                    )

            # ==================== Evaluation Metrics ==================== #

            for threshold in [0.05, 0.125, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99]:

                if threshold is not None:
                    additional_fetches["iou_scores"][str(threshold)] = tf.get_default_graph().get_tensor_by_name(
                        "IoU_Metrics/iou_score_ths_%s_ref:0" % threshold
                    )

            additional_fetches["confusion_matrix"]["tp"] = tf.get_default_graph(
            ).get_tensor_by_name("Confusion_Matrix/true_positives_ref:0")

            additional_fetches["confusion_matrix"]["tn"] = tf.get_default_graph(
            ).get_tensor_by_name("Confusion_Matrix/true_negatives_ref:0")

            additional_fetches["confusion_matrix"]["fp"] = tf.get_default_graph(
            ).get_tensor_by_name("Confusion_Matrix/false_positives_ref:0")

            additional_fetches["confusion_matrix"]["fn"] = tf.get_default_graph(
            ).get_tensor_by_name("Confusion_Matrix/false_negatives_ref:0")

            # Update `request_fetches` dict
            request_fetches.update(additional_fetches)

            print("\n######### START: %d ##############" % self._current_step)

        self._t0 = time.time()

        return tf.train.SessionRunArgs(fetches=request_fetches)

    def after_run(self, run_context, run_values):

        batch_time = time.time() - self._t0
        imgs_per_sec = int(self._global_batch_size / batch_time)

        is_log_step = self._current_step % self._log_every == 0

        if is_log_step:

            if self._current_step > self._warmup_steps:
                imgs_per_sec = float(ProfilerHook.moving_average(self._processing_speed_arr, n=30)[-1])

            Logger.log(
                step=(self._current_step,),
                data={"{prefix}.avg_ips".format(prefix=Logger._stage): float(imgs_per_sec)},
                verbosity=Logger.Verbosity.DEFAULT
            )

            if self._is_training:
                Logger.log(
                    step=(self._current_step,),
                    data={"{prefix}.weight_decay".format(prefix=Logger._stage): float(run_values.results["weight_decay"])},
                    verbosity=Logger.Verbosity.DEFAULT
                )
                Logger.log(
                    step=(self._current_step,),
                    data={"{prefix}.reconstruction_loss".format(prefix=Logger._stage): float(run_values.results["reconstruction_loss"])},
                    verbosity=Logger.Verbosity.DEFAULT
                )
                Logger.log(
                    step=(self._current_step,),
                    data={"{prefix}.total_loss".format(prefix=Logger._stage): float(run_values.results["total_loss"])},
                    verbosity=Logger.Verbosity.DEFAULT
                )
                Logger.log(
                    step=(self._current_step,),
                    data={"{prefix}.learning_rate".format(prefix=Logger._stage): float(run_values.results["learning_rate"])},
                    verbosity=Logger.Verbosity.DEFAULT
                )

            for key, val in sorted(run_values.results["iou_scores"].items(), key=operator.itemgetter(0)):
                Logger.log(
                    step=(self._current_step,),
                    data={"{prefix}.IoU_THS_{ths}".format(prefix=Logger._stage, ths=key): float(val)},
                    verbosity=Logger.Verbosity.DEFAULT
                )

            Logger.log(
                step=(self._current_step,),
                data={"{prefix}.true_positives".format(prefix=Logger._stage): str(run_values.results["confusion_matrix"]["tp"])},
                verbosity=Logger.Verbosity.DEFAULT
            )

            Logger.log(
                step=(self._current_step,),
                data={"{prefix}.true_negatives".format(prefix=Logger._stage): str(run_values.results["confusion_matrix"]["tn"])},
                verbosity=Logger.Verbosity.DEFAULT
            )

            Logger.log(
                step=(self._current_step,),
                data={"{prefix}.false_positives".format(prefix=Logger._stage): str(run_values.results["confusion_matrix"]["fp"])},
                verbosity=Logger.Verbosity.DEFAULT
            )

            Logger.log(
                step=(self._current_step,),
                data={"{prefix}.false_negatives".format(prefix=Logger._stage): str(run_values.results["confusion_matrix"]["fn"])},
                verbosity=Logger.Verbosity.DEFAULT
            )

            if self._sample_dir is not None and self._is_training:

                for key in sorted(run_values.results["samples"].keys(), key=operator.itemgetter(0)):

                    with open(
                        os.path.join(self._sample_dir, "sample_step_%04d_ths_%s.jpeg" % (self._current_step, key)), 'wb'
                    ) as fd:
                        fd.write(run_values.results["samples"][key])

                    with open(
                        os.path.join(self._sample_dir, "sample_step_%04d_mask.jpeg" % self._current_step), 'wb'
                    ) as fd:
                        fd.write(run_values.results["samples"]["mask"])

            print("######### STOP: %d ##############" % self._current_step)

        elif self._current_step > self._warmup_steps:
            # Do not store speed for log step due to additional fetches
            self._processing_speed_arr.append(imgs_per_sec)

    def end(self, session):

        try:
            avg_processing_speed = float(ProfilerHook.moving_average(self._processing_speed_arr, n=100)[-1])
        except:
            avg_processing_speed = float(np.mean(self._processing_speed_arr))

        total_processing_time = time.time() - self._start_training_time

        total_processing_hours, rem = divmod(total_processing_time, 3600)

        print("\n============== Final Summary ==============")
        Logger.log(
            step=(),
            data={"{prefix}.avg_ips".format(prefix=Logger._stage): avg_processing_speed},
            verbosity=Logger.Verbosity.DEFAULT
        )

        perf_dict = {'throughput': str(avg_processing_speed), 'processing_time': str(total_processing_time)}

        perf_filename = "performances_%s.json" % ("train" if self._is_training else "eval")

        with open(os.path.join(self._sample_dir, "..", perf_filename), 'w') as f:
            json.dump(perf_dict, f)
