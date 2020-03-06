#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import os
import sys

import inspect
import operator
import six
import subprocess
import time

from enum import Enum

from mask_rcnn.utils.logging_formatter import logging

from mask_rcnn.utils.decorators import atexit_hook
from mask_rcnn.utils.metaclasses import SingletonMetaClass

from mask_rcnn.utils.meters import ACCEPTED_INT_NUMBER_FORMATS
from mask_rcnn.utils.meters import ACCEPTED_FLOAT_NUMBER_FORMATS

import dllogger
from dllogger import Verbosity

__all__ = ["LoggingBackend", "LoggingScope", "DistributedStrategy", "RuntimeMode"]


class _BaseEnum(Enum):

    @classmethod
    def __values__(cls):
        return [getattr(cls, m.name) for m in cls]


class LoggingScope(_BaseEnum):
    ITER = 'Iteration'
    EPOCH = 'AllReduce'


class DistributedStrategy(_BaseEnum):
    REDUCE_SUM = 'AllGather'
    REDUCE_MEAN = 'AllReduce'
    NONE = None


class RuntimeMode(_BaseEnum):
    TRAIN = 'train'
    INFERENCE = 'inference'
    VALIDATION = 'validation'
    TEST = 'test'


def validate_runtime_mode(requested_mode):
    cls_attributes = inspect.getmembers(RuntimeMode, lambda a: not (inspect.isroutine(a)))
    authorized_modes = [a for a in cls_attributes if not (a[0].startswith('__') and a[0].endswith('__'))]

    for _, mode in authorized_modes:
        if mode == requested_mode:
            return
    else:
        raise ValueError(
            "Unknown requested mode: `%s` - Authorized: %s" % (requested_mode, [name for name, _ in authorized_modes])
        )


@atexit_hook
@six.add_metaclass(SingletonMetaClass)
class LoggingBackend(object):

    SEP_TARGET_LENGTH = 50

    # ================= Logging Methods ================= #

    LOGGING_PREFIX = ""

    def __init__(self):
        # super(LoggingBackend, self).__init__()

        self.runtime_initialized = {"train": False, "evaluation": False}

    # ================= Constructor/Destructor Methods ================= #

    def __atexit__(self):

        is_success = not (hasattr(sys, "last_traceback") and sys.last_traceback is not None)

        print()  # Visual spacing
        if is_success:
            self.log_info("Job finished with status: `SUCCESS`")
        else:
            logging.error("Job finished with an uncaught exception: `FAILURE`")

    def log_debug(self, message):
        logging.debug("%s%s" % (self.LOGGING_PREFIX, message))

    def log_info(self, message):
        logging.info("%s%s" % (self.LOGGING_PREFIX, message))

    def log_warning(self, message):
        logging.warning("%s%s" % (self.LOGGING_PREFIX, message))

    def log_error(self, message):
        logging.error("%s%s" % (self.LOGGING_PREFIX, message))

    def log_critical(self, message):
        logging.critical("%s%s" % (self.LOGGING_PREFIX, message))

    # ================= Automated Logging Methods ================= #
    @staticmethod
    def format_metric_value(value):

        if isinstance(value, ACCEPTED_FLOAT_NUMBER_FORMATS):

            if value < 1e-4 or value > 1e4:
                print_value = "%.4e" % value

            else:
                print_value = "{}".format(round(value, 5))

        elif isinstance(value, ACCEPTED_INT_NUMBER_FORMATS):
            print_value = "%d" % value

        else:
            print_value = value

        return print_value

    # ================= Runtime Logging Method ================= #
    def log_runtime(self, is_train=False):

        if is_train:
            if not self.runtime_initialized["train"]:
                self.runtime_initialized["train"] = True
                _message = "                 Start Training                  "
            else:
                _message = "                Restart Training                 "

        else:
            if not self.runtime_initialized["evaluation"]:
                self.runtime_initialized["evaluation"] = True
                _message = "                Start Evaluation                 "
            else:
                _message = "               Restart Evaluation                "

        print()  # Visual Spacing
        self.log_info("# ============================================= #")
        self.log_info(_message)
        self.log_info("# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #")
        print()  # Visual Spacing

    # ================= Automated Logging Methods ================= #

    def log_git_status(self):

        git_metadata = dict()

        def get_cmd_result(cmd):
            return subprocess.check_output(cmd, shell=True).decode("utf-8").strip()

        try:
            git_metadata["branch_name"] = get_cmd_result("git symbolic-ref -q HEAD | cut -d/ -f3-")  # current branch
            git_metadata["commit_id"] = get_cmd_result("git rev-parse HEAD")  # current commit ID
            git_metadata["remote_url"] = get_cmd_result("git remote get-url origin")  # git origin url

            if git_metadata["branch_name"] == "":
                del git_metadata["branch_name"]

        except subprocess.CalledProcessError:  # Not a git repository
            pass

        if git_metadata is None:
            raise ValueError("`git_metadata` value received is `None`")

        self.log_info("===================================== GIT REPOSITORY =====================================")
        for key, value in sorted(git_metadata.items(), key=operator.itemgetter(0)):
            self.log_info("%s: %s" % (key.replace("_", " ").upper(), value))
        self.log_info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

    def log_model_statistics(self, model_statistics=None):

        if model_statistics is None:
            raise ValueError("`model_statistics` value received is `None`")

        if not isinstance(model_statistics, dict):
            raise ValueError("`model_statistics` should be a `dict`")

        self.log_info("==================================== MODEL STATISTICS ====================================")
        for key, value in sorted(model_statistics.items(), key=operator.itemgetter(0)):
            self.log_info("%s: %s" % (key, value))
        self.log_info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

    def log_trainable_variables(self, var_list=None):

        if var_list is None:
            raise ValueError("`var_list` value received is `None`")

        self.log_info("=================================== TRAINABLE VARIABLES ==================================")
        for idx, (var_name, var_shape) in enumerate(var_list):
            self.log_info(
                "[#{idx:04d}] {name:<60s} => {shape}".format(idx=idx + 1, name=var_name, shape=str(var_shape))
            )
        self.log_info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

    # ================= Step Logging Method ================= #

    def log_step(self, iteration, throughput, gpu_stats):

        # print()  # Visual Spacing
        self.log_info("timestamp: %s" % time.time())
        self.log_info("iteration: %d" % int(iteration))

        if throughput is not None:
            self.log_info("throughput: %.1f samples/sec" % float(throughput))
        else:
            self.log_info("throughput: None")

    def log_amp_runtime(self, current_loss_scale, steps_non_skipped, steps_since_last_scale):

        header_name = " AMP Statistics "
        reference_len = int((LoggingBackend.SEP_TARGET_LENGTH - len(header_name)) / 2)

        if current_loss_scale is not None or steps_since_last_scale is not None:
            self.log_info(
                "%s%s%s" % (
                    "=" * reference_len, header_name, "=" *
                    (LoggingBackend.SEP_TARGET_LENGTH - len(header_name) - reference_len)
                )
            )

            self.log_info("Steps - Non Skipped: %s" % steps_non_skipped)

            if steps_since_last_scale is not None:
                self.log_info("Steps - Since last loss scale: %s" % steps_since_last_scale)

            if current_loss_scale is not None:
                self.log_info("Loss Scale: %s" % current_loss_scale)

    # ================= Metric Logging Methods ================= #

    def log_metrics(self, metric_data, iteration, runtime_mode):

        validate_runtime_mode(runtime_mode)

        if not isinstance(metric_data, dict):
            raise ValueError("`metric_data` should be a dictionary. Received: %s" % type(metric_data))

        if not isinstance(iteration, ACCEPTED_INT_NUMBER_FORMATS):
            raise ValueError("`iteration` should be an integer. Received: %s" % type(iteration))

        header_name = " Metrics "
        reference_len = int((LoggingBackend.SEP_TARGET_LENGTH - len(header_name)) / 2)

        self.log_info(
            "%s%s%s" % (
                "=" * reference_len, header_name, "=" *
                (LoggingBackend.SEP_TARGET_LENGTH - len(header_name) - reference_len)
            )
        )

        for key, value in sorted(metric_data.items(), key=operator.itemgetter(0)):
            print_value = LoggingBackend.format_metric_value(value)
            self.log_info("%s: %s" % (key, print_value))

    def log_final_metrics(self, metric_data, runtime_mode):

        validate_runtime_mode(runtime_mode)

        for key, value in sorted(metric_data.items(), key=operator.itemgetter(0)):
            print_value = LoggingBackend.format_metric_value(value)
            self.log_info("%s: %s" % (key, print_value))
        dllogger.log(step=(), data=metric_data, verbosity=Verbosity.DEFAULT)

    # ================= Summary Logging Method ================= #

    def log_summary(self, is_train, total_steps, total_processing_time, avg_throughput):

        if is_train:
            _message = "          Training Performance Summary           "

        else:
            _message = "         Evaluation Performance Summary          "

        print()  # Visual Spacing
        self.log_info("# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #")
        self.log_info(_message)
        self.log_info("# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #")

        total_processing_hours, rem = divmod(total_processing_time, 3600)
        total_processing_minutes, total_processing_seconds = divmod(rem, 60)

        print()  # Visual Spacing
        total_processing_time = total_processing_hours * 3600 + int(total_processing_minutes) * 60 + int(total_processing_seconds)
        dllogger.log(step=(), data={
            "Average_throughput": avg_throughput,
            "Total processed steps": int(total_steps), 
            "Total_processing_time": total_processing_time }, verbosity=Verbosity.DEFAULT)
        
        self.log_info("Average throughput: {throughput:.1f} samples/sec".format(throughput=avg_throughput))
        self.log_info("Total processed steps: {total_steps}".format(total_steps=total_steps))
        self.log_info(
            "Total processing time: {hours}h {minutes:02d}m {seconds:02d}s".format(
                hours=total_processing_hours,
                minutes=int(total_processing_minutes),
                seconds=int(total_processing_seconds)
            )
        )

        self.log_info("==================== Metrics ====================")
