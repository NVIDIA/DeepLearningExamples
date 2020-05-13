#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
import numpy

class dllogger_class():

    def format_step(self, step):
        if isinstance(step, str):
            return step
        elif isinstance(step, int):
            return "Iteration: {} ".format(step)
        elif len(step) > 0:
            return "Iteration: {} ".format(step[0])
        else:
            return ""

    def __init__(self, log_path="bert_dllog.json"):
        self.logger = Logger([
            StdOutBackend(Verbosity.DEFAULT, step_format=self.format_step),
            JSONStreamBackend(Verbosity.VERBOSE, log_path),
            ])
        self.logger.metadata("mlm_loss", {"format": ":.4f", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
        self.logger.metadata("nsp_loss", {"format": ":.4f", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
        self.logger.metadata("avg_loss_step", {"format": ":.4f", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
        self.logger.metadata("total_loss", {"format": ":.4f", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
        self.logger.metadata("loss", {"format": ":.4f", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
        self.logger.metadata("f1", {"format": ":.4f", "GOAL": "MINIMIZE", "STAGE": "VAL"})
        self.logger.metadata("precision", {"format": ":.4f", "GOAL": "MINIMIZE", "STAGE": "VAL"})
        self.logger.metadata("recall", {"format": ":.4f", "GOAL": "MINIMIZE", "STAGE": "VAL"})
        self.logger.metadata("mcc", {"format": ":.4f", "GOAL": "MINIMIZE", "STAGE": "VAL"})
        self.logger.metadata("exact_match", {"format": ":.4f", "GOAL": "MINIMIZE", "STAGE": "VAL"})
        self.logger.metadata(
            "throughput_train",
            {"unit": "seq/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"},
        )
        self.logger.metadata(
            "throughput_inf",
            {"unit": "seq/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "VAL"},
        )
