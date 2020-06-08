# !/usr/bin/env python
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

import dllogger as Logger


def format_step(step):
    if isinstance(step, str):
        return step

    if isinstance(step, int):
        return "Iteration: {} ".format(step)

    s = ""

    if len(step) > 0:
        s += "Epoch: {} ".format(step[0])

    if len(step) > 1:
        s += "Iteration: {} ".format(step[1])

    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])

    return s


def init_dllogger(log_dir):
    Logger.init([
        Logger.StdOutBackend(Logger.Verbosity.DEFAULT, step_format=format_step),
        Logger.JSONStreamBackend(Logger.Verbosity.VERBOSE, log_dir)
    ])
