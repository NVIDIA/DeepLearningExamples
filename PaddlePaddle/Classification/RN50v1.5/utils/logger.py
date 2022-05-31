# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

import logging
import paddle.distributed as dist
import dllogger


def format_step(step):
    """
    Define prefix for different prefix message for dllogger.
    Args:
        step(str|tuple): Dllogger step format.
    Returns:
        s(str): String to print in log.
    """
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += f"Epoch: {step[0]} "
    if len(step) > 1:
        s += f"Iteration: {step[1]} "
    if len(step) > 2:
        s += f"Validation Iteration: {step[2]} "
    if len(step) == 0:
        s = "Summary:"
    return s


def setup_dllogger(log_file):
    """
    Setup logging and dllogger.
    Args:
        log_file(str): Path to log file.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='{asctime}:{levelname}: {message}',
        style='{')
    if dist.get_rank() == 0:
        dllogger.init(backends=[
            dllogger.StdOutBackend(
                dllogger.Verbosity.DEFAULT, step_format=format_step),
            dllogger.JSONStreamBackend(dllogger.Verbosity.VERBOSE, log_file),
        ])
    else:
        dllogger.init([])
