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


def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s


l = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT, step_format=format_step),
        JSONStreamBackend(Verbosity.VERBOSE, "tmp.json"),
    ]
)

# You can log metrics in separate calls
l.log(step="PARAMETER", data={"HP1": 17}, verbosity=Verbosity.DEFAULT)
l.log(step="PARAMETER", data={"HP2": 23}, verbosity=Verbosity.DEFAULT)
# or together
l.log(step="PARAMETER", data={"HP3": 1, "HP4": 2}, verbosity=Verbosity.DEFAULT)

l.metadata("loss", {"unit": "nat", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
l.metadata("val.loss", {"unit": "nat", "GOAL": "MINIMIZE", "STAGE": "VAL"})
l.metadata(
    "speed",
    {"unit": "speeds/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"},
)

for epoch in range(0, 2):

    for it in range(0, 10):
        l.log(
            step=(epoch, it),
            data={"loss": 130 / (1 + epoch * 10 + it)},
            verbosity=Verbosity.DEFAULT,
        )
        if it % 3 == 0:
            for vit in range(0, 3):
                l.log(
                    step=(epoch, it, vit),
                    data={"val.loss": 230 / (1 + epoch * 10 + it + vit)},
                    verbosity=Verbosity.DEFAULT,
                )

    l.log(step=(epoch,), data={"speed": 10}, verbosity=Verbosity.DEFAULT)
l.flush()
