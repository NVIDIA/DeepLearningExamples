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


# Report JSON file structure:
# - "model"          : architecture of the model (e.g. "resnet50").
# - "ngpus"          : number of gpus on which training was performed.
# - "total_duration" : total duration of training in seconds.
# - "cmd"            : list of application arguments.
# - "metrics"        : per epoch metrics for train and validation
#                      (some of below metrics may not exist in the report,
#                       depending on application arguments)
#       - "train.top1"        : training top1 accuracy in epoch.
#       - "train.top5"        : training top5 accuracy in epoch.
#       - "train.loss"        : training loss in epoch.
#       - "train.total_ips"   : training speed (data and compute time taken into account) for epoch in images/sec.
#       - "train.latency_avg" : average latency of one iteration in seconds.
#       - "train.latency_50"  : median latency of one iteration in seconds.
#       - "train.latency_90"  : 90th percentile latency of one iteration in seconds.
#       - "train.latency_95"  : 95th percentile latency of one iteration in seconds.
#       - "train.latency_99"  : 99th percentile latency of one iteration in seconds.
#       - "train.latency_100" : highest observed latency of one iteration in seconds.
#       - "val.top1", "val.top5", "val.time", "val.total_ips", "val.latency_avg", "val.latency_50",
#         "val.latency_90", "val.latency_95", "val.latency_99", "val.latency_100"      : the same but for validation.

import json
from collections import OrderedDict

class Report:
    def __init__(self, model_name, ngpus, cmd):
        self.model_name = model_name
        self.ngpus = ngpus
        self.cmd = cmd
        self.total_duration = 0
        self.metrics = OrderedDict()

    def add_value(self, metric, value):
        if metric not in self.metrics:
            self.metrics[metric] = []
        self.metrics[metric].append(value)

    def set_total_duration(self, duration):
        self.total_duration = duration

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(self.get_report())

    def get_report(self):
        report = OrderedDict([
            ('model', self.model_name),
            ('ngpus', self.ngpus),
            ('total_duration', self.total_duration),
            ('cmd', self.cmd),
            ('metrics', self.metrics),
        ])
        return json.dumps(report, indent=4)
