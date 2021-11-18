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

import json
import tabulate

archs = ["a100", "v100"]
precs = ["full", "half"]

for arch in archs:
    for prec in precs:
        filename = f"inference/{arch}_{prec}.log"
        with open(filename) as opened:
            line = opened.readlines()[-1]
        log = json.loads(line[len("DLLL "):])['data']
        print(log)
        batch_sizes = [1024, 4096, 16384, 65536, 262144, 1048576]
        t_avg = "batch_{}_mean_throughput"
        l_mean = "batch_{}_mean_latency"
        l_90 = "batch_{}_p90_latency"
        l_95 = "batch_{}_p95_latency"
        l_99 = "batch_{}_p99_latency"
        headers = ["Batch size", "Throughput Avg", "Latency Avg", "Latency 90%", "Latency 95%", "Latency 99%"]
        table = []
        for bsize in batch_sizes:
            table.append([bsize,
                          "{:3.3f}".format(log[t_avg.format(bsize)]),
                          "{:.6f}".format(log[l_mean.format(bsize)]),
                          "{:.6f}".format(log[l_90.format(bsize)]),
                          "{:.6f}".format(log[l_95.format(bsize)]),
                          "{:.6f}".format(log[l_99.format(bsize)])])
        print(filename)
        print(tabulate.tabulate(table, headers, tablefmt='pipe'))
