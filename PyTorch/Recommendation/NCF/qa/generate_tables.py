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
import numpy as np


def get_training_data(filename):
    with open(filename, 'r') as opened:
        line = opened.readlines()[-1]
    json_content = line[len("DLLL "):]
    data = json.loads(json_content)["data"]
    with open(filename, 'r') as opened:
        for line in opened.readlines():
            d = json.loads(line[len("DLLL "):])
            if d.get("step", "") == "PARAMETER":
                data['batch_size'] = d["data"]["batch_size"]
    return data


a100 = "runs/pytorch_ncf_A100-SXM4-40GBx{numgpus}gpus_{precision}_{num_run}.json"
v16 = "runs/pytorch_ncf_Tesla V100-SXM2-16GBx{numgpus}gpus_{precision}_{num_run}.json"
v32 = "runs/pytorch_ncf_Tesla V100-SXM2-32GBx{numgpus}gpus_{precision}_{num_run}.json"
dgx2 = "runs/pytorch_ncf_Tesla V100-SXM3-32GBx{numgpus}gpus_{precision}_{num_run}.json"

fp32 = "FP32"
amp = "Mixed (AMP)"
tf32 = "TF32"

first = a100.format(numgpus=1, precision=fp32, num_run=1)

timevar = 'time_to_target' #"time_to_best_model"


def get_acc_table(arch, numgpus, fullprec):
    headers = ["GPUs", "Batch size / GPU", f"Accuracy - {fullprec}", "Accuracy - mixed precision", f"Time to train - {fullprec}", "Time to train - mixed precision", f"Time to train speedup ({fullprec} to mixed precision)"]
    table = []
    for numgpus in numgpus:
        data_full = [get_training_data(arch.format(numgpus=numgpus, num_run=num_run, precision=fullprec)) for num_run in range(1, 21)]
        data_mixed = [get_training_data(arch.format(numgpus=numgpus, num_run=num_run, precision=amp)) for num_run in range(1, 21)]
        bsize = data_full[0]['batch_size']/numgpus
        accs_full = np.mean([d["best_accuracy"] for d in data_full])
        accs_mixed = np.mean([d["best_accuracy"] for d in data_mixed])
        time_full = np.mean([d[timevar] for d in data_full])
        time_mixed = np.mean([d[timevar] for d in data_mixed])
        speedup = time_full / time_mixed
        row = [numgpus, int(bsize),
               "{:.6f}".format(accs_full),
               "{:.6f}".format(accs_mixed),
               "{:.6f}".format(time_full),
               "{:.6f}".format(time_mixed),
               "{:.2f}".format(speedup)]
        table.append(row)
    print(tabulate.tabulate(table, headers, tablefmt='pipe'))


def get_perf_table(arch, numgpus, fullprec):
    headers = ["GPUs",
               "Batch size / GPU",
               f"Throughput - {fullprec} (samples/s)",
               "Throughput - mixed precision (samples/s)",
               f"Throughput speedup ({fullprec} to mixed precision)",
               f"Strong scaling - {fullprec}",
               "Strong scaling - mixed precision",
               ]
    table = []
    base_full = None
    base_mixed = None
    for numgpus in numgpus:
        data_full = [get_training_data(arch.format(numgpus=numgpus, num_run=num_run, precision=fullprec)) for num_run in range(1, 21)]
        data_mixed = [get_training_data(arch.format(numgpus=numgpus, num_run=num_run, precision=amp)) for num_run in range(1, 21)]
        bsize = data_full[0]['batch_size']/numgpus
        _full = np.mean([d["best_train_throughput"] for d in data_full])
        _mixed = np.mean([d["best_train_throughput"] for d in data_mixed])
        if numgpus == 1:
            base_full = _full
            base_mixed = _mixed
        scaling_full = _full/ base_full
        scaling_mixed = _mixed / base_mixed
        time_mixed = np.mean([d[timevar] for d in data_mixed])
        speedup = _full / _mixed
        row = [numgpus, int(bsize),
               "{:.2f}M".format(_full / 10**6),
               "{:.2f}M".format(_mixed / 10**6),
               "{:.2f}".format(speedup),
               "{:.2f}".format(scaling_full),
               "{:.2f}".format(scaling_mixed)]

        table.append(row)
    print(tabulate.tabulate(table, headers, tablefmt='pipe'))


#get_acc_table(a100, (1, 8), tf32)
#get_acc_table(v16, (1, 8), fp32)
#get_acc_table(v32, (1, 8), fp32)
#get_acc_table(dgx2, (1, 8, 16), fp32)

#get_perf_table(a100, (1, 8), tf32)
#get_perf_table(v16, (1, 8), fp32)
#get_perf_table(v32, (1, 8), fp32)
#get_perf_table(dgx2, (1, 8, 16), fp32)