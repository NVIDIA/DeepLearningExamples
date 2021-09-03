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
import matplotlib.pyplot as plt


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


def get_accs(arch, numgpu, prec):
    data = [get_training_data(arch.format(numgpus=numgpu, num_run=num_run, precision=prec)) for num_run in range(1, 21)]
    accs = [d["best_accuracy"] for d in data]
    return accs


def get_plots():
    archs = [dgx2, a100]
    gpuranges = [(1, 8, 16), (1, 8)]
    titles = ["DGX2 32GB", "DGX A100 40GB"]
    fullprecs = [fp32, tf32]
    fig, axs = plt.subplots(2, 3, sharey=True, figsize=(8, 8))
    plt.subplots_adjust(hspace=0.5)
    for x, arch in enumerate(archs):
        gpurange = gpuranges[x]
        for y, gpu in enumerate(gpurange):
            f_data = get_accs(arch, gpu, fullprecs[x])
            h_data = get_accs(arch, gpu, amp)
            axs[x, y].boxplot([f_data, h_data])
            axs[x, y].set_xticklabels([fullprecs[x], amp])
            axs[x, y].set_title(f"{gpu} GPUs" if gpu > 1 else "1 GPU")
        axs[x, 0].set_ylabel(titles[x])
    fig.delaxes(axs[1, 2])
    # plt.show()
    plt.savefig("box_plots.png")


if __name__ == "__main__":
    get_plots()
