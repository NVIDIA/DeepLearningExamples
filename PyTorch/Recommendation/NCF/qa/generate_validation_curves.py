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


def get_curve(filename):
    hrs = []
    with open(filename, 'r') as opened:
        for line in opened.readlines():
            d = json.loads(line[len("DLLL "):])
            try:
                hrs.append(d["data"]["hr@10"])
            except KeyError:
                pass
    return hrs


a100 = "runs/pytorch_ncf_A100-SXM4-40GBx{numgpus}gpus_{precision}_{num_run}.json"
v16 = "runs/pytorch_ncf_Tesla V100-SXM2-16GBx{numgpus}gpus_{precision}_{num_run}.json"
v32 = "runs/pytorch_ncf_Tesla V100-SXM2-32GBx{numgpus}gpus_{precision}_{num_run}.json"
dgx2 = "runs/pytorch_ncf_Tesla V100-SXM3-32GBx{numgpus}gpus_{precision}_{num_run}.json"

fp32 = "FP32"
amp = "Mixed (AMP)"
tf32 = "TF32"


def get_accs(arch, numgpu, prec):
    data = [get_curve(arch.format(numgpus=numgpu, num_run=num_run, precision=prec)) for num_run in range(1, 21)]
    return data[0]


def get_plots():
    archs = [dgx2, a100]
    titles = ["DGX2 32GB", "DGX A100 40GB"]
    fullprecs = [fp32, tf32]
    halfprecs = [amp, amp]
    gpuranges = [(1, 8, 16), (1, 8)]
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    plt.subplots_adjust(hspace=0.5)
    for x, prec in enumerate([fullprecs, halfprecs]):
        for i, arch in enumerate(archs):
            for numgpu in gpuranges[i]:
                d = get_accs(arch, numgpu, prec[i])
                axs[x].plot(range(len(d)), d, label=f"{titles[i]} x {numgpu} {prec[i]}")
        axs[x].legend()

    #plt.show()
    plt.savefig("val_curves.png")


if __name__ == "__main__":
    get_plots()
