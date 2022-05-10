# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

"""
Evaluating the latency and accuracy of GPUNet

--------Configurations of GPUNet--------
## Without distillation
# GPUNet-2
modelJSON, cpkPath = get_configs(batch=1, latency="1.75ms", gpuType="GV100")
# GPUNet-1
modelJSON, cpkPath = get_configs(batch=1, latency="0.85ms", gpuType="GV100")
# GPUNet-0
modelJSON, cpkPath = get_configs(batch=1, latency="0.65ms", gpuType="GV100")

## With distillation
# GPUNet-D2
modelJSON, cpkPath = get_configs(batch=1, latency="2.25ms-D", gpuType="GV100")
# GPUNet-D1
modelJSON, cpkPath = get_configs(batch=1, latency="1.25ms-D", gpuType="GV100")
# GPUNet-P0
modelJSON, cpkPath = get_configs(batch=1, latency="0.5ms-D", gpuType="GV100")
# GPUNet-P1
modelJSON, cpkPath = get_configs(batch=1, latency="0.8ms-D", gpuType="GV100")
----------------------------------------

What can you do?
1. Test GPUNet accuracy.
2. Benchmarking the latency:
    Export GPUNet to ONNX, then 'trtexec --onnx=gpunet.onnx --fp16'.
    We reported the median GPU compute time. Here is an example,
        GPU Compute Time: ..., median = 0.752686 ms, ...
"""

from configs.model_hub import get_configs, get_model_list
from models.gpunet_builder import GPUNet_Builder

modelJSON, cpkPath = get_configs(batch=1, latency="0.65ms", gpuType="GV100")

print(get_model_list(1))

builder = GPUNet_Builder()
model = builder.get_model(modelJSON)
builder.export_onnx(model)
print(model, model.imgRes)

builder.test_model(
    model,
    testBatch=200,
    checkpoint=cpkPath,
    imgRes=(3, model.imgRes, model.imgRes),
    dtype="fp16",
    crop_pct=1,
    val_path="/root/data/imagenet/val",
)
