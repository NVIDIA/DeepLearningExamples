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

import os
import torch

def nvidia_gpunet(pretrained=True, **kwargs):
    """Constructs a gpunet model (nn.module with additional infer(input) method).
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com
    Args (type[, default value]):
        pretrained (bool, True): If True, returns a pretrained model. Pretrained only gpunets.
        model_math (str, 'fp32'): returns a model in given precision ('fp32' or 'fp16'). Precision fp32 only gpunets
        model_type (str, 'GPUNet-0'): loads selected model type GPUNet-1.... or GPUNet-P0/P1 or GPUNet-D1/D2. Defaults to GPUNet-0
    """

    from ..models.gpunet_builder import GPUNet_Builder
    from .model_hub import get_configs, MODEL_ZOO_NAME2TYPE_B1
    from timm.models.helpers import load_checkpoint

    modelType = kwargs.get('model_type', 'GPUNet-0')
    print("model_type=", modelType)

    errMsg = "model_type {} not found, available models are {}".format(
        modelType, list(MODEL_ZOO_NAME2TYPE_B1.keys())
    )
    assert modelType in MODEL_ZOO_NAME2TYPE_B1.keys(), errMsg

    is_prunet = False
    if "GPUNet-P0" in modelType or "GPUNet-P1" in modelType: 
        is_prunet = True

    modelLatency = MODEL_ZOO_NAME2TYPE_B1[modelType]
    print("mapped model latency=", modelLatency)
    modelJSON, cpkPath = get_configs(batch=1, latency=modelLatency, gpuType="GV100", download=pretrained, config_root_dir=os.path.dirname(__file__))

    builder = GPUNet_Builder()
    model = builder.get_model(modelJSON)

    if pretrained:
        errMsg = "checkpoint not found at {}, ".format(cpkPath)
        errMsg += "retrieve with get_config_and_checkpoint_files "
        assert os.path.isfile(cpkPath) is True, errMsg
        if is_prunet:
            model.load_state_dict(torch.load(cpkPath))
        else:
            load_checkpoint(model, cpkPath, use_ema=True)

    modelMath = kwargs.get('model_math', 'fp32')
    if modelMath == "fp16":
        model.half()
    
    return model
