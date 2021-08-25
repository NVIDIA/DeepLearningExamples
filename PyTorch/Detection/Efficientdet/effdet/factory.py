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

from .model import EfficientDet
from .bench import DetBenchTrain, DetBenchPredict
from .config import get_efficientdet_config
from utils.utils import load_checkpoint, freeze_layers_fn


def create_model(
        model_name, input_size=None, num_classes=None, bench_task='', pretrained=False, checkpoint_path='', checkpoint_ema=False, **kwargs):
    config = get_efficientdet_config(model_name)
    if num_classes is not None:
        config.num_classes = num_classes
    if input_size is not None:
        config.image_size = input_size

    pretrained_backbone_path = kwargs.pop('pretrained_backbone_path', '')
    if pretrained or checkpoint_path:
        pretrained_backbone_path = ''  # no point in loading backbone weights
    strict_load = kwargs.pop('strict_load', True)

    redundant_bias = kwargs.pop('redundant_bias', None)
    if redundant_bias is not None:
        # override config if set to something
        config.redundant_bias = redundant_bias

    soft_nms = kwargs.pop('soft_nms', False)
    config.label_smoothing = kwargs.pop('label_smoothing', 0.1)
    remove_params = kwargs.pop('remove_params', [])
    freeze_layers = kwargs.pop('freeze_layers', [])
    config.fused_focal_loss = kwargs.pop('fused_focal_loss', False)

    model = EfficientDet(config, pretrained_backbone_path=pretrained_backbone_path, **kwargs)

    # FIXME handle different head classes / anchors and re-init of necessary layers w/ pretrained load
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, use_ema=checkpoint_ema, strict=strict_load, remove_params=remove_params)

    if len(freeze_layers) > 0:
        freeze_layers_fn(model, freeze_layers=freeze_layers)

    # wrap model in task specific bench if set
    if bench_task == 'train':
        model = DetBenchTrain(model, config)
    elif bench_task == 'predict':
        model = DetBenchPredict(model, config, soft_nms)
    return model