""" PyTorch EfficientDet support benches

Hacked together by Ross Wightman
"""

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

import torch
import torch.nn as nn
from utils.model_ema import ModelEma
from .anchors import Anchors, AnchorLabeler, generate_detections, MAX_DETECTION_POINTS
from .loss import DetectionLoss


def _post_process(config, cls_outputs, box_outputs):
    """Selects top-k predictions.

    Post-proc code adapted from Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
    and optimized for PyTorch.

    Args:
        config: a parameter dictionary that includes `min_level`, `max_level`,  `batch_size`, and `num_classes`.

        cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].

        box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width, num_anchors * 4].
    """
    batch_size = cls_outputs[0].shape[0]
    if config.fused_focal_loss:
        batch_size, channels, _, _ = cls_outputs[0].shape
        padded_classes = (config.num_classes + 7) // 8 * 8
        anchors = channels // padded_classes
        _cls_outputs_all = []
        for level in range(config.num_levels):
            _, _, height, width = cls_outputs[level].shape
            _cls_output = cls_outputs[level].permute(0, 2, 3, 1)
            _cls_output = _cls_output.view(batch_size, height, width, anchors, padded_classes)
            _cls_output = _cls_output[..., :config.num_classes]
            _cls_output = _cls_output.reshape([batch_size, -1, config.num_classes])
            _cls_outputs_all.append(_cls_output)
        cls_outputs_all = torch.cat(_cls_outputs_all, 1)
    else:
        cls_outputs_all = torch.cat([
            cls_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, config.num_classes])
            for level in range(config.num_levels)], 1)

    box_outputs_all = torch.cat([
        box_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, 4])
        for level in range(config.num_levels)], 1)

    _, cls_topk_indices_all = torch.topk(cls_outputs_all.reshape(batch_size, -1), dim=1, k=MAX_DETECTION_POINTS, sorted=False)
    indices_all = cls_topk_indices_all // config.num_classes
    classes_all = cls_topk_indices_all % config.num_classes

    box_outputs_all_after_topk = torch.gather(
        box_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, 4))

    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, config.num_classes))
    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all_after_topk, 2, classes_all.unsqueeze(2))

    return cls_outputs_all_after_topk, box_outputs_all_after_topk, indices_all, classes_all


def _batch_detection(batch_size: int, class_out, box_out, anchor_boxes, indices, classes, img_scale, img_size, soft_nms: bool = False):
    batch_detections = []
    # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
    for i in range(batch_size):
        detections = generate_detections(
            class_out[i], box_out[i], anchor_boxes, indices[i], classes[i], img_scale[i], img_size[i], soft_nms=soft_nms)
        batch_detections.append(detections)
    return torch.stack(batch_detections, dim=0)


class DetBenchPredict(nn.Module):
    def __init__(self, model, config, soft_nms=False):
        super(DetBenchPredict, self).__init__()
        self.config = config
        self.model = model
        self.soft_nms = soft_nms
        self.anchors = Anchors(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)

    def forward(self, x, img_scales, img_size):
        class_out, box_out = self.model(x)
        class_out, box_out, indices, classes = _post_process(self.config, class_out, box_out)
        return _batch_detection(
            x.shape[0], class_out, box_out, self.anchors.boxes, indices, classes, img_scales, img_size, self.soft_nms)


class DetBenchTrain(nn.Module):
    def __init__(self, model, config):
        super(DetBenchTrain, self).__init__()
        self.config = config
        self.model = model
        self.anchors = Anchors(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)
        self.loss_fn = DetectionLoss(self.config)

    def forward(self, x, target):
        class_out, box_out = self.model(x)
        loss, class_loss, box_loss = self.loss_fn(class_out, box_out, target, target['num_positives'])
        output = dict(loss=loss, class_loss=class_loss, box_loss=box_loss)
        if not self.training:
            # if eval mode, output detections for evaluation
            class_out, box_out, indices, classes = _post_process(self.config, class_out, box_out)
            output['detections'] = _batch_detection(
                x.shape[0], class_out, box_out, self.anchors.boxes, indices, classes,
                target['img_scale'], target['img_size'])
        return output


def unwrap_bench(model):
    # Unwrap a model in support bench so that various other fns can access the weights and attribs of the
    # underlying model directly
    if isinstance(model, ModelEma):  # unwrap ModelEma
        return unwrap_bench(model.ema)
    elif hasattr(model, 'module'):  # unwrap DDP
        return unwrap_bench(model.module)
    elif hasattr(model, 'model'):  # unwrap Bench -> model
        return unwrap_bench(model.model)
    else:
        return model
