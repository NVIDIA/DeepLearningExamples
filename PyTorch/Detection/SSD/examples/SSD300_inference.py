# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
import numpy as np

from apex.fp16_utils import network_to_half

from dle.inference import prepare_input
from src.model import SSD300, ResNet
from src.utils import dboxes300_coco, Encoder


def load_checkpoint(model, model_file):
    cp = torch.load(model_file)['model']
    model.load_state_dict(cp)


def build_predictor(model_file, backbone='resnet50'):
    ssd300 = SSD300(backbone=ResNet(backbone))
    load_checkpoint(ssd300, model_file)

    return ssd300


def prepare_model(checkpoint_path):
    ssd300 = build_predictor(checkpoint_path)
    ssd300 = ssd300.cuda()
    ssd300 = network_to_half(ssd300)
    ssd300 = ssd300.eval()

    return ssd300


def prepare_tensor(inputs):
    NHWC = np.array(inputs)
    NCHW = np.swapaxes(np.swapaxes(NHWC, 2, 3), 1, 2)
    tensor = torch.from_numpy(NCHW)
    tensor = tensor.cuda()
    tensor = tensor.half()

    return tensor


def decode_results(predictions):
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    ploc, plabel = [val.float() for val in predictions]
    results = encoder.decode_batch(ploc, plabel, criteria=0.5, max_output=20)

    return [ [ pred.detach().cpu().numpy()
               for pred in detections
             ]
             for detections in results
           ]


def pick_best(detections, treshold):
    bboxes, classes, confidences = detections
    best = np.argwhere(confidences > 0.3).squeeze(axis=1)

    return [pred[best] for pred in detections]


def main(checkpoint_path, imgs):
    inputs = [prepare_input(uri) for uri in imgs]
    tensor = prepare_tensor(inputs)
    ssd300 = prepare_model(checkpoint_path)

    predictions = ssd300(tensor)

    results = decode_results(predictions)
    best_results = [pick_best(detections, treshold=0.3) for detections in results]
    return best_results

if __name__ == '__main__':
    best_results = main(
            checkpoint_path='/checkpoints/SSD300v1.1.pt',
            imgs=[ 'http://images.cocodataset.org/val2017/000000397133.jpg',
                   'http://images.cocodataset.org/val2017/000000037777.jpg',
                   'http://images.cocodataset.org/val2017/000000252219.jpg',
                 ]
    )
    print(best_results)
