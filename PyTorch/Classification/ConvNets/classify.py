# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from PIL import Image
import argparse
import numpy as np
import json
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import image_classification.resnet as models
from image_classification.dataloaders import load_jpeg_from_file

try:
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )


def add_parser_arguments(parser):
    model_names = models.resnet_versions.keys()
    model_configs = models.resnet_configs.keys()
    parser.add_argument("--image-size", default="224", type=int)
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet50",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
    )
    parser.add_argument(
        "--model-config",
        "-c",
        metavar="CONF",
        default="classic",
        choices=model_configs,
        help="model configs: " + " | ".join(model_configs) + "(default: classic)",
    )
    parser.add_argument("--weights", metavar="<path>", help="file with model weights")
    parser.add_argument(
        "--precision", metavar="PREC", default="FP16", choices=["AMP", "FP16", "FP32"]
    )
    parser.add_argument("--image", metavar="<path>", help="path to classified image")


def main(args):
    imgnet_classes = np.array(json.load(open("./LOC_synset_mapping.json", "r")))
    model = models.build_resnet(args.arch, args.model_config, verbose=False)

    if args.weights is not None:
        weights = torch.load(args.weights)
        model.load_state_dict(weights)

    model = model.cuda()

    if args.precision == "FP16":
        model = network_to_half(model)

    model.eval()

    with torch.no_grad():
        input = load_jpeg_from_file(args.image, cuda=True, fp16=args.precision!='FP32')

        output = torch.nn.functional.softmax(model(input), dim=1).cpu().view(-1).numpy()
        top5 = np.argsort(output)[-5:][::-1]

        print(args.image)
        for c, v in zip(imgnet_classes[top5], output[top5]):
            print(f"{c}: {100*v:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    add_parser_arguments(parser)
    args = parser.parse_args()

    cudnn.benchmark = True

    main(args)
