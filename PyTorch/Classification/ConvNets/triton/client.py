# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from image_classification.dataloaders import get_pytorch_val_loader

from tqdm import tqdm

import tritongrpcclient
from tritonclientutils import InferenceServerException


def get_data_loader(batch_size, *, data_path):
    valdir = os.path.join(data_path, "val-jpeg")
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
        ),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--triton-server-url",
        type=str,
        required=True,
        help="URL adress of trtion server (with port)",
    )
    parser.add_argument(
        "--triton-model-name",
        type=str,
        required=True,
        help="Triton deployed model name",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Verbose mode."
    )

    parser.add_argument(
        "--inference_data", type=str, help="Path to file with inference data."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Inference request batch size"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use fp16 precision for input data",
    )
    FLAGS = parser.parse_args()

    triton_client = tritongrpcclient.InferenceServerClient(
        url=FLAGS.triton_server_url, verbose=FLAGS.verbose
    )
    dataloader = get_data_loader(FLAGS.batch_size, data_path=FLAGS.inference_data)

    inputs = []
    inputs.append(
        tritongrpcclient.InferInput(
            "input__0",
            [FLAGS.batch_size, 3, 224, 224],
            "FP16" if FLAGS.fp16 else "FP32",
        )
    )

    outputs = []
    outputs.append(tritongrpcclient.InferRequestedOutput("output__0"))

    all_img = 0
    cor_img = 0

    result_prev = None
    for image, target in tqdm(dataloader):
        if FLAGS.fp16:
            image = image.half()
        inputs[0].set_data_from_numpy(image.numpy())

        result = triton_client.infer(
            FLAGS.triton_model_name, inputs, outputs=outputs, headers=None
        )
        result = result.as_numpy("output__0")
        result = np.argmax(result, axis=1)
        cor_img += np.sum(result == target.numpy())
        all_img += result.shape[0]

    acc = cor_img / all_img
    print(f"Final accuracy {acc:.04f}")
