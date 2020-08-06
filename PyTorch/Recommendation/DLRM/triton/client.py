#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import json
import sys

import numpy as np
import torch
import tritonhttpclient
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dlrm.data.datasets import SyntheticDataset, SplitCriteoDataset


def get_data_loader(batch_size, *, data_path, model_config):
    with open(model_config.dataset_config) as f:
        categorical_sizes = list(json.load(f).values())
    if data_path:
        data = SplitCriteoDataset(
            data_path=data_path,
            batch_size=batch_size,
            numerical_features=True,
            categorical_features=range(len(categorical_sizes)),
            categorical_feature_sizes=categorical_sizes,
            prefetch_depth=1
        )
    else:
        data = SyntheticDataset(
            num_entries=batch_size * 1024,
            batch_size=batch_size,
            numerical_features=model_config.num_numerical_features,
            categorical_feature_sizes=categorical_sizes,
            device="cpu"
        )

    return torch.utils.data.DataLoader(data,
                                       batch_size=None,
                                       num_workers=0,
                                       pin_memory=False)


def run_infer(model_name, model_version, numerical_features, categorical_features, headers=None):
    inputs = []
    outputs = []
    num_type = "FP16" if numerical_features.dtype == np.float16 else "FP32"
    inputs.append(tritonhttpclient.InferInput('input__0', numerical_features.shape, num_type))
    inputs.append(tritonhttpclient.InferInput('input__1', categorical_features.shape, "INT64"))

    # Initialize the data
    inputs[0].set_data_from_numpy(numerical_features, binary_data=True)
    inputs[1].set_data_from_numpy(categorical_features, binary_data=False)

    outputs.append(tritonhttpclient.InferRequestedOutput('output__0', binary_data=True))
    results = triton_client.infer(model_name,
                                  inputs,
                                  model_version=str(model_version) if model_version != -1 else '',
                                  outputs=outputs,
                                  headers=headers)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--triton-server-url',
                        type=str,
                        required=True,
                        help='URL adress of triton server (with port)')
    parser.add_argument('--triton-model-name', type=str, required=True,
                        help='Triton deployed model name')
    parser.add_argument('--triton-model-version', type=int, default=-1,
                        help='Triton model version')
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-H', dest='http_headers', metavar="HTTP_HEADER",
                        required=False, action='append',
                        help='HTTP headers to add to inference server requests. ' +
                        'Format is -H"Header:Value".')

    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--inference_data", type=str,
                        help="Path to file with inference data.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Inference request batch size")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use 16bit for numerical input")

    FLAGS = parser.parse_args()
    try:
        triton_client = tritonhttpclient.InferenceServerClient(url=FLAGS.triton_server_url, verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    if FLAGS.http_headers is not None:
        headers_dict = {l.split(':')[0]: l.split(':')[1]
                        for l in FLAGS.http_headers}
    else:
        headers_dict = None

    triton_client.load_model(FLAGS.triton_model_name)
    if not triton_client.is_model_ready(FLAGS.triton_model_name):
        sys.exit(1)

    dataloader = get_data_loader(FLAGS.batch_size,
                                 data_path=FLAGS.inference_data,
                                 model_config=FLAGS)
    results = []
    tgt_list = []

    for numerical_features, categorical_features, target in tqdm(dataloader):
        numerical_features = numerical_features.cpu().numpy()
        numerical_features = numerical_features.astype(np.float16 if FLAGS.fp16 else np.float32)
        categorical_features = categorical_features.long().cpu().numpy()

        output = run_infer(FLAGS.triton_model_name, FLAGS.triton_model_version,
                           numerical_features, categorical_features, headers_dict)

        results.append(output.as_numpy('output__0'))
        tgt_list.append(target.cpu().numpy())

    results = np.concatenate(results).squeeze()
    tgt_list = np.concatenate(tgt_list)

    score = roc_auc_score(tgt_list, results)
    print(F"Model score: {score}")

    statistics = triton_client.get_inference_statistics(model_name=FLAGS.triton_model_name, headers=headers_dict)
    print(statistics)
    if len(statistics['model_stats']) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)
