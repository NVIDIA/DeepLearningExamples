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

import torch

from dlrm.data import data_loader
from dlrm.data.synthetic_dataset import SyntheticDataset

from tqdm import tqdm
from tensorrtserver.api import *

from sklearn.metrics import roc_auc_score
from functools import partial

def get_data_loader(batch_size, *, data_file, model_config):
    with open(model_config.dataset_config) as f:
        categorical_sizes = list(json.load(f).values())
    if data_file:
        data = data_loader.CriteoBinDataset(data_file=data_file,
                batch_size=batch_size, subset=None,
                numerical_features=model_config.num_numerical_features,
                categorical_features=len(categorical_sizes),
                online_shuffle=False)
    else:
        data = SyntheticDataset(num_entries=batch_size * 1024, batch_size=batch_size,
                dense_features=model_config.num_numerical_features,
                categorical_feature_sizes=categorical_sizes,
                device="cpu")

    return torch.utils.data.DataLoader(data,
                                       batch_size=None,
                                       num_workers=0,
                                       pin_memory=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triton-server-url", type=str, required=True, 
                        help="URL adress of trtion server (with port)")
    parser.add_argument("--triton-model-name", type=str, required=True,
                        help="Triton deployed model name")
    parser.add_argument("--triton-model-version", type=int, default=-1,
                        help="Triton model version")
    parser.add_argument("--protocol", type=str, default="HTTP",
                        help="Communication protocol (HTTP/GRPC)")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Verbose mode.")
    parser.add_argument('-H', dest='http_headers', metavar="HTTP_HEADER",
                        required=False, action='append',
                        help='HTTP headers to add to inference server requests. ' +
                        'Format is -H"Header:Value".')

    parser.add_argument("--num_numerical_features", type=int, default=13)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--inference_data", type=str, 
                        help="Path to file with inference data.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Inference request batch size")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use 16bit for numerical input")
    FLAGS = parser.parse_args()

    FLAGS.protocol = ProtocolType.from_str(FLAGS.protocol)
    
    # Create a health context, get the ready and live state of server.
    health_ctx = ServerHealthContext(FLAGS.triton_server_url, FLAGS.protocol, 
                                     http_headers=FLAGS.http_headers, verbose=FLAGS.verbose)
    print("Health for model {}".format(FLAGS.triton_model_name))
    print("Live: {}".format(health_ctx.is_live()))
    print("Ready: {}".format(health_ctx.is_ready()))
    
    with ModelControlContext(FLAGS.triton_server_url, FLAGS.protocol) as ctx:
        ctx.load(FLAGS.triton_model_name)

    # Create a status context and get server status
    status_ctx = ServerStatusContext(FLAGS.triton_server_url, FLAGS.protocol, FLAGS.triton_model_name, 
                                     http_headers=FLAGS.http_headers, verbose=FLAGS.verbose)
    print("Status for model {}".format(FLAGS.triton_model_name))
    print(status_ctx.get_server_status())
    
    # Create the inference context for the model.
    infer_ctx = InferContext(FLAGS.triton_server_url, FLAGS.protocol, FLAGS.triton_model_name, 
                             FLAGS.triton_model_version, 
                             http_headers=FLAGS.http_headers, verbose=FLAGS.verbose)

    dataloader = get_data_loader(FLAGS.batch_size, 
                                 data_file=FLAGS.inference_data,
                                 model_config=FLAGS)

    results = []
    tgt_list = []

    for num, cat, target in tqdm(dataloader):
        num = num.cpu().numpy()
        if FLAGS.fp16:
            num = num.astype(np.float16)
        cat = cat.long().cpu().numpy()

        input_dict = {"input__0": tuple(num[i] for i in range(len(num))),
                      "input__1": tuple(cat[i] for i in range(len(cat)))}
        output_keys = ["output__0"]
        output_dict = {x: InferContext.ResultFormat.RAW for x in output_keys}

        result = infer_ctx.run(input_dict, output_dict, len(num))
        results.append(result["output__0"])
        tgt_list.append(target.cpu().numpy())

    results = np.concatenate(results).squeeze()
    tgt_list = np.concatenate(tgt_list)

    score = roc_auc_score(tgt_list, results)
    print(F"Model score: {score}")

    with ModelControlContext(FLAGS.triton_server_url, FLAGS.protocol) as ctx:
        ctx.unload(FLAGS.triton_model_name)




