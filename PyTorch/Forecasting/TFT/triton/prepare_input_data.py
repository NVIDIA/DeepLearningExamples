# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from configuration import ElectricityConfig
from data_utils import TFTDataset
import argparse
from deployment_toolkit.dump import JsonDumpWriter

def _verify_and_format_dump(**x):
    data = {}
    for k, v in x.items():
        temp_data = {}
        for i in range(v.shape[1]):
            temp_data["INPUT" + str(i)] = v[:,i]
        data[k] = temp_data
    return data

def main():
    args = _parse_args()
    state_dict = torch.load(os.path.join(args.checkpoint, "checkpoint.pt"))
    config = state_dict['config']
    test_split = TFTDataset(os.path.join(args.dataset, "test.csv"), config)
    data_loader = DataLoader(test_split, batch_size=args.batch_size, num_workers=2)
    input_names_dict = {'s_cat': 's_cat__0', 's_cont':'s_cont__1', 'k_cat':'k_cat__2', 'k_cont':'k_cont__3', 'o_cat':'o_cat__4', 'o_cont':'o_cont__5', 'target':'target__6', 'id':'id__7'}
    reshaper = [-1] + [1]
    for step, batch in enumerate(data_loader):
        bs = batch['target'].shape[0]
        x = {input_names_dict[key]: tensor.numpy() if tensor.numel() else np.ones([bs]).reshape(reshaper) for key, tensor in batch.items()}
        ids = batch['id'][:,0,:].numpy()
        y_real = {'target__0':batch['target'][:,config.encoder_length:,:].numpy()}
        break


    import json
    data = {"data": [{k: {"content": v[i].flatten().tolist(), "shape": list(v[i].shape), "dtype": str(v[i].dtype)} for k, v in x.items()} for i in range(args.batch_size)]}
    with open(os.path.join(args.input_data_dir, "data.json"), "w") as f:
        f.write(json.dumps(data))
        f.close()
    # d = json.load(f)
    # print(d)



def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", required=False, default=1)
    parser.add_argument("--dataset", help="Path to dataset", required=True)
    parser.add_argument("--input-data-dir", help="Path to output folder", required=True)


    args, *_ = parser.parse_known_args()
    args = parser.parse_args()

    return args
    

if __name__ == "__main__":
    main()