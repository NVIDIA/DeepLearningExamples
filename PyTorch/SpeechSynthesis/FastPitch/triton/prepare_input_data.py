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

import argparse
import importlib
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataloader', type=str, required=True,
        help='Path to file containing get_dataloader function')
    parser.add_argument('--input-data-dir', type=str, required=True,
        help='Path to directory where input data for perf client will be saved')
    parser.add_argument('--dataset-path', required=False, help='Path to the datset')
    parser.add_argument('--precision', type=str, default="fp16",
                        help='Precision for the generated input data')
    parser.add_argument('--length', type=int, required=True,
                        help='Length of the generated input data')

    args = parser.parse_args()
    args.batch_size = 1
    return args

def main():
    args = parse_args()

    spec = importlib.util.spec_from_file_location('dataloader', args.dataloader)
    dm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dm)
    dataloader = dm.get_dataloader_fn(dataset_path=args.dataset_path,
                                      batch_size=1,
                                      precision=args.precision)

    _, x, _ = next(dataloader())
    for name, t in x.items():
        if name == 'INPUT__0':
            if t.shape[1] > args.length:
                t = t[:,:,:args.length]
            elif t.shape[1] < args.length:
                num_tiles = int(np.ceil(1.0*args.length/t.shape[1]))
                t = np.tile(t, (1,1,num_tiles))
                t = t[:,:,:args.length]
        t.tofile(os.path.join(args.input_data_dir, name))


if __name__ == '__main__':
    main()
