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

import numpy as np
import json
from timm.data import create_dataset, create_loader
import torch
def update_argparser(parser):
    parser.add_argument(
        "--config", type=str, required=True, help="Network to deploy")
    parser.add_argument("--val-path", type=str, help="Path to dataset to be used", required=True)
    parser.add_argument("--batch-size", type=int, help="Batch size to use", default=10)
    parser.add_argument("--precision", type=str, default="fp32", 
                        choices=["fp32", "fp16"], help="Inference precision")
    parser.add_argument(
        "--is-prunet", type=bool, required=True, help="Bool on whether network is a prunet")


def get_dataloader_fn(config, val_path, batch_size, precision, is_prunet):
    imagenet_val_path = val_path
    dataset = create_dataset( root=imagenet_val_path, name='', split='validation', load_bytes=False, class_map='')

    with open(config) as configFile:
        modelJSON = json.load(configFile)
        configFile.close()
    config = modelJSON
    assert len(config) > 0
    dataLayer = config[0]
    assert dataLayer['layer_type'] == 'data'
    assert dataLayer['img_resolution'] > 0
    imgRes = dataLayer['img_resolution']
    crop_pct = 1.0
    if is_prunet == "True":
        crop_pct = 0.875
    data_config = {'input_size': (3, imgRes, imgRes), 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'crop_pct': crop_pct}
    batch_size = int(batch_size)
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=batch_size,
        use_prefetcher=True,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=1,
        crop_pct=data_config['crop_pct'],
        pin_memory=False,
        tf_preprocessing=False)
    dtype = precision
    if dtype == 'fp16':
        dtype = torch.float16
    elif dtype == 'fp32':
        dtype = torch.float32
    else:
        raise NotImplementedError
    def _get_dataloader():
            for batch_idx, (input, target) in enumerate(loader):
                x = {"INPUT__0": input.to(dtype).cpu().numpy()}
                y_real = {"OUTPUT__0": np.tile(target.to(dtype).cpu().numpy()[:, np.newaxis], (1, 1000))}
                ids = np.tile(batch_idx, target.shape[0])
                yield (ids, x, y_real)


    return _get_dataloader