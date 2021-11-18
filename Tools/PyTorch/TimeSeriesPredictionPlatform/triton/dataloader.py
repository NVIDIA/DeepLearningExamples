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

import os
import torch
import torch.nn as nn
import hydra
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import OmegaConf

from torch.utils.data.dataloader import default_collate
from functools import partial
import dgl

def update_argparser(parser):
    parser.add_argument("--model-dir", type=str, help="Path to the model directory you would like to use (likely in outputs)", required=True)
    parser.add_argument("--batch-size", type=int, required=True)



def get_dataloader_fn(model_dir, batch_size):
    with open(os.path.join(model_dir, ".hydra/config_merged.yaml"), "rb") as f:
        config = OmegaConf.load(f)
    config._target_ = config.config.dataset._target_
    dataset_dir = config.config.dataset.dest_path
    if os.path.isdir(dataset_dir):
        train, valid, test = hydra.utils.call(config)
        del train
        del valid
    else:
        raise ValueError('dataset_dir must be a directory')
    input_names_dict = {'s_cat': 's_cat__0', 's_cont':'s_cont__1', 'k_cat':'k_cat__2', 'k_cont':'k_cont__3', 'o_cat':'o_cat__4', 'o_cont':'o_cont__5', 'target':'target__6', 'weight':'weight__7', 'sample_weight': 'sample_weight__8', 'id':'id__9'}
    reshaper = [-1] + [1 for i in range(9)]
    test_target = "target_masked" if config.config.model.get("test_target_mask", True) else "target"
    if config.config.model.get("quantiles", None):
        tile_num = len(config.config.model.quantiles)
    else:
        tile_num = 1
    if config.config.dataset.get('graph', False) and config.config.model.get('graph_eligible', False):
        def _collate_graph(samples, target):
            batch = dgl.batch(samples)
            labels = batch.ndata['target']
            # XXX: we need discuss how to do this neatly
            if target == "target_masked":
                labels = labels[:, config.config.dataset.encoder_length:, :]

            return batch, labels

        _collate = _collate_graph
    else:
        def _collate_dict(samples, target):
            batch = default_collate(samples)
            labels = batch['target']
            if target == "target_masked":
                labels = labels[:,config.config.dataset.encoder_length:, :]
            return batch, labels

        _collate = _collate_dict
    data_loader = DataLoader(test, batch_size=int(batch_size), num_workers=2,  pin_memory=True, collate_fn=partial(_collate, target=test_target))
    def _get_dataloader():
        for step, (batch, labels) in enumerate(data_loader):
            bs = batch['target'].shape[0]
            x = {input_names_dict[key]: batch[key].numpy() if batch[key].numel() else np.ones([bs]).reshape(reshaper) for key in input_names_dict.keys()}
            ids = batch['id'][:,0].numpy()
            y_real = {'target__0':np.tile(labels.numpy(), (1, 1, tile_num))} #Probably need to expand the final dims here as well
            yield (ids, x, y_real)


    return _get_dataloader
