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
import dgl

from data.datasets import get_collate_fn
from data.data_utils import Preprocessor

def update_argparser(parser):
    parser.add_argument("--model-dir", type=str, help="Path to the model directory you would like to use (likely in outputs)", required=True)
    parser.add_argument("--batch-size", type=int, required=True)



def get_dataloader_fn(model_dir, batch_size):
    with open(os.path.join(model_dir, ".hydra/config_merged.yaml"), "rb") as f:
        config = OmegaConf.load(f)
    if config.inference.get("dataset_path", None):
        preprocessor = Preprocessor(config.dataset.config)
        if config.inference.get("preproc_state_path", None):
            preprocessor_state_file = config.inference.preproc_state_path
        else:
            preprocessor_state_file = None
        preprocessor.load_state(preprocessor_state_file)
        test_df = preprocessor.preprocess_test(dataset=config.inference.dataset_path)
        test_df = preprocessor.apply_scalers(test_df)
        test_df = preprocessor.impute(test_df)
        train, valid, test = hydra.utils.call(config.dataset, input_df=test_df)
    else:
        train, valid, test = hydra.utils.call(config.dataset)
    del train
    del valid
    input_names_dict = {'s_cat': 's_cat__0', 's_cont':'s_cont__1', 'k_cat':'k_cat__2', 'k_cont':'k_cont__3', 'o_cat':'o_cat__4', 'o_cont':'o_cont__5', 'target':'target__6', 'sample_weight': 'sample_weight__7', 'id':'id__8'}
    if config.model.config.get("quantiles", None):
        tile_num = len(config.model.config.quantiles)
    else:
        tile_num = 1
    
    data_loader = DataLoader(
        test,
        batch_size=int(batch_size),
        num_workers=1,
        pin_memory=True,
        collate_fn=get_collate_fn(config.trainer.config.model_type, config.trainer.config.encoder_length, test=True),
    )
    def _get_dataloader():
        for step, (batch, labels, _) in enumerate(data_loader):
            bs = batch['target'].shape[0]
            x = {input_names_dict[key]: batch[key].numpy() if batch[key].numel() else np.full([bs, 1], np.nan) for key in input_names_dict.keys()}
            weights = batch.ndata['weight'] if isinstance(batch, dgl.DGLGraph) else batch['weight']
            x['weight__9']= weights[:, config.dataset.config.encoder_length:, :].numpy() if weights is not None and weights.numel() else np.full([bs, 1], np.nan)
            ids = batch.ndata['id'] if isinstance(batch, dgl.DGLGraph) else batch["id"]
            ids = ids[:,0].numpy()
            y_real = {'target__0':np.tile(labels.numpy(), (1, 1, tile_num))} #Probably need to expand the final dims here as well
            yield (ids, x, y_real)


    return _get_dataloader
