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
import pandas as pd
import numpy as np
import pickle
import torch
from criterions import QuantileLoss
from triton.deployment_toolkit.core import BaseMetricsCalculator

def update_argparser(parser):
    parser.add_argument("--dataset", type=str, help="Path to dataset to be used", required=True)
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to be used", required=True)



def _unscale_per_id(config, values, ids, scalers):
    # values = values.cpu().numpy()
    num_horizons = config.example_length - config.encoder_length + 1
    flat_values = pd.DataFrame(
            values,
            columns=[f't{j}' for j in range(num_horizons - values.shape[1], num_horizons)]
            )
    flat_values['id'] = ids
    df_list = []
    for idx, group in flat_values.groupby('id'):
        scaler = scalers[idx]
        group_copy = group.copy()
        for col in group_copy.columns:
            if not 'id' in col:
                _col = np.expand_dims(group_copy[col].values, -1)
                _t_col = scaler.inverse_transform(_col)[:,-1]
                group_copy[col] = _t_col
        df_list.append(group_copy)
    flat_values = pd.concat(df_list, axis=0)

    flat_values = flat_values[[col for col in flat_values if not 'id' in col]]
    flat_tensor = torch.from_numpy(flat_values.values)
    return flat_tensor
def _unscale(config, values, scaler):
    # values = values.cpu().numpy()
    num_horizons = config.example_length - config.encoder_length + 1
    flat_values = pd.DataFrame(
            values,
            columns=[f't{j}' for j in range(num_horizons - values.shape[1], num_horizons)]
            )
    for col in flat_values.columns:
        if not 'id' in col:
            _col = np.expand_dims(flat_values[col].values, -1)
            _t_col = scaler.inverse_transform(_col)[:,-1]
            flat_values[col] = _t_col

    flat_values = flat_values[[col for col in flat_values if not 'id' in col]]
    flat_tensor = torch.from_numpy(flat_values.values)
    return flat_tensor

class MetricsCalculator(BaseMetricsCalculator):
    def __init__(self, dataset, checkpoint):
        state_dict = torch.load(os.path.join(checkpoint, "checkpoint.pt"))
        self.config = state_dict['config']
        self.predictions = []
        self.targets = []
        self.ids = []
        self.scalers = pickle.load(open(os.path.join(dataset, 'tgt_scalers.bin'), 'rb'))

    @property
    def metrics(self):
        targets = np.concatenate(self.targets, axis=0)
        # targets = torch.cat(self.targets, dim=0)
        predictions = np.concatenate(self.predictions, axis=0)
        # predictions = torch.cat(self.predictions, dim=0)

        ids = np.concatenate(self.ids, axis=0)
        if self.config.scale_per_id:

            unscaled_predictions = torch.stack(
                [_unscale_per_id(self.config, predictions[:,:,i], ids, self.scalers) for i in range(len(self.config.quantiles))], 
                dim=-1)
            unscaled_targets = _unscale_per_id(self.config, targets[:,:,0], ids, self.scalers).unsqueeze(-1)
        else:
            ids = None
            unscaled_predictions = torch.stack(
                    [_unscale(self.config, predictions[:,:,i], self.scalers['']) for i in range(len(self.config.quantiles))], 
                    dim=-1)
            unscaled_targets = _unscale(self.config, targets[:,:,0], self.scalers['']).unsqueeze(-1)
        
        losses = QuantileLoss(self.config)(unscaled_predictions, unscaled_targets)
        normalizer = unscaled_targets.abs().mean()
        q_risk = 2 * losses / normalizer
        return {'test_p10': q_risk[0].cpu().numpy(), 'test_p50': q_risk[1].cpu().numpy(), 'test_p90': q_risk[2].cpu().numpy()}

    def update(
        self,
        ids,
        y_pred,
        x,
        y_real,
    ):
        #can probably just pass all of this to the evaluator main class
        self.predictions.append(y_pred["target__0"])
        self.targets.append(y_real['target__0'][:,:,0][:,:,np.newaxis])
        self.ids.append(ids)
        # return self.metrics
