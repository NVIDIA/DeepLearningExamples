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

import pickle
from abc import ABC

import dgl
import numpy as np
import torch
from data.datasets import get_collate_fn
from distributed_utils import get_mp_context
from torch.utils.data import DataLoader
from training.utils import to_device

from .evaluation_metrics import METRICS
import pandas as pd


class MetricEvaluator(ABC):
    def __init__(self, config):
        self.output_selector = config.get("output_selector", None)
        self.metrics = []
        preprocessor_state = pickle.load(open(config.preprocessor_state_path, "rb"))
        self.scalers = preprocessor_state["scalers"]
        self.save_predictions = config.get("save_predictions", False)
        self.example_history = []

        for name in config.metrics:
            if name not in METRICS:
                raise ValueError(f"No metric of name: {name}")
            self.metrics.append(METRICS[name]())
        self.config = config

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def save_preds(self, preds, ids):
        all_examples = self.example_history 
        all_examples = all_examples.transpose(2,0,1).reshape(-1, all_examples.shape[1])

        if len(preds.shape) == 4:
            tgt_ords = np.arange(preds.shape[2]).repeat(preds.shape[0])
            tgt_ords = pd.DataFrame(tgt_ords, columns=['#target'])
            preds = preds.transpose(2,0,1,3).reshape(-1,preds.shape[1], preds.shape[3])
            ids = ids.transpose().reshape(-1)
        else:
            tgt_ords = None
        all_examples = self.scalers.inverse_transform_targets(all_examples, ids)

        hist_df = pd.DataFrame(all_examples, columns=[f't{i+1}' for i in range(-self.config.encoder_length, 0)])
        ids = pd.DataFrame(ids, columns=['id'])
        col_labels = [f'Estimator{j}_t{i:+}' for j in range(preds.shape[2]) for i in range(preds.shape[1])]
        preds_df = pd.DataFrame(preds.reshape(preds.shape[0],-1, order='F'), columns=col_labels)
        df = pd.concat([ids, tgt_ords, hist_df, preds_df], axis=1)
        df.to_csv('predictions.csv')

    def evaluate(self, preds, labels, ids, weights):
        results = {}

        # In multi target case we treat each target as a separate example.
        # Then we can reduce it to a single target case setting BS = prev_BS * num_targets
        if len(preds.shape) == 4:
            if self.scalers.scale_per_id:
                ids = np.arange(preds.shape[-2])
                ids = np.repeat(ids, preds.shape[0])
            else:
                ids = None
            # TODO: this causes a memory movement. Rewrite this with views!
            preds = np.concatenate([preds[:, :, i] for i in range(preds.shape[-2])], axis=0)
            labels = np.concatenate([labels[:, :, i] for i in range(labels.shape[-1])], axis=0)
            weights = np.concatenate([weights[:, :, i] for i in range(weights.shape[-1])], axis=0)
        elif len(preds.shape) == 3:
            labels = labels.squeeze(-1)
            if weights.size:
                weights = weights.squeeze(-1)
        else:
            raise ValueError("Expected shape of predictions is either BSxTxFxH or BSxTxH")

        upreds = np.stack([self.scalers.inverse_transform_targets(preds[..., i], ids) for i in range(preds.shape[-1])],
                          axis=-1)
        labels = self.scalers.inverse_transform_targets(labels, ids)

        if self.save_predictions:
            self.save_preds(upreds, ids)

        for metric in self.metrics:
            selector = getattr(metric, 'selector', self.output_selector)
            preds = upreds[..., selector]
            results[metric.name] = metric(preds, labels, weights) if np.all(np.isfinite(preds)) else np.NaN
        results = {k: float(v) for k, v in results.items()}
        return results


class CTLMetricEvaluator(MetricEvaluator):
    def __init__(self, test_data, config):
        super().__init__(config)
        self.device = config.device
        if test_data is not None:
            mp_context = get_mp_context()
            self.dataloader = DataLoader(
                test_data,
                batch_size=self.config.batch_size,
                num_workers=1,
                pin_memory=True,
                collate_fn=get_collate_fn(config.model_type, config.encoder_length, test=True),
                multiprocessing_context=mp_context
            )
        else:
            self.dataloader = None

    def prep_data(self, batch):
        ids = batch.ndata['id'] if isinstance(batch, dgl.DGLGraph) else batch["id"]
        ids = ids[:, 0, ...]  # Shape BS x T x F [x H]
        weights = batch.ndata['weight'] if isinstance(batch, dgl.DGLGraph) else batch['weight']
        weights = weights[:, self.config.encoder_length:,
                  :] if weights is not None and weights.numel() else torch.empty(0)
        batch = to_device(batch, device=self.device)

        return batch, weights, ids

    def predict(self, model, dataloader=None):
        if not dataloader:
            dataloader = self.dataloader
        assert dataloader is not None, "Dataloader cannot be None, either pass in a valid dataloader or \
        initialize evaluator with valid test_data"
        test_method_name = 'predict' if hasattr(model, "predict") else '__call__'
        test_method = getattr(model, test_method_name)

        model.eval()

        with torch.no_grad():

            preds_full = []
            labels_full = []
            weights_full = []
            ids_full = []

            for i, (batch, labels, _) in enumerate(dataloader):
                if self.save_predictions:
                    self.example_history.append(batch['target'][:,:self.config.encoder_length].detach().cpu())
                batch, weights, ids = self.prep_data(batch)

                labels_full.append(labels)
                weights_full.append(weights)
                preds = test_method(batch)
                ids_full.append(ids)
                preds_full.append(preds)

            preds_full = torch.cat(preds_full, dim=0).cpu().numpy()
            labels_full = torch.cat(labels_full, dim=0).cpu().numpy()

            weights_full = torch.cat(weights_full).cpu().numpy()
            ids_full = torch.cat(ids_full).cpu().numpy()
            if self.save_predictions:
                self.example_history = torch.cat(self.example_history, dim=0).cpu().numpy()
        return preds_full, labels_full, ids_full, weights_full


class StatMetricEvaluator(MetricEvaluator):
    def __init__(self, test_data, config):
        super().__init__(config)
        self.dataloader = test_data

    def predict(self, model, dataloader=None):
        dataloader = dataloader or self.dataloader
        assert dataloader, "Test dataloader not provided"

        preds_full = []
        labels_full = []
        weights_full = []
        ids_full = []

        for i, test_batch in enumerate(dataloader):
            labels = test_batch["endog"]
            ids = test_batch["id"].iloc[0]
            preds = np.array(model.predict(test_batch["exog"], i))
            labels_full.append(labels)
            weights_full.append(test_batch.get('weight', []))
            ids_full.append(ids)
            preds_full.append(preds)

        preds_full = np.stack(preds_full)
        labels_full = np.stack(labels_full)
        weights_full = np.stack(weights_full)
        ids_full = np.stack(ids_full)
        if len(preds_full.shape) == 2:
            preds_full = preds_full[:, :, np.newaxis]
        return preds_full, labels_full, ids_full, weights_full


class XGBMetricEvaluator(MetricEvaluator):
    def __init__(self, test_data, config):
        super().__init__(config)
        self.dataloader = test_data

    def predict(self, model, dataloader=None):
        dataloader = dataloader or self.dataloader
        assert dataloader, "Test dataloader not provided"
        out = []
        labels = []
        ids = []
        weights = []
        for i, (test_step, test_label) in enumerate(dataloader):
            labels.append(test_label.to_numpy())
            ids.append(test_step['_id_'].to_numpy())
            outt = model.predict(test_step, i)
            weights.append([])
            out.append(outt)
        outtemp = np.vstack(out).transpose()
        labels_temp = np.hstack(labels)
        ids_temp = np.vstack(ids).transpose()[:, 0]
        if len(outtemp.shape) == 2:
            outtemp = outtemp[:, :, np.newaxis]
        if len(labels_temp.shape) == 2:
            labels_temp = labels_temp[:, :, np.newaxis]
        if self.save_predictions:
            labels_ids = self.dataloader.data[['_id_', self.dataloader.target[0]]]
            for n, g in labels_ids.groupby("_id_"):
                labels_all = g[self.dataloader.target[0]].to_numpy().round(6)
                windows_labels = np.lib.stride_tricks.sliding_window_view(labels_all, self.dataloader.example_length)
                self.example_history.append(windows_labels.copy()[:, :self.dataloader.encoder_length])
            self.example_history = np.concatenate(self.example_history, axis=0)[:, :, np.newaxis]
        return outtemp, labels_temp, ids_temp, np.stack(weights)
