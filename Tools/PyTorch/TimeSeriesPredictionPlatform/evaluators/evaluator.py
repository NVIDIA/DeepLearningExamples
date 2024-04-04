# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

import sys
import pickle
from abc import ABC
import warnings

import dgl
import numpy as np
import pandas as pd
import torch
from data.datasets import get_collate_fn
from models.interpretability import InterpretableModelBase
from torch.utils.data import DataLoader
from training.utils import to_device
from distributed_utils import get_mp_context
from data.data_utils import DTYPE_MAP

from .evaluation_metrics import METRICS

from typing import List, Callable


class Postprocessor:
    """
    PoC class used for simple transformations like rounding or clipping
    """
    def __init__(self, transformations: List[Callable[[torch.Tensor], torch.Tensor]]):
        self._transformations = transformations

    def __call__(self, x):
        return self.transform(x)

    def transform(self, x):
        for t in self._transformations:
            x = t(x)
        return x


class MetricEvaluator(ABC):
    def __init__(self, config, postprocessor=None, scaler="scalers"):
        self.output_selector = config.get("output_selector", None)
        self.per_step_metrics = config.get("per_step_metrics", False)
        self.save_predictions = config.get("save_predictions", False)
        self.metrics = []
        preprocessor_state = pickle.load(open(config.preprocessor_state_path, "rb"))
        self.postprocessor = postprocessor
        self.scalers = preprocessor_state[scaler]
        self.time_embed_dtype = preprocessor_state.get("timestamp_embed_type")
        self.example_history = []

        for name in config.metrics:
            if name not in METRICS:
                raise ValueError(f"No metric of name: {name}")
            self.metrics.append(METRICS[name]())
        self.config = config

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def save_preds(self, preds, ids, timestamps):
        all_examples = self.example_history  # make this a separate function in each eval
        if len(all_examples.shape) == 4:  # MultiID case
            all_examples = all_examples.transpose(2,0,1,3).reshape(-1, all_examples.shape[1])
        elif len(all_examples.shape) == 3: 
            all_examples = all_examples.transpose(2, 0, 1).reshape(-1, all_examples.shape[1])

        if len(preds.shape) == 4:
            tgt_ords = np.arange(preds.shape[2]).repeat(preds.shape[0])
            tgt_ords = pd.DataFrame(tgt_ords, columns=['#target'])
            preds = preds.transpose(2, 0, 1, 3).reshape(-1, preds.shape[1], preds.shape[3])
            ids = ids.transpose().reshape(-1)
        else:
            tgt_ords = None

        all_examples = self.scalers.inverse_transform_targets(all_examples, ids)

        hist_df = pd.DataFrame(all_examples, columns=[f't{i + 1}' for i in range(-self.config.encoder_length, 0)])
        ids = pd.DataFrame(ids, columns=['id'])
        timestamps = pd.DataFrame(timestamps, columns=['timestamp'])
        col_labels = [f'Estimator{j}_t{i:+}' for j in range(preds.shape[2]) for i in range(preds.shape[1])]
        preds_df = pd.DataFrame(preds.reshape(preds.shape[0], -1, order='F'), columns=col_labels)
        df = pd.concat([ids, timestamps, tgt_ords, hist_df, preds_df], axis=1)
        df.to_csv('predictions.csv')

    def transpose_preds(self, preds, labels, ids, weights, timestamps):
        """
        This fuction reshapes all legal shapes into num_examples x time x num_estimators
        """
        if labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        if weights.size and weights.shape[-1] == 1:
            weights = weights.squeeze(-1)

        # preds:  BS x T x ID x F x H
        # labels: BS x T x ID x F
        # ids: BS x ID
        if len(preds.shape) == 5:
            assert ids is not None
            ids = ids.transpose(1,0).flatten().repeat(preds.shape[3])
            preds = preds.transpose(2,3,0,1,4)
            labels = labels.transpose(2,3,0,1)
            if weights.size:
                weights = weights.transpose(2,3,0,1)

        # preds:  BS x T x ID x H or BS x T x F x H, it should be processed in the same way
        # labels: BS x T x ID
        # ids: BS x ID
        elif len(preds.shape) == 4:
            ids = ids.transpose(1,0).flatten() if ids is not None else None
            timestamps = timestamps.transpose(1,0).flatten() if timestamps is not None else None
            preds = preds.transpose(2,0,1,3)
            labels = labels.transpose(2,0,1)
            if weights.size:
                weights = weights.transpose(2,0,1)

        elif len(preds.shape) != 3:
            raise ValueError("Predictions are expected to have 3, 4 or 5 dimensions")

        if len(preds.shape) > 3:
            preds = preds.reshape(-1, *preds.shape[2:])
            labels = labels.reshape(-1, labels.shape[-1])
            if weights.size:
                weights = weights.reshape(-1, *weights.shape[2:])

        return preds, labels, ids, weights, timestamps

    def evaluate(self, preds, labels, ids, weights, timestamps, unscale=True):

        if unscale:
            print('Deprecation warning: Target unscaling will be moved from the evaluate function to the predict function', file=sys.stderr)
            preds = self.scalers.inverse_transform_targets(preds, ids)
            labels = self.scalers.inverse_transform_targets(labels, ids)

        upreds, labels, ids, weights, timestamps = self.transpose_preds(preds, labels, ids, weights, timestamps)
        
        if self.save_predictions:
            self.save_preds(upreds, ids, timestamps)

        results = {}
        for metric in self.metrics:
            selector = getattr(metric, 'selector', self.output_selector)
            preds = upreds[..., selector]
            m = metric(preds, labels, weights) if np.all(np.isfinite(preds)) else np.NaN
            results[metric.name] = float(m)
            if self.per_step_metrics:
                if metric.name == 'TDI': # The only metric that requires whole time series to be computed
                    continue
                results[metric.name + '@step'] = []
                for i in range(preds.shape[-1]):
                    m = metric(preds[..., i:i+1], labels[..., i:i+1], weights[..., i:i+1]) if np.all(np.isfinite(preds[..., i:i+1])) else np.NaN
                    results[metric.name + '@step'].append(float(m))
        return results


class CTLMetricEvaluator(MetricEvaluator):
    def __init__(self, test_data, config, postprocessor=None):
        super().__init__(config, postprocessor)
        self.device = config.device
        self.visualisation_indices = config.get("visualisation_indices", None)

        mp_context = get_mp_context() if config.num_workers else None
        if test_data is not None:
            self.dataloader = DataLoader(
                test_data,
                batch_size=self.config.batch_size,
                num_workers=config.num_workers,
                pin_memory=True,
                collate_fn=get_collate_fn(config.model_type, config.encoder_length, test=True),
                multiprocessing_context=mp_context
            )
        else:
            self.dataloader = None

    def prep_data(self, batch):
        ids = batch.ndata['id'] if isinstance(batch, dgl.DGLGraph) else batch["id"]
        ids = ids[:, 0, ...]  # Shape BS x T x F [x H]

        timestamp = batch.ndata['timestamp'] if isinstance(batch, dgl.DGLGraph) else batch["timestamp"]
        timestamp = timestamp[:, 0, ...]

        weights = batch.ndata['weight'] if isinstance(batch, dgl.DGLGraph) else batch['weight']
        weights = weights[:, self.config.encoder_length:,
                  :] if weights is not None and weights.numel() else torch.empty(0)

        batch = to_device(batch, device=self.device)

        return batch, weights, ids, timestamp

    def predict(self, model, dataloader=None):
        self.device = next(model.parameters()).device

        if not dataloader:
            dataloader = self.dataloader
        assert dataloader is not None, "Dataloader cannot be None, either pass in a valid dataloader or \
        initialize evaluator with valid test_data"

        if self.visualisation_indices is not None:
            assert isinstance(model, InterpretableModelBase), "Visualisation is only possible for interpretable models"
            model.enable_activations_dump()

        test_method_name = 'predict' if hasattr(model, "predict") else '__call__'
        test_method = getattr(model, test_method_name)

        model.eval()

        with torch.no_grad():

            preds_full = []
            labels_full = []
            weights_full = []
            ids_full = []
            timestamps_full = []
            figures_full = []

            for i, (batch, labels, _) in enumerate(dataloader):
                if self.save_predictions:
                    batch_data = batch.ndata if isinstance(batch, dgl.DGLGraph) else batch
                    self.example_history.append(batch_data['target'][:, :self.config.encoder_length].detach().cpu())
                batch, weights, ids, timestamp = self.prep_data(batch)

                labels_full.append(labels)
                weights_full.append(weights)
                preds = test_method(batch)
                ids_full.append(ids)
                preds_full.append(preds)
                timestamps_full.append(timestamp)

                if self.visualisation_indices is not None:
                    current_indices = [sample_number for sample_number in self.visualisation_indices if
                                       i * self.config.batch_size <= sample_number < (i + 1) * self.config.batch_size]
                    for sample_number in current_indices:
                        activations = model.get_activations(sample_number % self.config.batch_size,
                                                            dataloader.dataset.features)
                        for name, fig in activations.items():
                            figures_full.append((fig, name, sample_number))

            preds_full = torch.cat(preds_full, dim=0).cpu().numpy()
            labels_full = torch.cat(labels_full, dim=0).cpu().numpy()

            weights_full = torch.cat(weights_full).cpu().numpy()
            ids_full = torch.cat(ids_full).cpu().numpy()

            timestamps_full = torch.cat(timestamps_full).cpu().numpy()

            if self.save_predictions:
                self.example_history = torch.cat(self.example_history, dim=0).cpu().numpy()

        preds_full = self.scalers.inverse_transform_targets(preds_full, ids_full)
        if self.postprocessor is not None:
            preds_full = self.postprocessor(preds_full)
        labels_full = self.scalers.inverse_transform_targets(labels_full, ids_full)

        timestamps_full = timestamps_full.astype(DTYPE_MAP[self.time_embed_dtype])

        predictions_dict = {
            'preds_full': preds_full,
            'labels_full': labels_full,
            'ids_full': ids_full,
            'weights_full': weights_full,
            'timestamps_full': timestamps_full
        }

        if figures_full:
            predictions_dict['figures_full'] = figures_full

        return predictions_dict

    def evaluate(self, preds, labels, ids, weights, timestamps):
        # This function is part of a rework aimed to move unscaling to the predict function
        # It should be removed in the future
        return super().evaluate(preds, labels, ids, weights, timestamps, unscale=False)


class StatMetricEvaluator(MetricEvaluator):
    def __init__(self, test_data, config, postprocessor=None):
        super().__init__(config, postprocessor, scaler="alt_scalers")
        self.dataloader = test_data
        self.dataloader.test = True


    def predict(self, model, dataloader=None):
        dataloader = dataloader or self.dataloader
        assert dataloader, "Test dataloader not provided"

        preds_full = []
        labels_full = []
        weights_full = []
        ids_full = []

        for test_example in dataloader:
            labels = test_example["endog"]
            id = test_example['id']
            preds = np.array(model.predict(test_example))
            labels_full.append(labels)
            weights_full.append(
                weights 
                if (weights := test_example['weight']).shape[-1] else []
            )
            ids_full.append(id)
            preds_full.append(preds)

        preds_full = np.stack(preds_full)
        labels_full = np.stack(labels_full)
        weights_full = np.stack(weights_full)
        ids_full = np.stack(ids_full)
        if len(preds_full.shape) == 2:
            preds_full = preds_full[:, :, np.newaxis]
        
        preds_full = self.scalers.inverse_transform_targets(preds_full, ids_full)
        if self.postprocessor is not None:
            preds_full = self.postprocessor(preds_full)
        labels_full = self.scalers.inverse_transform_targets(labels_full, ids_full)
        
        
        predictions_dict = {
            'preds_full': preds_full,
            'labels_full': labels_full,
            'ids_full': ids_full,
            'weights_full': weights_full
        }

        return predictions_dict
    
    def evaluate(self, preds, labels, ids, weights, timestamps):
        return super().evaluate(preds, labels, ids, weights, timestamps, unscale=False)
    
    def save_preds(self, preds, ids, timestamps):
        ids = pd.DataFrame(ids, columns=['id'])
        timestamps = pd.DataFrame(timestamps, columns=['timestamp'])
        col_labels = [f'Estimator{j}_t{i:+}' for j in range(preds.shape[2]) for i in range(preds.shape[1])]
        preds_df = pd.DataFrame(preds.reshape(preds.shape[0], -1, order='F'), columns=col_labels)
        df = pd.concat([ids, timestamps, preds_df], axis=1)
        df.to_csv('predictions.csv')


class XGBMetricEvaluator(MetricEvaluator):
    def __init__(self, test_data, config , postprocessor=None):
        super().__init__(config, postprocessor)
        self.dataloader = test_data
        self.dataloader.test = True

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
        
        preds_full = self.scalers.inverse_transform_targets(outtemp, ids_temp)
        if self.postprocessor is not None:
            preds_full = self.postprocessor(preds_full)
        labels_full = self.scalers.inverse_transform_targets(labels_temp, ids_temp)
    
        predictions_dict = {
            'preds_full': preds_full,
            'labels_full': labels_full,
            'ids_full': ids_temp,
            'weights_full': np.stack(weights)
        }

        return predictions_dict
    
    def evaluate(self, preds, labels, ids, weights, timestamps):
        return super().evaluate(preds, labels, ids, weights, timestamps, unscale=False)


def unpack_predictions(predictions_dict):
    preds = predictions_dict.get('preds_full', None)
    labels = predictions_dict.get('labels_full', None)
    ids = predictions_dict.get('ids_full', None)
    weights = predictions_dict.get('weights_full', None)
    timestamps = predictions_dict.get('timestamps_full', None)
    figures = predictions_dict.get('figures_full', None)

    return preds, labels, ids, weights, timestamps, figures
