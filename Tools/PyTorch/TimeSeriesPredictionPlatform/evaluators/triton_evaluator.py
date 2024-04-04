# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
import pickle
import numpy as np
import time
import logging
from tqdm import tqdm
from .evaluation_metrics import METRICS
from .evaluator import MetricEvaluator
from triton.run_inference_on_triton import AsyncGRPCTritonRunner

import tritonclient.http as triton_http
import tritonclient.grpc as triton_grpc
import xgboost as xgb
import hydra

class TritonEvaluator(MetricEvaluator):
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



    def predict(self, dataloader, model_name, server_url="localhost:8001"):
        LOGGER = logging.getLogger("run_inference_on_triton")

        runner = AsyncGRPCTritonRunner(
                server_url,
                model_name,
                "1",
                dataloader=dataloader(),
                verbose=False,
                resp_wait_s=120,
                max_unresponded_reqs=128,
            )
        start = time.time()
        preds_full = []
        labels_full = []
        weights_full = []
        ids_full = []
        for ids, x, y_pred, y_real in tqdm(runner, unit="batch", mininterval=10):
            if self.save_predictions:
                self.example_history.append(x['target__6'][:,:self.config.encoder_length])
            ids_full.append(ids)
            preds_full.append(y_pred['target__0'])
            labels_full.append(y_real['target__0'][:,:,0][:,:,np.newaxis])
            weights_full.append(x['weight__9'])
        stop = time.time()
        preds_full = np.concatenate(preds_full, axis=0)
        labels_full = np.concatenate(labels_full, axis=0)
        weights_full = np.concatenate(weights_full, axis=0)
        if np.isnan(weights_full).any():
            weights_full = np.empty([0])
        ids_full = np.concatenate(ids_full, axis=0)
        LOGGER.info(f"\nThe inference took {stop - start:0.3f}s")
        if self.save_predictions:
            self.example_history = np.concatenate(self.example_history, axis=0)

        predictions_dict = {
            'preds_full': preds_full,
            'labels_full': labels_full,
            'ids_full': ids_full,
            'weights_full': weights_full
        }

        return predictions_dict

    def predict_xgboost(self, dataloader, max_batch_size, server_url="localhost:8001"):
        grpc_client = triton_grpc.InferenceServerClient(
            url=server_url,
            verbose = False
        )
        out = []
        labels = []
        ids = []
        weights = []
        for i, (test_step, test_label) in enumerate(dataloader):
            labels.append(test_label.to_numpy())
            ids.append(test_step['_id_'].to_numpy())
            data = test_step.to_numpy().astype('float32')
            weights.append([])
            test_len = len(data)
            num_iters = int(test_len/max_batch_size) + 1
            temp_out = []
            for j in range(num_iters):
                sliced_data = data[j*max_batch_size:(j+1)*max_batch_size]
                dims = sliced_data.shape
                triton_input_grpc = triton_grpc.InferInput(
                    'input__0',
                    dims,
                    'FP32'
                )
                triton_input_grpc.set_data_from_numpy(sliced_data)
                triton_output_grpc = triton_grpc.InferRequestedOutput('output__0')
                request_grpc = grpc_client.infer(
                    f'xgb_{i+1}',
                    model_version='1',
                    inputs=[triton_input_grpc],
                    outputs=[triton_output_grpc]
                )
                outt = request_grpc.as_numpy('output__0')
                temp_out = np.hstack((temp_out, outt))
            out.append(temp_out)
            weights.append([])
        outtemp = np.vstack(out).transpose()
        labels_temp = np.hstack(labels)
        ids_temp = np.vstack(ids).transpose()
        if len(outtemp.shape) == 2:
            outtemp = outtemp[:,:,np.newaxis]
        if len(labels_temp.shape) == 2:
            labels_temp = labels_temp[:, :, np.newaxis]
        if self.save_predictions:
            labels_ids = dataloader.data[['_id_', dataloader.target[0]]]
            for n, g in labels_ids.groupby("_id_"):
                labels_all = g[dataloader.target[0]].to_numpy().round(6)
                windows_labels = np.lib.stride_tricks.sliding_window_view(labels_all, dataloader.example_length)
                self.example_history.append(windows_labels.copy()[:, :dataloader.encoder_length])
            self.example_history = np.concatenate(self.example_history, axis=0)[:, :, np.newaxis]

        predictions_dict = {
            'preds_full': outtemp,
            'labels_full': labels_temp,
            'ids_full': ids_temp[:,0],
            'weights_full': np.stack(weights)
        }

        return predictions_dict
