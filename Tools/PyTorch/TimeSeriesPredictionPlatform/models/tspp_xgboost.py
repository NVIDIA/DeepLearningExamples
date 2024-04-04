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

import cudf
import pandas as pd

import pynvml
import numpy as np
import xgboost as xgb
import os
import glob

import dask_cudf
from distributed_utils import create_client
#Deal with the pateince and log_interval.  Also objective, cluster
class TSPPXGBoost():
    def __init__(self, config):
        self.config = config
        self.models = []

    def fit(self, train, label, valid, valid_label, **kwargs):
        train = train.drop(['_id_', '_timestamp_'], axis=1, errors='ignore')
        valid = valid.drop(['_id_', '_timestamp_'], axis=1, errors='ignore')
        X = xgb.DeviceQuantileDMatrix(cudf.from_pandas(train), label=cudf.from_pandas(label))
        V = xgb.DMatrix(cudf.from_pandas(valid), label=cudf.from_pandas(valid_label))
        model = xgb.train(params=self.config,
                          dtrain=X,
                          num_boost_round=self.config.n_rounds,
                          evals=[(X, 'train'), (V, 'valid')],
                          early_stopping_rounds=kwargs.get('patience', 5),
                          verbose_eval=kwargs.get("log_interval", 25),
                          )
        self.models.append(model)

    def predict(self, test, i):
        test = test.drop(['_id_', '_timestamp_'], axis=1, errors='ignore')
        model = self.models[i]
        X = xgb.DMatrix(cudf.from_pandas(test))
        return model.predict(X)

    def save(self, path):
        os.makedirs(os.path.join(path, 'checkpoints'), exist_ok=True)
        for i in range(len(self.models)):
            model = self.models[i]
            model.save_model(os.path.join(path, f'checkpoints/xgb_{i+1}.model'))

    def load(self, path):
        self.models = []
        for i in range(self.config.example_length - self.config.encoder_length):
            p = os.path.join(path, f'checkpoints/xgb_{i+1}.model')
            model = xgb.Booster()
            model.load_model(p)
            self.models.append(model)

class TSPPDaskXGBoost():
    def __init__(self, config):
        self.config = config
        self.models = []
        self.client = create_client(config)
        self.npartitions = self.config.cluster.npartitions

    def fit(self, train, label, valid, valid_label, **kwargs):
        X = xgb.dask.DaskDeviceQuantileDMatrix(self.client,
                                               dask_cudf.from_cudf(cudf.from_pandas(train), npartitions=self.npartitions),
                                               label=dask_cudf.from_cudf(cudf.from_pandas(label), npartitions=self.npartitions))
        V = xgb.dask.DaskDMatrix(self.client,
                                 dask_cudf.from_cudf(cudf.from_pandas(valid), npartitions=self.npartitions),
                                 label=dask_cudf.from_cudf(cudf.from_pandas(valid_label), npartitions=self.npartitions))
        model = xgb.dask.train(client=self.client,
                          params=self.config,
                          dtrain=X,
                          num_boost_round=self.config.n_rounds,
                          evals=[(X, 'train'), (V, 'valid')],
                          early_stopping_rounds=kwargs.get('patience', 5),
                          verbose_eval=kwargs.get("log_interval", 25),
                          )
        self.models.append(model)
        self.client.restart()

    def predict(self, test, i):
        test = test.reset_index(drop=True)
        model = self.models[i]
        test = dask_cudf.from_cudf(cudf.from_pandas(test), npartitions=self.npartitions)
        test = xgb.dask.DaskDMatrix(self.client, test)
        out =  xgb.dask.predict(self.client, model, test)
        return out.compute()

    def save(self, path):
        os.makedirs(os.path.join(path, 'checkpoints'), exist_ok=True)
        for i in range(len(self.models)):
            model = self.models[i]
            model['booster'].save_model(os.path.join(path, f'checkpoints/xgb_{i+1}.model'))

    def load(self, path):
        self.models = []
        for i in range(self.config.example_length - self.config.encoder_length):
            p = os.path.join(path, f'checkpoints/xgb_{i+1}.model')
            model = {'booster': xgb.dask.Booster()}
            model['booster'].load_model(p)
            self.models.append(model)
