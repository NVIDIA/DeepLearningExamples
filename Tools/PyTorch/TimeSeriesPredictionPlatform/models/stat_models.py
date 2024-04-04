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

# SPDX-License-Identifier: Apache-2.0
from abc import ABC
import os
import pmdarima as pm
import numpy as np
import pickle as pkl

class StatModel(ABC):
    def __init__(self, config):
        self.horizon = config.example_length - config.encoder_length
        self.config = config

    def fit(self, label, data):
        return

    def predict(self, data, i):
        return
    
    def save(self):
        return

    def load(self, path):
        return

class AutoARIMA(StatModel):
    def __init__(self, config):
        super().__init__(config)
        self.models = {}

    def fit(self, example):
        id, label, data = example['id'], example['endog'], example['exog']
        data = data if data.shape[-1] != 0 else None
        model = pm.auto_arima(label, X=data, **self.config)
        self.model = model

    def predict(self, example):
        model = self.model
        if len(example['endog_update']) != 0:
            model.update(example['endog_update'], X=data if (data := example['exog_update']).shape[-1] != 0 else None)
        # Issue is related to https://github.com/alkaline-ml/pmdarima/issues/492
        try:
            preds = model.predict(self.horizon, X=data if (data := example['exog']).shape[-1] != 0 else None)
        except ValueError as e:
            if "Input contains NaN, infinity or a value too large for dtype('float64')." in str(e):
                print(str(e))
                preds = np.empty(self.horizon)
                preds.fill(self.model.arima_res_.data.endog[-1])
            else:
                raise
        return preds
    
    def save(self):
        with open('arima.pkl', 'wb') as f:
            pkl.dump(self.models, f)
    
    def load(self, path):
        with open(os.path.join(path, 'arima.pkl'), 'rb') as f:
            self.models = pkl.load(f)
