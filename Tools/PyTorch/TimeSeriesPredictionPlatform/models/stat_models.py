# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

from abc import ABC
import os
import pmdarima as pm
# import cuml
import numpy as np
from cuml.tsa.auto_arima import AutoARIMA as cuMLAutoArima
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
        self.models = []

    def fit(self, label, data):
        self.model = pm.auto_arima(label, X=data)
        self.models.append(self.model)

    def predict(self, data, i):
        model = self.models[i]
        return model.predict(self.horizon, X=data)
    
    def save(self):
        with open('arima.pkl', 'wb') as f:
            pkl.dump(self.models, f)
    
    def load(self, path):
        with open(os.path.join(path, 'arima.pkl'), 'rb') as f:
            self.models = pkl.load(f)

class CUMLAutoARIMA(StatModel):
    def __init__(self, config):
        super().__init__(config)
        self.models = []

    def fit(self, label, data):
        self.model = cuMLAutoArima(label.astype(np.float64))
        self.model.search()
        self.model.fit()
        self.models.append(self.model)

    def predict(self, data, i):
        model = self.models[i]
        return model.forecast(self.horizon).get()
    
    def save(self):
        with open('arima.pkl', 'wb') as f:
            pkl.dump(self.models, f)
    
    def load(self, path):
        with open(os.path.join(path, 'arima.pkl'), 'rb') as f:
            self.models = pkl.load(f)
