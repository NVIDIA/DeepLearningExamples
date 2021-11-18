# SPDX-License-Identifier: Apache-2.0
from abc import ABC

import pmdarima as pm


class StatModel(ABC):
    def __init__(self, config):
        self.horizon = config.dataset.example_length - config.dataset.encoder_length
        self.config = config

    def fit(self, endog, exog):
        return

    def predict(self, exog):
        return


class AutoARIMA(StatModel):
    def __init__(self, config):
        super().__init__(config)

    def fit(self, endog, exog):
        self.model = pm.auto_arima(endog, X=exog)

    def predict(self, exog):
        return self.model.predict(self.horizon, X=exog)
