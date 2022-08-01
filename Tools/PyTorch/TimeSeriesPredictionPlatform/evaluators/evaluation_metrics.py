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

import sys
import numpy as np

from abc import ABC, abstractmethod


class AbstractMetric(ABC):
    @staticmethod
    @abstractmethod
    def __call__(pred, label, weights):
        pass

class SMAPE(AbstractMetric):
    name = "SMAPE"

    @staticmethod
    def __call__(preds, labels, weights):
        if not weights.size:
            weights = None
        return 100 * np.average(2 * np.abs(preds - labels) / (np.abs(labels) + np.abs(preds)), weights=weights)



def normalised_quantile_loss(y_pred, y, quantile, weights=None):                                       
    """Implementation of the q-Risk function from https://arxiv.org/pdf/1912.09363.pdf"""              
    prediction_underflow = y - y_pred                                                                  
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.0) + (1.0 - quantile) * np.maximum(
        -prediction_underflow, 0.0                                                                     
    )                                                                                                  
    if weights is not None and weights.size:                                                           
        weighted_errors = weighted_errors * weights                                                    
        y = y * weights                                                                                
                                                                                                       
    loss = weighted_errors.sum()                                                                       
    normaliser = abs(y).sum()                                                                          
    return 2 * loss / normaliser    

class P50_loss(AbstractMetric):
    name = "P50"
    selector = 1

    @staticmethod
    def __call__(labels, preds, weights):
        return normalised_quantile_loss(labels, preds, 0.5,weights)


class P90_loss(AbstractMetric):
    name = "P90"
    selector = 2

    @staticmethod
    def __call__(labels, preds, weights):
        return normalised_quantile_loss(labels, preds, 0.9,weights)

# Normalized Deviation
class ND(AbstractMetric):
    name = "ND"

    @staticmethod
    def __call__(preds, labels, weights):
        diff = np.abs(labels - preds)

        if not weights.size:
            return np.sum(diff) / np.sum(np.abs(labels))
        else:
            return np.sum(diff * weights) / np.sum(np.abs(labels) * weights)


class MAE(AbstractMetric):
    name = "MAE"

    @staticmethod
    def __call__(preds, labels, weights, return_individual=False):
        if not weights.size:
            weights = None
        if return_individual:
            return np.average(np.abs(preds - labels), weights=weights, axis=0)
        else:
            return np.average(np.abs(preds - labels), weights=weights)


class MSE(AbstractMetric):
    name = "MSE"

    @staticmethod
    def __call__(preds, labels, weights, return_individual=False):
        if not weights.size:
            weights = None
        if return_individual:
            return np.average((preds - labels)**2, weights=weights, axis=0)
        else:
            return np.average((preds - labels)**2, weights=weights)


class RMSE(AbstractMetric):
    name = "RMSE"

    @staticmethod
    def __call__(preds, labels, weights):

        if not weights.size:
            weights = None
        return np.sqrt(np.average((preds - labels)**2, weights=weights))


class R_Squared(AbstractMetric):
    name = "R_Squared"

    @staticmethod
    def __call__(preds, labels, weights, return_individual=False):
        if not weights.size:
            if return_individual:
                return r2_score(preds, labels, multioutput="raw_values")
            return r2_score(preds, labels)
        else:
            values = r2_score(preds, labels, multioutput="raw_values")
            if return_individual:
                return values * weights
            return np.sum(values * weights) / np.sum(weights)


class WMSMAPE(AbstractMetric):
    name = "WMSMAPE"

    @staticmethod
    def __call__(preds, labels, weights, return_individual=False):
        if weights.size:
            if return_individual:
                return 2 * weights * np.abs(preds - labels) / (np.maximum(labels, 1) + np.abs(preds))
            else:
                return (
                    100.0
                    / np.sum(weights)
                    * np.sum(2 * weights * np.abs(preds - labels) / (np.maximum(labels, 1) + np.abs(preds)))
                )
        if return_individual:
            return 2 * np.abs(preds - labels) / (np.maximum(labels, 1) + np.abs(preds))
        else:
            return 100.0 / len(labels) * np.sum(2 * np.abs(preds - labels) / (np.maximum(labels, 1) + np.abs(preds)))


METRICS = {
    "SMAPE": SMAPE,
    "WMSMAPE": WMSMAPE,
    "MSE": MSE,
    "MAE": MAE,
    "P50": P50_loss,
    "P90": P90_loss,
    "RMSE": RMSE,
    "R_Squared": R_Squared,
    "ND": ND,
}
