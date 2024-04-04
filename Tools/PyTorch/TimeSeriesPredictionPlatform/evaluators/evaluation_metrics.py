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

# SPDX-License-Identifier: Apache-2.0
import sys
import numpy as np

from abc import ABC, abstractmethod
from numba import jit, prange


class AbstractMetric(ABC):
    @staticmethod
    @abstractmethod
    def __call__(pred, label, weights):
        pass


class TemporalDistortionIndex(AbstractMetric):
    name = "TDI"

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _calculate_tdi(X, Y):
        """Calculate TDI scores
        
        Parameters
        ----------
        X: np.ndarray
            2D array with shape (B x seq_len) of predictions
        Y: np.ndarray
            2D array with shape (B x seq_len) of labels
        
        Returns
        -------
        np.ndarray
            tdi scores for each example
        """

        batch_size, n  = X.shape
        tdis = np.full(batch_size, 0, dtype=np.float64)
        for tidx in prange(batch_size):
            d = np.abs(X[tidx].reshape(-1, 1) - Y[tidx])
            dist_matrix = np.ones((n, 2), dtype=np.float64) * np.inf
            step_matrix = np.full((n, n), -1, dtype=np.int8)

            dist_matrix[:, 0] = np.cumsum(d[:, 0])
            step_matrix[0, 1:] = 1
            step_matrix[1:, 0] = 2
            pattern_cost = np.ones(3, dtype=np.float64)

            for j in range(1, n):
                dist_matrix[0, j%2] = dist_matrix[0, (j-1)%2] + d[0, j]
                for i in range(1, n):
                    # modulo operator is used to avoid copying memory 
                    # from column 1 to column 0 at the end of iteration (traid memops for ops)
                    #diagonal
                    pattern_cost[0] = dist_matrix[i-1, (j-1)%2] + d[i, j] * 2
                    #left
                    pattern_cost[1] = dist_matrix[i, (j-1)%2] + d[i, j]
                    #up
                    pattern_cost[2] = dist_matrix[i-1, j%2] + d[i, j]
                    
                    step = np.argmin(pattern_cost)
                    dist_matrix[i, j%2] = pattern_cost[step]
                    step_matrix[i, j] = step
            tdi = 0.0
            y = 0.0
            dx = 0
            dy = 0
            step = -1
            i = n-1
            j = n-1
            while i != 0 or j != 0:
                step = int(step_matrix[i, j])
                if i < 0 or j < 0:break
                dx = int((step == 0) + 1)
                dy = int((step == 2) - (step == 1))
                tdi += abs(y + float(dy) / 2) * float(dx)
                y = y + dy
                i -= int((dx + dy) / 2)
                j -= int((dx - dy) / 2)
            tdis[tidx] = tdi
        return tdis
                        

    @staticmethod
    def __call__(pred, label, weights):
        if weights.size:
            print('Weights are not supported for TDI metric', file=sys.stderr)
        normalizer = (pred.shape[1]-1)**2
        if not pred.flags['C_CONTIGUOUS']:
            pred = np.ascontiguousarray(pred)
        tdi = np.mean(TemporalDistortionIndex._calculate_tdi(pred, label)) / normalizer
        return tdi


class SMAPE(AbstractMetric):
    name = "SMAPE"

    @staticmethod
    def __call__(preds, labels, weights):
        if not weights.size:
            weights = None
        a = 2 * np.abs(preds - labels)
        b = np.abs(labels) + np.abs(preds)
        b[b == 0] = 1  # numerator for this values is also 0 anyway
        return 100 * np.average(a / b, weights=weights)


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
    "TDI": TemporalDistortionIndex
}
