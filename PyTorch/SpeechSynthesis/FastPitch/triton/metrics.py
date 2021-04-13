# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
from deployment_toolkit.core import BaseMetricsCalculator

class MetricsCalculator(BaseMetricsCalculator):
    def __init__(self, output_used_for_metrics: str):
        self._output_used_for_metrics = output_used_for_metrics
        self._MEL_MIN = -15.0
        self._MEL_MAX = 3.0

    def calc(
            self,
            *,
            ids: List[Any],
            y_pred: Dict[str, np.ndarray],
            x: Optional[Dict[str, np.ndarray]],
            y_real: Optional[Dict[str, np.ndarray]],
    ) -> Dict[str, float]:

        y_pred = y_pred[self._output_used_for_metrics]
        value_range_correct = np.ones(y_pred.shape[0]).astype(np.int32)
        for idx, mel in enumerate(y_pred):
            mel = mel[~np.isnan(mel)]
            if mel.min() < self._MEL_MIN or mel.max() > self._MEL_MAX:
                value_range_correct[idx] = 0
        return {
            "accuracy": np.mean(value_range_correct)
        }

    # from LJSpeech:
    # min(mins)    # Out[27]: -11.512925148010254
    # max(maxs)    # Out[28]: 2.0584452152252197
    # min(sizes)   # Out[29]: 96
    # max(sizes)   # Out[30]: 870
