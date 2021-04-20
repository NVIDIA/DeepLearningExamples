from typing import Any, Dict, List, Optional

import numpy as np

from deployment_toolkit.core import BaseMetricsCalculator


class MetricsCalculator(BaseMetricsCalculator):
    def __init__(self, output_used_for_metrics: str = "classes"):
        self._output_used_for_metrics = output_used_for_metrics

    def calc(self, *, y_pred: Dict[str, np.ndarray], y_real: Optional[Dict[str, np.ndarray]], **_) -> Dict[str, float]:
        y_true = y_real[self._output_used_for_metrics]
        y_pred = y_pred[self._output_used_for_metrics]
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        assert y_true.shape == y_pred.shape
        return {"accuracy": (y_true == y_pred).mean()}
