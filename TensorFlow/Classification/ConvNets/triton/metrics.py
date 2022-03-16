from typing import Any, Dict, List, Optional

import numpy as np

from deployment_toolkit.core import BaseMetricsCalculator


class MetricsCalculator(BaseMetricsCalculator):
    def __init__(self):
        self._equals = []

    def update(
            self,
            *,
            ids: List[Any],
            y_pred: Dict[str, np.ndarray],
            x: Optional[Dict[str, np.ndarray]],
            y_real: Optional[Dict[str, np.ndarray]],
    ):
        classes_real = y_real["classes"]
        classes_pred = y_pred["classes"]
        classes_real = np.squeeze(classes_real)
        classes_pred = np.squeeze(classes_pred)

        assert classes_real.shape == classes_pred.shape, (
            f"classes_pred.shape={classes_pred.shape} != " f"classes_real.shape={classes_real.shape}"
        )
        self._equals.append(classes_real == classes_pred)

    @property
    def metrics(self) -> Dict[str, Any]:
        return {"accuracy": np.concatenate(self._equals, axis=0).mean()}