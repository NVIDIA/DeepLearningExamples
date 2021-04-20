from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
from deployment_toolkit.core import BaseMetricsCalculator

class MetricsCalculator(BaseMetricsCalculator):
    def __init__(self):
        pass

    def calc(
            self,
            *,
            ids: List[Any],
            y_pred: Dict[str, np.ndarray],
            x: Optional[Dict[str, np.ndarray]],
            y_real: Optional[Dict[str, np.ndarray]],
    ) -> Dict[str, float]:
        categories = np.argmax(y_pred["OUTPUT__0"], axis=-1)
        print(categories.shape)
        print(categories[:128], y_pred["OUTPUT__0"] )
        print(y_real["OUTPUT__0"][:128])

        return {
            "accuracy": np.mean(np.argmax(y_pred["OUTPUT__0"], axis=-1) == 
                                np.argmax(y_real["OUTPUT__0"], axis=-1))
        }
