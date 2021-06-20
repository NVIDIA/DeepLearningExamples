from typing import Any, Dict, List, Optional

import numpy as np

from triton.deployment_toolkit.core import BaseMetricsCalculator


class MetricsCalculator(BaseMetricsCalculator):
    def calc(
        self,
        *,
        ids: List[Any],
        x: Optional[Dict[str, np.ndarray]],
        y_real: Optional[Dict[str, np.ndarray]],
        y_pred: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        y_pred = y_pred["OUTPUT__0"]
        y_true = y_real["OUTPUT__0"]
        
        n_examples = y_pred.shape[0]
        nclass = max(np.max(y_pred), np.max(y_true))
        dice = np.zeros((nclass,))
        for i in range(n_examples):
            for c in range(nclass):
                if not (y_true[i] == c).any():
                    # no foreground class
                    dice[c] += 1 if not (y_pred[i] == c).any() else 0
                    continue
                true_pos, false_neg, false_pos = self.get_stats(y_pred[i], y_true[i], c + 1)
                denom = 2 * true_pos + false_neg + false_pos
                dice[c] += 2 * true_pos / denom if denom != 0 else 0.0

        dice /= n_examples
        dice = np.mean(dice)
        return {"dice": dice}

    @staticmethod
    def get_stats(pred, targ, class_idx):
        true_pos = np.logical_and(pred == class_idx, targ == class_idx).sum()
        false_neg = np.logical_and(pred != class_idx, targ == class_idx).sum()
        false_pos = np.logical_and(pred == class_idx, targ != class_idx).sum()
        return true_pos, false_neg, false_pos
