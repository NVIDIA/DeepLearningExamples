import monai
import torch
import torch.nn as nn
from pytorch_lightning.metrics.functional import stat_scores
from pytorch_lightning.metrics.metric import Metric


class Dice(Metric):
    def __init__(self, nclass):
        super().__init__(dist_sync_on_step=True)
        self.add_state("dice", default=torch.zeros((nclass,)), dist_reduce_fx="mean")

    def update(self, pred, target):
        self.dice = self.compute_stats(pred, target)

    def compute(self):
        return self.dice

    @staticmethod
    def compute_stats(pred, target):
        num_classes = pred.shape[1]
        _bg = 1
        scores = torch.zeros(num_classes - _bg, device=pred.device, dtype=torch.float32)
        precision = torch.zeros(num_classes - _bg, device=pred.device, dtype=torch.float32)
        recall = torch.zeros(num_classes - _bg, device=pred.device, dtype=torch.float32)
        for i in range(_bg, num_classes):
            if not (target == i).any():
                # no foreground class
                _, _pred = torch.max(pred, 1)
                scores[i - _bg] += 1 if not (_pred == i).any() else 0
                recall[i - _bg] += 1 if not (_pred == i).any() else 0
                precision[i - _bg] += 1 if not (_pred == i).any() else 0
                continue
            _tp, _fp, _tn, _fn, _ = stat_scores(pred=pred, target=target, class_index=i)
            denom = (2 * _tp + _fp + _fn).to(torch.float)
            score_cls = (2 * _tp).to(torch.float) / denom if torch.is_nonzero(denom) else 0.0
            scores[i - _bg] += score_cls
        return scores


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, softmax=True, batch=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        cross_entropy = self.cross_entropy(y_pred, y_true[:, 0].long())
        return (dice + cross_entropy) / 2
