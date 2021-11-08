# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F


def MAE_wrapper(config):
    return nn.L1Loss()


def MSE_wrapper(config):
    return nn.MSELoss()


def quantile_wrapper(config):
    return QuantileLoss(config)


# assumed for anomaly detection task
def cross_entropy_wrapper(config):
    return nn.CrossEntropyLoss()


def huber_wrapper(config):
    return nn.SmoothL1Loss()


def GLL_wrapper(config):
    return GaussianLogLikelihood()


class QuantileLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quantiles = list(config.model.quantiles)

    def forward(self, predictions, targets):
        losses = []

        for i, q in enumerate(self.quantiles):
            diff = predictions[..., i] - targets[..., 0]
            loss = ((1 - q) * F.relu(diff) + q * F.relu(-diff)).mean()
            losses.append(loss)
        losses = torch.stack(losses)

        return losses


class WeightedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, weights):
        x = F.l1_loss(inputs, targets, reduction="none")
        x = x * weights  # broadcasted along 0th dimension
        x = x.mean()
        return x


class GaussianLogLikelihood(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # Inputs with shape  [BS, window, 2] (mean + std)
        # Targets with shape [BS, window, 1]

        zero_index = targets[..., 0:1] != 0
        mu = inputs[..., 0:1]
        sigma = inputs[..., 1:2]
        distribution = torch.distributions.normal.Normal(mu[zero_index], sigma[zero_index])
        likelihood = distribution.log_prob(targets[zero_index])
        likelihood = -likelihood.view(inputs.shape[0], inputs.shape[1])
        loss = likelihood.mean(0).mean()

        return loss
