# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class TrivialModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device(config.device.get("name", "cpu"))
        self.bias = nn.Parameter(torch.ones(1))
        self.encoder_length = config.dataset.encoder_length
        self.example_length = config.dataset.example_length
        self.predict_steps = self.example_length - self.encoder_length
        self.output_dim = len(config.model.get("quantiles", [""]))

    def test_with_last(self, batch):
        bs = max([tensor.shape[0] if tensor is not None else 0 for tensor in batch.values()])
        values = (
            # TODO: this will become disfuntional after removing "targer_masked" from dataset. Seed comment in data_utils.py
            batch["target_masked"]
            .clone()[:, -1, :]
            .reshape((bs, 1, self.output_dim))
        )

        return torch.cat((self.example_length - self.encoder_length) * [values], dim=1)

    def forward(self, batch):
        bs = max([tensor.shape[0] if tensor is not None else 0 for tensor in batch.values()])
        return (
            torch.ones([bs, self.example_length - self.encoder_length, self.output_dim]).to(device=self.device) + self.bias
        )

    def test_with_previous(self, batch):
        targets = batch["target"].clone()
        prev_predictions = targets.roll(1, 1)
        return prev_predictions[:, -self.predict_steps :, :]

    def test_with_previous_window(self, batch):
        targets = batch["target"].clone()
        prev_predictions = targets.roll(self.predict_steps, 1)
        return prev_predictions[:, -self.predict_steps :, :]
