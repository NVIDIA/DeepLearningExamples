# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
import torch
import torch.nn as nn


class TrivialModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.encoder_length = config.encoder_length
        self.example_length = config.example_length
        self.predict_steps = self.example_length - self.encoder_length
        self.output_dim = len(config.get("quantiles", [""]))

    
    def forward(self, batch):
        t = next(t for t in batch.values() if t is not None)
        bs = t.shape[0]
        return torch.ones([bs, self.example_length - self.encoder_length, self.output_dim]).to(device=t.device) + self.bias

    def predict(self, batch):
        targets = batch["target"].clone()
        prev_predictions = targets.roll(1, 1)
        return prev_predictions[:, -self.predict_steps :, :]

    def test_with_last(self, batch):
        bs = max([tensor.shape[0] if tensor is not None else 0 for tensor in batch.values()])
        values = (
            batch["target_masked"]
            .clone()[:, -1, :]
            .reshape((bs, 1, self.output_dim))
        )

        return torch.cat((self.example_length - self.encoder_length) * [values], dim=1)

    def test_with_previous_window(self, batch):
        targets = batch["target"].clone()
        prev_predictions = targets.roll(self.predict_steps, 1)
        return prev_predictions[:, -self.predict_steps :, :]
