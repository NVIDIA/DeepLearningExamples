# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import torch


def pitch_transform_custom(pitch, pitch_lens):
    """Apply a custom pitch transformation to predicted pitch values.

    This sample modification linearly increases the pitch throughout
    the utterance from 0.5 of predicted pitch to 1.5 of predicted pitch.
    In other words, it starts low and ends high.

    PARAMS
    ------
    pitch: torch.Tensor (bs, max_len)
        Predicted pitch values for each lexical unit, padded to max_len (in Hz).
    pitch_lens: torch.Tensor (bs, max_len)
        Number of lexical units in each utterance.

    RETURNS
    -------
    pitch: torch.Tensor
        Modified pitch (in Hz).
    """

    weights = torch.arange(pitch.size(1), dtype=torch.float32, device=pitch.device)

    # The weights increase linearly from 0.0 to 1.0 in every i-th row
    # in the range (0, pitch_lens[i])
    weights = weights.unsqueeze(0) / pitch_lens.unsqueeze(1)

    # Shift the range from (0.0, 1.0) to (0.5, 1.5)
    weights += 0.5

    return pitch * weights
