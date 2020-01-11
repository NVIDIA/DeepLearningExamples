# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""
Recall is counting the number of relevant recommended items in R and normalizes it 
by dividing by minimum of R and number of clicked items by user

Recall@R(u,ω) := Σ_{r=1}^{R} I[ω(r) ∈ I_u] / min(R,|I_u|)

https://arxiv.org/pdf/1802.05814.pdf, chapter 4.2
"""

import numpy as np
from scipy.sparse import csr_matrix


def recall(X_true: csr_matrix, X_top_k: np.array, R=100) -> np.array:
    """ Calculates recall@R for each users in X_true and X_top_k matrices

    Args:
        X_true: Matrix containing True values for user-item interactions
        X_top_k: Matrix containing indices picked by model
        R: Number of elements taken into consideration

    Returns:
        Numpy array containing calculated recall@R for each user
    """

    selected = np.take_along_axis(X_true, X_top_k[:, :R], axis=-1)
    hit = selected.sum(axis=-1)

    maxhit = np.minimum(X_true.getnnz(axis=1), R)

    return np.squeeze(np.asarray(hit)) / maxhit
