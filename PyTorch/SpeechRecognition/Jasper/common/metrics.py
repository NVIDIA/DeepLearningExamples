# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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


def __levenshtein(a, b):
    """Calculates the Levenshtein distance between two sequences."""

    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def word_error_rate(hypotheses, references):
    """Computes average Word Error Rate (WER) between two text lists."""

    scores = 0
    words = 0
    len_diff = len(references) - len(hypotheses)
    if len_diff > 0:
        raise ValueError("Uneqal number of hypthoses and references: "
                         "{0} and {1}".format(len(hypotheses), len(references)))
    elif len_diff < 0:
        hypotheses = hypotheses[:len_diff]

    for h, r in zip(hypotheses, references):
        h_list = h.split()
        r_list = r.split()
        words += len(r_list)
        scores += __levenshtein(h_list, r_list)
    if words!=0:
        wer = 1.0*scores/words
    else:
        wer = float('inf')
    return wer, scores, words
