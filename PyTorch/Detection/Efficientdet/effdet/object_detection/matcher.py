# Copyright 2020 Google Research. All Rights Reserved.
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
# ==============================================================================

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

"""Matcher interface and Match class.

This module defines the Matcher interface and the Match object. The job of the
matcher is to match row and column indices based on the similarity matrix and
other optional parameters. Each column is matched to at most one row. There
are three possibilities for the matching:

1) match: A column matches a row.
2) no_match: A column does not match any row.
3) ignore: A column that is neither 'match' nor no_match.

The ignore case is regularly encountered in object detection: when an anchor has
a relatively small overlap with a ground-truth box, one neither wants to
consider this box a positive example (match) nor a negative example (no match).

The Match class is used to store the match results and it provides simple apis
to query the results.
"""
import torch


@torch.jit.script
class Match(object):
    """Class to store results from the matcher.

    This class is used to store the results from the matcher. It provides
    convenient methods to query the matching results.
    """

    def __init__(self, match_results: torch.Tensor):
        """Constructs a Match object.

        Args:
            match_results: Integer tensor of shape [N] with (1) match_results[i]>=0,
                meaning that column i is matched with row match_results[i].
                (2) match_results[i]=-1, meaning that column i is not matched.
                (3) match_results[i]=-2, meaning that column i is ignored.

        Raises:
            ValueError: if match_results does not have rank 1 or is not an integer int32 scalar tensor
        """
        if len(match_results.shape) != 1:
            raise ValueError('match_results should have rank 1')
        if match_results.dtype not in (torch.int32, torch.int64):
            raise ValueError('match_results should be an int32 or int64 scalar tensor')
        self.match_results = match_results

    def matched_column_indices(self):
        """Returns column indices that match to some row.

        The indices returned by this op are always sorted in increasing order.

        Returns:
            column_indices: int32 tensor of shape [K] with column indices.
        """
        return torch.nonzero(self.match_results > -1).flatten().long()

    def matched_column_indicator(self):
        """Returns column indices that are matched.

        Returns:
            column_indices: int32 tensor of shape [K] with column indices.
        """
        return self.match_results >= 0

    def num_matched_columns(self):
        """Returns number (int32 scalar tensor) of matched columns."""
        return self.matched_column_indices().numel()

    def unmatched_column_indices(self):
        """Returns column indices that do not match any row.

        The indices returned by this op are always sorted in increasing order.

        Returns:
          column_indices: int32 tensor of shape [K] with column indices.
        """
        return torch.nonzero(self.match_results == -1).flatten().long()

    def unmatched_column_indicator(self):
        """Returns column indices that are unmatched.

        Returns:
          column_indices: int32 tensor of shape [K] with column indices.
        """
        return self.match_results == -1

    def num_unmatched_columns(self):
        """Returns number (int32 scalar tensor) of unmatched columns."""
        return self.unmatched_column_indices().numel()

    def ignored_column_indices(self):
        """Returns column indices that are ignored (neither Matched nor Unmatched).

        The indices returned by this op are always sorted in increasing order.

        Returns:
          column_indices: int32 tensor of shape [K] with column indices.
        """
        return torch.nonzero(self.ignored_column_indicator()).flatten().long()

    def ignored_column_indicator(self):
        """Returns boolean column indicator where True means the column is ignored.

        Returns:
            column_indicator: boolean vector which is True for all ignored column indices.
        """
        return self.match_results == -2

    def num_ignored_columns(self):
        """Returns number (int32 scalar tensor) of matched columns."""
        return self.ignored_column_indices().numel()

    def unmatched_or_ignored_column_indices(self):
        """Returns column indices that are unmatched or ignored.

        The indices returned by this op are always sorted in increasing order.

        Returns:
            column_indices: int32 tensor of shape [K] with column indices.
        """
        return torch.nonzero(0 > self.match_results).flatten().long()

    def matched_row_indices(self):
        """Returns row indices that match some column.

        The indices returned by this op are ordered so as to be in correspondence with the output of
        matched_column_indicator().  For example if self.matched_column_indicator() is [0,2],
        and self.matched_row_indices() is [7, 3], then we know that column 0 was matched to row 7 and
        column 2 was matched to row 3.

        Returns:
            row_indices: int32 tensor of shape [K] with row indices.
        """
        return torch.gather(self.match_results, 0, self.matched_column_indices()).flatten().long()

    def gather_based_on_match(self, input_tensor, unmatched_value, ignored_value):
        """Gathers elements from `input_tensor` based on match results.

        For columns that are matched to a row, gathered_tensor[col] is set to input_tensor[match_results[col]].
        For columns that are unmatched, gathered_tensor[col] is set to unmatched_value. Finally, for columns that
        are ignored gathered_tensor[col] is set to ignored_value.

        Note that the input_tensor.shape[1:] must match with unmatched_value.shape
        and ignored_value.shape

        Args:
            input_tensor: Tensor to gather values from.
            unmatched_value: Constant tensor value for unmatched columns.
            ignored_value: Constant tensor value for ignored columns.

        Returns:
            gathered_tensor: A tensor containing values gathered from input_tensor.
                The shape of the gathered tensor is [match_results.shape[0]] + input_tensor.shape[1:].
        """
        ss = torch.stack([ignored_value, unmatched_value])
        input_tensor = torch.cat([ss, input_tensor], dim=0)
        gather_indices = torch.clamp(self.match_results + 2, min=0)
        gathered_tensor = torch.index_select(input_tensor, 0, gather_indices)
        return gathered_tensor
