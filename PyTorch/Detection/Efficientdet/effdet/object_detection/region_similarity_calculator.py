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

"""Region Similarity Calculators for BoxLists.

Region Similarity Calculators compare a pairwise measure of similarity
between the boxes in two BoxLists.
"""
import torch
from .box_list import BoxList


def area(boxlist: BoxList):
    """Computes area of boxes.

    Args:
        boxlist: BoxList holding N boxes

    Returns:
        a tensor with shape [N] representing box areas.
    """
    y_min, x_min, y_max, x_max = boxlist.boxes().chunk(4, dim=1)
    out = (y_max - y_min).squeeze(1) * (x_max - x_min).squeeze(1)
    return out


def intersection(boxlist1: BoxList, boxlist2: BoxList):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxlist1: BoxList holding N boxes
        boxlist2: BoxList holding M boxes

    Returns:
        a tensor with shape [N, M] representing pairwise intersections
    """
    y_min1, x_min1, y_max1, x_max1 = boxlist1.boxes().chunk(4, dim=1)
    y_min2, x_min2, y_max2, x_max2 = boxlist2.boxes().chunk(4, dim=1)
    all_pairs_min_ymax = torch.min(y_max1, y_max2.t())
    all_pairs_max_ymin = torch.max(y_min1, y_min2.t())
    intersect_heights = torch.clamp(all_pairs_min_ymax - all_pairs_max_ymin, min=0)
    all_pairs_min_xmax = torch.min(x_max1, x_max2.t())
    all_pairs_max_xmin = torch.max(x_min1, x_min2.t())
    intersect_widths = torch.clamp(all_pairs_min_xmax - all_pairs_max_xmin, min=0)
    return intersect_heights * intersect_widths


def iou(boxlist1: BoxList, boxlist2: BoxList):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxlist1: BoxList holding N boxes
        boxlist2: BoxList holding M boxes

    Returns:
        a tensor with shape [N, M] representing pairwise iou scores.
    """
    intersections = intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = areas1.unsqueeze(1) + areas2.unsqueeze(0) - intersections
    return torch.where(intersections == 0.0, torch.zeros_like(intersections), intersections / unions)


@torch.jit.script
class IouSimilarity(object):
    """Class to compute similarity based on Intersection over Union (IOU) metric.

    This class computes pairwise similarity between two BoxLists based on IOU.
    """
    def __init__(self):
        pass

    def compare(self, boxlist1: BoxList, boxlist2: BoxList):
        """Computes matrix of pairwise similarity between BoxLists.

        This op (to be overridden) computes a measure of pairwise similarity between
        the boxes in the given BoxLists. Higher values indicate more similarity.

        Note that this method simply measures similarity and does not explicitly
        perform a matching.

        Args:
            boxlist1: BoxList holding N boxes.
            boxlist2: BoxList holding M boxes.

        Returns:
            a (float32) tensor of shape [N, M] with pairwise similarity score.
        """
        return iou(boxlist1, boxlist2)
