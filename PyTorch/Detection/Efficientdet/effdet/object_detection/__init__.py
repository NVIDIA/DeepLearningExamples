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
# Object detection data loaders and libraries are mostly based on RetinaNet:
# https://github.com/tensorflow/tpu/tree/master/models/official/retinanet
from .argmax_matcher import ArgMaxMatcher
from .box_coder import FasterRcnnBoxCoder
from .box_list import BoxList
from .matcher import Match
from .region_similarity_calculator import IouSimilarity
from .target_assigner import TargetAssigner
