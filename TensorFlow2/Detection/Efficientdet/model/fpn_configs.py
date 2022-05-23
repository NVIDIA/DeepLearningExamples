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
"""BiFPN/QuFPN and other FPN configs.

BiFPN is presented in the EfficientDet paper.
QuFPN is proposed in https://github.com/google/automl/pull/580
"""
import itertools
from utils import hparams_config


def bifpn_config(min_level, max_level, weight_method):
  """A dynamic bifpn config that can adapt to different min/max levels."""
  p = hparams_config.Config()
  p.weight_method = weight_method or 'fastattn'

  # Node id starts from the input features and monotonically increase whenever
  # a new node is added. Here is an example for level P3 - P7:
  #     P7 (4)              P7" (12)
  #     P6 (3)    P6' (5)   P6" (11)
  #     P5 (2)    P5' (6)   P5" (10)
  #     P4 (1)    P4' (7)   P4" (9)
  #     P3 (0)              P3" (8)
  # So output would be like:
  # [
  #   {'feat_level': 6, 'inputs_offsets': [3, 4]},  # for P6'
  #   {'feat_level': 5, 'inputs_offsets': [2, 5]},  # for P5'
  #   {'feat_level': 4, 'inputs_offsets': [1, 6]},  # for P4'
  #   {'feat_level': 3, 'inputs_offsets': [0, 7]},  # for P3"
  #   {'feat_level': 4, 'inputs_offsets': [1, 7, 8]},  # for P4"
  #   {'feat_level': 5, 'inputs_offsets': [2, 6, 9]},  # for P5"
  #   {'feat_level': 6, 'inputs_offsets': [3, 5, 10]},  # for P6"
  #   {'feat_level': 7, 'inputs_offsets': [4, 11]},  # for P7"
  # ]
  num_levels = max_level - min_level + 1
  node_ids = {min_level + i: [i] for i in range(num_levels)}

  level_last_id = lambda level: node_ids[level][-1]
  level_all_ids = lambda level: node_ids[level]
  id_cnt = itertools.count(num_levels)

  p.nodes = []
  for i in range(max_level - 1, min_level - 1, -1):
    # top-down path.
    p.nodes.append({
        'feat_level': i,
        'inputs_offsets': [level_last_id(i),
                           level_last_id(i + 1)]
    })
    node_ids[i].append(next(id_cnt))

  for i in range(min_level + 1, max_level + 1):
    # bottom-up path.
    p.nodes.append({
        'feat_level': i,
        'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)]
    })
    node_ids[i].append(next(id_cnt))

  return p


def qufpn_config(min_level, max_level, weight_method=None):
  """A dynamic quad fpn config that can adapt to different min/max levels."""
  # It extends the idea of BiFPN, and has four paths:
  #   (up_down -> bottom_up) + (bottom_up -> up_down).
  # See test for an example for level 2 and 7.
  p = hparams_config.Config()
  p.weight_method = weight_method or 'fastattn'
  p.quad_method = 'fastattn'
  num_levels = max_level - min_level + 1
  node_ids = {min_level + i: [i] for i in range(num_levels)}
  level_last_id = lambda level: node_ids[level][-1]
  level_all_ids = lambda level: node_ids[level]
  level_first_id = lambda level: node_ids[level][0]
  id_cnt = itertools.count(num_levels)

  p.nodes = []
  for i in range(max_level - 1, min_level - 1, -1):
    # top-down path 1.
    p.nodes.append({
        'feat_level': i,
        'inputs_offsets': [level_last_id(i),
                           level_last_id(i + 1)],
        'weight_method': p.weight_method
    })
    node_ids[i].append(next(id_cnt))
  node_ids[max_level].append(node_ids[max_level][-1])

  for i in range(min_level + 1, max_level):
    # bottom-up path 2.
    p.nodes.append({
        'feat_level': i,
        'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)],
        'weight_method': p.weight_method
    })
    node_ids[i].append(next(id_cnt))

  i = max_level
  p.nodes.append({
      'feat_level': i,
      'inputs_offsets': [level_first_id(i)] + [level_last_id(i - 1)],
      'weight_method': p.weight_method
  })
  node_ids[i].append(next(id_cnt))
  node_ids[min_level].append(node_ids[min_level][-1])

  for i in range(min_level + 1, max_level + 1, 1):
    # bottom-up path 3.
    p.nodes.append({
        'feat_level': i,
        'inputs_offsets': [
            level_first_id(i),
            level_last_id(i - 1) if i != min_level + 1 else level_first_id(i -
                                                                           1)
        ],
        'weight_method': p.weight_method
    })
    node_ids[i].append(next(id_cnt))
  node_ids[min_level].append(node_ids[min_level][-1])

  for i in range(max_level - 1, min_level, -1):
    # top-down path 4.
    p.nodes.append({
        'feat_level':
            i,
        'inputs_offsets': [node_ids[i][0]] + [node_ids[i][-1]] +
                          [level_last_id(i + 1)],
        'weight_method':
            p.weight_method
    })
    node_ids[i].append(next(id_cnt))
  i = min_level
  p.nodes.append({
      'feat_level': i,
      'inputs_offsets': [node_ids[i][0]] + [level_last_id(i + 1)],
      'weight_method': p.weight_method
  })
  node_ids[i].append(next(id_cnt))
  node_ids[max_level].append(node_ids[max_level][-1])

  for i in range(max_level, min_level - 1, -1):
    # quad-add path.
    p.nodes.append({
        'feat_level': i,
        'inputs_offsets': [node_ids[i][2], node_ids[i][4]],
        'weight_method': p.quad_method
    })
    node_ids[i].append(next(id_cnt))

  return p


def get_fpn_config(fpn_name, min_level, max_level, weight_method):
  """Get fpn related configuration."""
  if not fpn_name:
    fpn_name = 'bifpn'
  name_to_config = {
      'bifpn': bifpn_config(min_level, max_level, weight_method),
      'qufpn': qufpn_config(min_level, max_level, weight_method),
      # legacy only: to be deprecated.
      'bifpn_dyn': bifpn_config(min_level, max_level, weight_method),
  }
  return name_to_config[fpn_name]
