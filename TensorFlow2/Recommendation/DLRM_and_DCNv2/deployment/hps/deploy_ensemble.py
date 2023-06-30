# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#
# author: Tomasz Grel (tgrel@nvidia.com)


import os
from collections import namedtuple


Tensor = namedtuple("Tensor", ["name", "dtype", "dims"])

_config_template = r'''
name: "{model_name}"
platform: "ensemble"
max_batch_size: {max_batch_size}
input [
  {{
    name: "EMB_KEY"
    data_type: TYPE_INT64
    dims: [-1]
  }},
  {{
    name: "EMB_N_KEY"
    data_type: TYPE_INT32
    dims: [-1]
  }},
  {{
    name: "numerical_features"
    data_type: TYPE_FP32
    dims: [-1]
  }}
]
output [
  {{
    name: "DENSE_OUTPUT"
    data_type: TYPE_FP32
    dims: [-1]
  }}
]
ensemble_scheduling {{
  step [
    {{
      model_name: "{sparse_model_name}"
      model_version: -1
      input_map {{
        key: "KEYS"
        value: "EMB_KEY"
      }},
      input_map {{
        key: "NUMKEYS"
        value: "EMB_N_KEY"
      }},
      output_map {{
        key: "OUTPUT0"
        value: "LOOKUP_VECTORS"
      }}
    }},
    {{
      model_name: "{dense_model_name}"
      model_version: -1
      input_map {{
        key: "args_1"
        value: "LOOKUP_VECTORS"
      }},
      input_map {{
        key: "args_0"
        value: "numerical_features"
      }},
      output_map {{
        key: "output_1"
        value: "DENSE_OUTPUT"
      }}
    }}
  ]
}}
'''


def deploy_ensemble(dst, model_name, sparse_model_name, dense_model_name,
                    num_cat_features, num_numerical_features, max_batch_size, version):

    config_str = _config_template.format(model_name=model_name,
                                         sparse_model_name=sparse_model_name,
                                         dense_model_name=dense_model_name,
                                         max_batch_size=max_batch_size)

    with open(os.path.join(dst, "config.pbtxt"), "w") as f:
        f.write(config_str)
    os.mkdir(os.path.join(dst, str(version)))

    print("Ensemble configuration:")
    print(config_str)
