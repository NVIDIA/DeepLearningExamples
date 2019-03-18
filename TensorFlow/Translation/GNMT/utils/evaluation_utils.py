# Copyright 2017 Google Inc. All Rights Reserved.
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
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

"""Utility for evaluating various tasks, e.g., translation & summarization."""
import codecs
import os
import re
import subprocess

import tensorflow as tf

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.training import checkpoint_management as cm


def get_all_checkpoints(output_dir):
  """docstring."""
  ckpt = cm.get_checkpoint_state(output_dir, None)
  res = []
  if not ckpt:
    return None
  for path in ckpt.all_model_checkpoint_paths:
    # Look for either a V2 path or a V1 path, with priority for V2.
    v2_path = cm._prefix_to_checkpoint_path(path, saver_pb2.SaverDef.V2)
    v1_path = cm._prefix_to_checkpoint_path(path, saver_pb2.SaverDef.V1)
    if file_io.get_matching_files(v2_path) or file_io.get_matching_files(
        v1_path):
      res.append(path)
    else:
      tf.logging.error("Couldn't match files for checkpoint %s", path)
  return res
