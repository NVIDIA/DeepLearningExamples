# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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


# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""File paths for the Criteo Classification pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


TEMP_DIR = 'tmp'
TRANSFORM_FN_DIR = 'transform_fn'
RAW_METADATA_DIR = 'raw_metadata'
TRANSFORMED_METADATA_DIR = 'transformed_metadata'
TRANSFORMED_TRAIN_DATA_FILE_PREFIX = 'features_train'
TRANSFORMED_EVAL_DATA_FILE_PREFIX = 'features_eval'
TRANSFORMED_PREDICT_DATA_FILE_PREFIX = 'features_predict'
TRAIN_RESULTS_FILE = 'train_results'
DEPLOY_SAVED_MODEL_DIR = 'saved_model'
MODEL_EVALUATIONS_FILE = 'model_evaluations'
BATCH_PREDICTION_RESULTS_FILE = 'batch_prediction_results'
