# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

CATEGORICAL_CHANNEL = "categorical"
NUMERICAL_CHANNEL = "numerical"
LABEL_CHANNEL = "label"

SPLIT_BINARY = "split_binary"

TRAIN_MAPPING = "train"
TEST_MAPPING = "test"

TYPE_SELECTOR = "type"
FEATURES_SELECTOR = "features"
FILES_SELECTOR = "files"

DTYPE_SELECTOR = "dtype"
CARDINALITY_SELECTOR = "cardinality"


def get_categorical_feature_type(size: int):
    """This function works both when max value and cardinality is passed.
        Consistency by the user is required"""
    types = (np.int8, np.int16, np.int32)

    for numpy_type in types:
        if size < np.iinfo(numpy_type).max:
            return numpy_type

    raise RuntimeError(f"Categorical feature of size {size} is too big for defined types")
