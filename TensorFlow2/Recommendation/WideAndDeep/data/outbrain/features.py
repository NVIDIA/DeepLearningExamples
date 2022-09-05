# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
from setuptools import glob

from data.feature_spec import CARDINALITY_SELECTOR, MAX_HOTNESS_SELECTOR, TYPE_SELECTOR, FEATURES_SELECTOR, \
    FILES_SELECTOR, FeatureSpec
from data.outbrain.defaults import TEST_MAPPING, TRAIN_MAPPING, PARQUET_TYPE, MULTIHOT_CHANNEL, ONEHOT_CHANNEL, \
    LABEL_CHANNEL, NUMERICAL_CHANNEL, MAP_FEATURE_CHANNEL
import os

DISPLAY_ID_COLUMN = "display_id"

NUMERIC_COLUMNS = [
    "document_id_document_id_promo_sim_categories",
    "document_id_document_id_promo_sim_topics",
    "document_id_document_id_promo_sim_entities",
    "document_id_promo_ctr",
    "publisher_id_promo_ctr",
    "source_id_promo_ctr",
    "document_id_promo_count",
    "publish_time_days_since_published",
    "ad_id_ctr",
    "advertiser_id_ctr",
    "campaign_id_ctr",
    "ad_id_count",
    "publish_time_promo_days_since_published",
]

ONEHOT_COLUMNS = [
    "ad_id",
    "document_id",
    "platform",
    "document_id_promo",
    "campaign_id",
    "advertiser_id",
    "source_id",
    "geo_location",
    "geo_location_country",
    "geo_location_state",
    "publisher_id",
    "source_id_promo",
    "publisher_id_promo",
]

# Multihot columns with their hotness
MULTIHOT_COLUMNS = {
    "topic_id_list": 3,
    "entity_id_list": 3,
    "category_id_list": 3
}

CATEGORICAL_COLUMNS = ONEHOT_COLUMNS + list(MULTIHOT_COLUMNS.keys())

HASH_BUCKET_SIZES = {
    "document_id": 300000,
    "ad_id": 250000,
    "document_id_promo": 100000,
    "source_id_promo": 4000,
    "source_id": 4000,
    "geo_location": 2500,
    "advertiser_id": 2500,
    "geo_location_state": 2000,
    "publisher_id_promo": 1000,
    "publisher_id": 1000,
    "geo_location_country": 300,
    "platform": 4,
    "campaign_id": 5000,
    "topic_id_list": 350,
    "entity_id_list": 10000,
    "category_id_list": 100,
}

EMBEDDING_DIMENSIONS = {
    "document_id": 128,
    "ad_id": 128,
    "document_id_promo": 128,
    "source_id_promo": 64,
    "source_id": 64,
    "geo_location": 64,
    "advertiser_id": 64,
    "geo_location_state": 64,
    "publisher_id_promo": 64,
    "publisher_id": 64,
    "geo_location_country": 64,
    "platform": 19,
    "campaign_id": 128,
    "topic_id_list": 64,
    "entity_id_list": 64,
    "category_id_list": 64,
}

LABEL_NAME = "clicked"

def get_features_keys():
    return CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [DISPLAY_ID_COLUMN]

def get_outbrain_feature_spec(base_directory):
    multihot_dict = {feature_name: {CARDINALITY_SELECTOR:HASH_BUCKET_SIZES[feature_name],
                                    MAX_HOTNESS_SELECTOR: hotness}
                     for feature_name, hotness in MULTIHOT_COLUMNS.items()}
    onehot_dict = {feature_name: {CARDINALITY_SELECTOR:HASH_BUCKET_SIZES[feature_name]}
                     for feature_name in ONEHOT_COLUMNS}
    numeric_dict = {feature_name: {} for feature_name in NUMERIC_COLUMNS}

    feature_dict = {**multihot_dict, **onehot_dict, **numeric_dict, DISPLAY_ID_COLUMN:{}, LABEL_NAME:{}}

    # these patterns come from partially our code (output_train_folder and output_valid_folder in utils/setup.py)
    # and partially from how nvtabular works (saving as sorted *.parquet in a chosen folder)
    train_data_pattern=f"{base_directory}/train/*.parquet"
    valid_data_pattern=f"{base_directory}/valid/*.parquet"
    absolute_train_paths = sorted(glob.glob(train_data_pattern))
    absolute_valid_paths = sorted(glob.glob(valid_data_pattern))
    train_paths = [os.path.relpath(p, base_directory) for p in absolute_train_paths]
    valid_paths = [os.path.relpath(p, base_directory) for p in absolute_valid_paths]

    source_spec = {}

    for mapping_name, paths in zip((TRAIN_MAPPING, TEST_MAPPING),(train_paths, valid_paths)):
        all_features = [LABEL_NAME] + ONEHOT_COLUMNS + list(MULTIHOT_COLUMNS.keys()) + NUMERIC_COLUMNS
        if mapping_name == TEST_MAPPING:
            all_features = all_features + [DISPLAY_ID_COLUMN]

        source_spec[mapping_name] = []
        source_spec[mapping_name].append({TYPE_SELECTOR: PARQUET_TYPE,
                                          FEATURES_SELECTOR: all_features,
                                          FILES_SELECTOR: paths})

    channel_spec = {MULTIHOT_CHANNEL: list(MULTIHOT_COLUMNS.keys()),
                    ONEHOT_CHANNEL: ONEHOT_COLUMNS,
                    LABEL_CHANNEL: [LABEL_NAME],
                    NUMERICAL_CHANNEL: NUMERIC_COLUMNS,
                    MAP_FEATURE_CHANNEL: [DISPLAY_ID_COLUMN]}

    return FeatureSpec(feature_spec=feature_dict, source_spec=source_spec, channel_spec=channel_spec, metadata={})