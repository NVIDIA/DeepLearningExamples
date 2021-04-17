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

import logging

import tensorflow as tf

PREBATCH_SIZE = 4096
DISPLAY_ID_COLUMN = 'display_id'

TIME_COLUMNS = [
    'doc_event_days_since_published_log_01scaled',
    'doc_ad_days_since_published_log_01scaled'
]

GB_COLUMNS = [
    'pop_document_id',
    'pop_publisher_id',
    'pop_source_id',
    'pop_ad_id',
    'pop_advertiser_id',
    'pop_campain_id',
    'doc_views_log_01scaled',
    'ad_views_log_01scaled'
]

SIM_COLUMNS = [
    'doc_event_doc_ad_sim_categories',
    'doc_event_doc_ad_sim_topics',
    'doc_event_doc_ad_sim_entities'
]

NUMERIC_COLUMNS = TIME_COLUMNS + SIM_COLUMNS + GB_COLUMNS

CATEGORICAL_COLUMNS = [
    'ad_id',
    'campaign_id',
    'doc_event_id',
    'event_platform',
    'doc_id',
    'ad_advertiser',
    'doc_event_source_id',
    'doc_event_publisher_id',
    'doc_ad_source_id',
    'doc_ad_publisher_id',
    'event_geo_location',
    'event_country',
    'event_country_state',
]

HASH_BUCKET_SIZES = {
    'doc_event_id': 300000,
    'ad_id': 250000,
    'doc_id': 100000,
    'doc_ad_source_id': 4000,
    'doc_event_source_id': 4000,
    'event_geo_location': 2500,
    'ad_advertiser': 2500,
    'event_country_state': 2000,
    'doc_ad_publisher_id': 1000,
    'doc_event_publisher_id': 1000,
    'event_country': 300,
    'event_platform': 4,
    'campaign_id': 5000
}

EMBEDDING_DIMENSIONS = {
    'doc_event_id': 128,
    'ad_id': 128,
    'doc_id': 128,
    'doc_ad_source_id': 64,
    'doc_event_source_id': 64,
    'event_geo_location': 64,
    'ad_advertiser': 64,
    'event_country_state': 64,
    'doc_ad_publisher_id': 64,
    'doc_event_publisher_id': 64,
    'event_country': 64,
    'event_platform': 16,
    'campaign_id': 128
}

EMBEDDING_TABLE_SHAPES = {
    column: (HASH_BUCKET_SIZES[column], EMBEDDING_DIMENSIONS[column]) for column in CATEGORICAL_COLUMNS
}


def get_features_keys():
    return CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [DISPLAY_ID_COLUMN]


def get_feature_columns():
    logger = logging.getLogger('tensorflow')
    wide_columns, deep_columns = [], []

    for column_name in CATEGORICAL_COLUMNS:
        if column_name in EMBEDDING_TABLE_SHAPES:
            categorical_column = tf.feature_column.categorical_column_with_identity(
                column_name, num_buckets=EMBEDDING_TABLE_SHAPES[column_name][0])
            wrapped_column = tf.feature_column.embedding_column(
                categorical_column,
                dimension=EMBEDDING_TABLE_SHAPES[column_name][1],
                combiner='mean')
        else:
            raise ValueError(f'Unexpected categorical column found {column_name}')

        wide_columns.append(categorical_column)
        deep_columns.append(wrapped_column)

    numerics = [tf.feature_column.numeric_column(column_name, shape=(1,), dtype=tf.float32)
                for column_name in NUMERIC_COLUMNS]

    wide_columns.extend(numerics)
    deep_columns.extend(numerics)

    logger.warning('deep columns: {}'.format(len(deep_columns)))
    logger.warning('wide columns: {}'.format(len(wide_columns)))
    logger.warning('wide&deep intersection: {}'.format(len(set(wide_columns).intersection(set(deep_columns)))))

    return wide_columns, deep_columns
