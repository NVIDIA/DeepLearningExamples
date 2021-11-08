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

CATEGORICAL_COLUMNS = [
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
}

EMBEDDING_TABLE_SHAPES = {
    column: (HASH_BUCKET_SIZES[column], EMBEDDING_DIMENSIONS[column])
    for column in CATEGORICAL_COLUMNS
}


def get_features_keys():
    return CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [DISPLAY_ID_COLUMN]


def get_feature_columns():
    logger = logging.getLogger("tensorflow")
    wide_columns, deep_columns = [], []

    for column_name in CATEGORICAL_COLUMNS:
        if column_name in EMBEDDING_TABLE_SHAPES:
            categorical_column = tf.feature_column.categorical_column_with_identity(
                column_name, num_buckets=EMBEDDING_TABLE_SHAPES[column_name][0]
            )
            wrapped_column = tf.feature_column.embedding_column(
                categorical_column,
                dimension=EMBEDDING_TABLE_SHAPES[column_name][1],
                combiner="mean",
            )
        else:
            raise ValueError(f"Unexpected categorical column found {column_name}")

        wide_columns.append(categorical_column)
        deep_columns.append(wrapped_column)

    numerics = [
        tf.feature_column.numeric_column(column_name, shape=(1,), dtype=tf.float32)
        for column_name in NUMERIC_COLUMNS
        if column_name != DISPLAY_ID_COLUMN
    ]

    wide_columns.extend(numerics)
    deep_columns.extend(numerics)

    logger.warning("deep columns: {}".format(len(deep_columns)))
    logger.warning("wide columns: {}".format(len(wide_columns)))
    logger.warning(
        "wide&deep intersection: {}".format(
            len(set(wide_columns).intersection(set(deep_columns)))
        )
    )

    return wide_columns, deep_columns
