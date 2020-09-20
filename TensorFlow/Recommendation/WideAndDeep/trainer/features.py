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

import tensorflow as tf

LABEL_COLUMN = "label"

DISPLAY_ID_COLUMN = 'display_id'

IS_LEAK_COLUMN = 'is_leak'

DISPLAY_ID_AND_IS_LEAK_ENCODED_COLUMN = 'display_ad_and_is_leak'

CATEGORICAL_COLUMNS = [
    'ad_id',
    'doc_id',
    'doc_event_id',
    'ad_advertiser',
    'doc_ad_source_id',
    'doc_ad_publisher_id',
    'doc_event_publisher_id',
    'doc_event_source_id',
    'event_country',
    'event_country_state',
    'event_geo_location',
    'event_hour',
    'event_platform',
    'traffic_source']

DOC_CATEGORICAL_MULTIVALUED_COLUMNS = {
    'doc_ad_category_id': ['doc_ad_category_id_1',
                           'doc_ad_category_id_2',
                           'doc_ad_category_id_3'],
    'doc_ad_topic_id': ['doc_ad_topic_id_1',
                        'doc_ad_topic_id_2',
                        'doc_ad_topic_id_3'],
    'doc_ad_entity_id': ['doc_ad_entity_id_1',
                         'doc_ad_entity_id_2',
                         'doc_ad_entity_id_3',
                         'doc_ad_entity_id_4',
                         'doc_ad_entity_id_5',
                         'doc_ad_entity_id_6'],
    'doc_event_category_id': ['doc_event_category_id_1',
                              'doc_event_category_id_2',
                              'doc_event_category_id_3'],
    'doc_event_topic_id': ['doc_event_topic_id_1',
                           'doc_event_topic_id_2',
                           'doc_event_topic_id_3'],
    'doc_event_entity_id': ['doc_event_entity_id_1',
                            'doc_event_entity_id_2',
                            'doc_event_entity_id_3',
                            'doc_event_entity_id_4',
                            'doc_event_entity_id_5',
                            'doc_event_entity_id_6']
}

BOOL_COLUMNS = [
    'event_weekend',
    'user_has_already_viewed_doc']

INT_COLUMNS = [
    'user_views',
    'ad_views',
    'doc_views',
    'doc_event_days_since_published',
    'doc_event_hour',
    'doc_ad_days_since_published']

FLOAT_COLUMNS_LOG_BIN_TRANSFORM = [
    'pop_ad_id',
    'pop_ad_id_conf_multipl',
    'pop_document_id',
    'pop_document_id_conf_multipl',
    'pop_publisher_id',
    'pop_publisher_id_conf_multipl',
    'pop_advertiser_id',
    'pop_advertiser_id_conf_multipl',
    'pop_campain_id',
    'pop_campain_id_conf_multipl',
    'pop_doc_event_doc_ad',
    'pop_doc_event_doc_ad_conf_multipl',
    'pop_source_id',
    'pop_source_id_conf_multipl',
    'pop_source_id_country',
    'pop_source_id_country_conf_multipl',
    'pop_entity_id',
    'pop_entity_id_conf_multipl',
    'pop_entity_id_country',
    'pop_entity_id_country_conf_multipl',
    'pop_topic_id',
    'pop_topic_id_conf_multipl',
    'pop_topic_id_country',
    'pop_topic_id_country_conf_multipl',
    'pop_category_id',
    'pop_category_id_conf_multipl',
    'pop_category_id_country',
    'pop_category_id_country_conf_multipl',
    'user_doc_ad_sim_categories',
    'user_doc_ad_sim_categories_conf_multipl',
    'user_doc_ad_sim_topics',
    'user_doc_ad_sim_topics_conf_multipl',
    'user_doc_ad_sim_entities',
    'user_doc_ad_sim_entities_conf_multipl',
    'doc_event_doc_ad_sim_categories',
    'doc_event_doc_ad_sim_categories_conf_multipl',
    'doc_event_doc_ad_sim_topics',
    'doc_event_doc_ad_sim_topics_conf_multipl',
    'doc_event_doc_ad_sim_entities',
    'doc_event_doc_ad_sim_entities_conf_multipl']

FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM = [
    'pop_ad_id_conf',
    'pop_document_id_conf',
    'pop_publisher_id_conf',
    'pop_advertiser_id_conf',
    'pop_campain_id_conf',
    'pop_doc_event_doc_ad_conf',
    'pop_source_id_conf',
    'pop_source_id_country_conf',
    'pop_entity_id_conf',
    'pop_entity_id_country_conf',
    'pop_topic_id_conf',
    'pop_topic_id_country_conf',
    'pop_category_id_conf',
    'pop_category_id_country_conf',
    'user_doc_ad_sim_categories_conf',
    'user_doc_ad_sim_topics_conf',
    'user_doc_ad_sim_entities_conf',
    'doc_event_doc_ad_sim_categories_conf',
    'doc_event_doc_ad_sim_topics_conf',
    'doc_event_doc_ad_sim_entities_conf']

FLOAT_COLUMNS = FLOAT_COLUMNS_LOG_BIN_TRANSFORM + FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM

# Let's define the columns we're actually going to use
# during training
REQUEST_SINGLE_HOT_COLUMNS = [
    "doc_event_id",
    "doc_id",
    "doc_event_source_id",
    "event_geo_location",
    "event_country_state",
    "doc_event_publisher_id",
    "event_country",
    "event_hour",
    "event_platform",
    "traffic_source",
    "event_weekend",
    "user_has_already_viewed_doc"]

REQUEST_MULTI_HOT_COLUMNS = [
    "doc_event_entity_id",
    "doc_event_topic_id",
    "doc_event_category_id"]

REQUEST_NUMERIC_COLUMNS = [
    "pop_document_id_conf",
    "pop_publisher_id_conf",
    "pop_source_id_conf",
    "pop_entity_id_conf",
    "pop_topic_id_conf",
    "pop_category_id_conf",
    "pop_document_id",
    "pop_publisher_id",
    "pop_source_id",
    "pop_entity_id",
    "pop_topic_id",
    "pop_category_id",
    "user_views",
    "doc_views",
    "doc_event_days_since_published",
    "doc_event_hour"]

ITEM_SINGLE_HOT_COLUMNS = [
    "ad_id",
    "doc_ad_source_id",
    "ad_advertiser",
    "doc_ad_publisher_id"]

ITEM_MULTI_HOT_COLUMNS = [
    "doc_ad_topic_id",
    "doc_ad_entity_id",
    "doc_ad_category_id"]

ITEM_NUMERIC_COLUMNS = [
    "pop_ad_id_conf",
    "user_doc_ad_sim_categories_conf",
    "user_doc_ad_sim_topics_conf",
    "pop_advertiser_id_conf",
    "pop_campain_id_conf_multipl",
    "pop_ad_id",
    "pop_advertiser_id",
    "pop_campain_id",
    "user_doc_ad_sim_categories",
    "user_doc_ad_sim_topics",
    "user_doc_ad_sim_entities",
    "doc_event_doc_ad_sim_categories",
    "doc_event_doc_ad_sim_topics",
    "doc_event_doc_ad_sim_entities",
    "ad_views",
    "doc_ad_days_since_published"]

NV_TRAINING_COLUMNS = (
        REQUEST_SINGLE_HOT_COLUMNS +
        REQUEST_MULTI_HOT_COLUMNS +
        REQUEST_NUMERIC_COLUMNS +
        ITEM_SINGLE_HOT_COLUMNS +
        ITEM_MULTI_HOT_COLUMNS +
        ITEM_NUMERIC_COLUMNS)

HASH_BUCKET_SIZES = {
    'doc_event_id': 300000,
    'ad_id': 250000,
    'doc_id': 100000,
    'doc_ad_entity_id': 10000,
    'doc_event_entity_id': 10000,
    'doc_ad_source_id': 4000,
    'doc_event_source_id': 4000,
    'event_geo_location': 2500,
    'ad_advertiser': 2500,
    'event_country_state': 2000,
    'doc_ad_publisher_id': 1000,
    'doc_event_publisher_id': 1000,
    'doc_ad_topic_id': 350,
    'doc_event_topic_id': 350,
    'event_country': 300,
    'doc_ad_category_id': 100,
    'doc_event_category_id': 100}

IDENTITY_NUM_BUCKETS = {
    'event_hour': 6,
    'event_platform': 3,
    'traffic_source': 3,
    'event_weekend': 2,
    'user_has_already_viewed_doc': 2}

EMBEDDING_DIMENSIONS = {
    'doc_event_id': 128,
    'ad_id': 128,
    'doc_id': 128,
    'doc_ad_entity_id': 64,
    'doc_event_entity_id': 64,
    'doc_ad_source_id': 64,
    'doc_event_source_id': 64,
    'event_geo_location': 64,
    'ad_advertiser': 64,
    'event_country_state': 64,
    'doc_ad_publisher_id': 64,
    'doc_event_publisher_id': 64,
    'doc_ad_topic_id': 64,
    'doc_event_topic_id': 64,
    'event_country': 64,
    'doc_ad_category_id': 64,
    'doc_event_category_id': 64}


def get_feature_columns(force_subset=None):
    # adding the force_subset as a way to directly pass in column changes for testing/profiling
    deep_columns, wide_columns = [], []

    if force_subset is not None:
        training_columns = force_subset
    else:
        training_columns = NV_TRAINING_COLUMNS

    tf.compat.v1.logging.warn('number of features: {}'.format(len(training_columns)))

    for column_name in training_columns:
        if column_name in HASH_BUCKET_SIZES:
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                column_name,
                hash_bucket_size=HASH_BUCKET_SIZES[column_name],
                dtype=tf.int32)
            wide_columns.append(categorical_column)

        elif column_name in IDENTITY_NUM_BUCKETS:
            categorical_column = tf.feature_column.categorical_column_with_identity(
                column_name, num_buckets=IDENTITY_NUM_BUCKETS[column_name])
            wide_columns.append(categorical_column)

        else:
            columns = []
            if column_name in FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM:
                # add a categorical_column for column_name + "_binned"
                # just add the regular float column for now
                columns.append(tf.feature_column.numeric_column(
                    column_name, shape=(1,)))
            elif column_name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM:
                # add a categorical_column for column_name + "_log_binned")
                columns.append(tf.feature_column.numeric_column(
                    column_name + "_log_01scaled", shape=(1,)))
            elif column_name in INT_COLUMNS:
                # add a categorical_column for column_name + "_log_int"
                columns.append(tf.feature_column.numeric_column(
                    column_name + "_log_01scaled", shape=(1,)))

            for column in columns:
                wide_columns.append(column)
                deep_columns.append(column)
            continue
        if column_name in EMBEDDING_DIMENSIONS:
            column = tf.feature_column.embedding_column(
                categorical_column,
                dimension=EMBEDDING_DIMENSIONS[column_name],
                combiner='mean')
        else:
            column = tf.feature_column.indicator_column(categorical_column)
        deep_columns.append(column)
    tf.compat.v1.logging.warn('deep columns: {}'.format(len(deep_columns)))
    tf.compat.v1.logging.warn('wide columns: {}'.format(len(wide_columns)))
    tf.compat.v1.logging.warn(
        'wide&deep intersection: {}'.format(len(set(wide_columns).intersection(set(deep_columns)))))
    return wide_columns, deep_columns
