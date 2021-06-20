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

LABEL_COLUMN = "label"

DISPLAY_ID_COLUMN = 'display_id'

IS_LEAK_COLUMN = 'is_leak'

DISPLAY_ID_AND_IS_LEAK_ENCODED_COLUMN = 'display_ad_and_is_leak'

CATEGORICAL_COLUMNS = [
    'ad_id',
    'campaign_id',
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
    'event_platform']

DOC_CATEGORICAL_MULTIVALUED_COLUMNS = {
}

BOOL_COLUMNS = []

INT_COLUMNS = [
    'ad_views',
    'doc_views',
    'doc_event_days_since_published',
    'doc_ad_days_since_published']

FLOAT_COLUMNS_LOG_BIN_TRANSFORM = []
FLOAT_COLUMNS_NO_TRANSFORM = [
    'pop_ad_id',
    'pop_document_id',
    'pop_publisher_id',
    'pop_advertiser_id',
    'pop_campain_id',
    'pop_source_id',
    'doc_event_doc_ad_sim_categories',
    'doc_event_doc_ad_sim_topics',
    'doc_event_doc_ad_sim_entities',
]
FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM = []
FLOAT_COLUMNS = FLOAT_COLUMNS_LOG_BIN_TRANSFORM + FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM + FLOAT_COLUMNS_NO_TRANSFORM

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
    'campaign_id',
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
