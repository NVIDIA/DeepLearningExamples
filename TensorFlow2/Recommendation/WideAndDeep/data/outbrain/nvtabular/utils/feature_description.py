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

DISPLAY_ID_COLUMN = "display_id"

BASE_CONT_COLUMNS = [
    "publish_time",
    "publish_time_promo",
    "timestamp",
    "document_id_promo_clicked_sum_ctr",
    "publisher_id_promo_clicked_sum_ctr",
    "source_id_promo_clicked_sum_ctr",
    "document_id_promo_count",
    "publish_time_days_since_published",
    "ad_id_clicked_sum_ctr",
    "advertiser_id_clicked_sum_ctr",
    "campaign_id_clicked_sum_ctr",
    "ad_id_count",
    "publish_time_promo_days_since_published",
]

SIM_COLUMNS = [
    "doc_event_doc_ad_sim_categories",
    "doc_event_doc_ad_sim_topics",
    "doc_event_doc_ad_sim_entities",
]

CONTINUOUS_COLUMNS = BASE_CONT_COLUMNS + SIM_COLUMNS + [DISPLAY_ID_COLUMN]

exclude_conts = ["publish_time", "publish_time_promo", "timestamp"]

NUMERIC_COLUMNS = [col for col in CONTINUOUS_COLUMNS if col not in exclude_conts]

CATEGORICAL_COLUMNS = [
    "ad_id",
    "document_id",
    "platform",
    "document_id_promo",
    "campaign_id",
    "advertiser_id",
    "source_id",
    "publisher_id",
    "source_id_promo",
    "publisher_id_promo",
]

CTR_INPUTS = [
    "ad_id",
    "source_id_promo",
    "document_id_promo",
    "publisher_id_promo",
    "advertiser_id",
    "campaign_id",
]

EXCLUDE_COLUMNS = [
    "publish_time",
    "publish_time_promo",
    "timestamp",
    "ad_id_clicked_sum",
    "source_id_promo_count",
    "source_id_promo_clicked_sum",
    "document_id_promo_clicked_sum",
    "publisher_id_promo_count",
    "publisher_id_promo_clicked_sum",
    "advertiser_id_count",
    "advertiser_id_clicked_sum",
    "campaign_id_count",
    "campaign_id_clicked_sum",
    "uuid",
    "day_event",
]
