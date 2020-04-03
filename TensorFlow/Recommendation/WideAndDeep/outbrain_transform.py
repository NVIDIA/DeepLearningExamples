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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import metadata_io
import numpy as np

from trainer.features import LABEL_COLUMN, DISPLAY_ID_COLUMN, IS_LEAK_COLUMN, DISPLAY_ID_AND_IS_LEAK_ENCODED_COLUMN, CATEGORICAL_COLUMNS, DOC_CATEGORICAL_MULTIVALUED_COLUMNS, BOOL_COLUMNS, INT_COLUMNS, FLOAT_COLUMNS, FLOAT_COLUMNS_LOG_BIN_TRANSFORM, FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM

RENAME_COLUMNS = False

CSV_ORDERED_COLUMNS = ['label','display_id','ad_id','doc_id','doc_event_id','is_leak','event_weekend',
              'user_has_already_viewed_doc','user_views','ad_views','doc_views',
              'doc_event_days_since_published','doc_event_hour','doc_ad_days_since_published',              
              'pop_ad_id','pop_ad_id_conf',
              'pop_ad_id_conf_multipl','pop_document_id','pop_document_id_conf',
              'pop_document_id_conf_multipl','pop_publisher_id','pop_publisher_id_conf',
              'pop_publisher_id_conf_multipl','pop_advertiser_id','pop_advertiser_id_conf',
              'pop_advertiser_id_conf_multipl','pop_campain_id','pop_campain_id_conf',
              'pop_campain_id_conf_multipl','pop_doc_event_doc_ad','pop_doc_event_doc_ad_conf',
              'pop_doc_event_doc_ad_conf_multipl','pop_source_id','pop_source_id_conf',
              'pop_source_id_conf_multipl','pop_source_id_country','pop_source_id_country_conf',
              'pop_source_id_country_conf_multipl','pop_entity_id','pop_entity_id_conf',
              'pop_entity_id_conf_multipl','pop_entity_id_country','pop_entity_id_country_conf',
              'pop_entity_id_country_conf_multipl','pop_topic_id','pop_topic_id_conf',
              'pop_topic_id_conf_multipl','pop_topic_id_country','pop_topic_id_country_conf',
              'pop_topic_id_country_conf_multipl','pop_category_id','pop_category_id_conf',
              'pop_category_id_conf_multipl','pop_category_id_country','pop_category_id_country_conf',
              'pop_category_id_country_conf_multipl','user_doc_ad_sim_categories',
              'user_doc_ad_sim_categories_conf','user_doc_ad_sim_categories_conf_multipl',
              'user_doc_ad_sim_topics','user_doc_ad_sim_topics_conf','user_doc_ad_sim_topics_conf_multipl',
              'user_doc_ad_sim_entities','user_doc_ad_sim_entities_conf','user_doc_ad_sim_entities_conf_multipl',
              'doc_event_doc_ad_sim_categories','doc_event_doc_ad_sim_categories_conf',
              'doc_event_doc_ad_sim_categories_conf_multipl','doc_event_doc_ad_sim_topics',
              'doc_event_doc_ad_sim_topics_conf','doc_event_doc_ad_sim_topics_conf_multipl',
              'doc_event_doc_ad_sim_entities','doc_event_doc_ad_sim_entities_conf',
              'doc_event_doc_ad_sim_entities_conf_multipl','ad_advertiser','doc_ad_category_id_1',
              'doc_ad_category_id_2','doc_ad_category_id_3','doc_ad_topic_id_1','doc_ad_topic_id_2',
              'doc_ad_topic_id_3','doc_ad_entity_id_1','doc_ad_entity_id_2','doc_ad_entity_id_3',
              'doc_ad_entity_id_4','doc_ad_entity_id_5','doc_ad_entity_id_6','doc_ad_publisher_id',
              'doc_ad_source_id','doc_event_category_id_1','doc_event_category_id_2','doc_event_category_id_3',
              'doc_event_topic_id_1','doc_event_topic_id_2','doc_event_topic_id_3','doc_event_entity_id_1',
              'doc_event_entity_id_2','doc_event_entity_id_3','doc_event_entity_id_4','doc_event_entity_id_5',
              'doc_event_entity_id_6','doc_event_publisher_id','doc_event_source_id','event_country',
              'event_country_state','event_geo_location','event_hour','event_platform','traffic_source']

def make_spec(output_dir, batch_size=None):
  fixed_shape = [batch_size,1] if batch_size is not None else []
  spec = {}
  spec[LABEL_COLUMN] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
  spec[DISPLAY_ID_COLUMN] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
  spec[IS_LEAK_COLUMN] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
  spec[DISPLAY_ID_AND_IS_LEAK_ENCODED_COLUMN] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)

  for name in BOOL_COLUMNS:
    spec[name] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
  for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM+FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM:
    spec[name] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.float32, default_value=None)  
  for name in FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM:
    spec[name + '_binned'] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
  for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM:
    spec[name + '_binned'] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
    spec[name + '_log_01scaled'] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.float32, default_value=None)
  for name in INT_COLUMNS:
    spec[name + '_log_int'] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
    spec[name + '_log_01scaled'] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.float32, default_value=None)
  for name in BOOL_COLUMNS + CATEGORICAL_COLUMNS:
    spec[name] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)

  for multi_category in DOC_CATEGORICAL_MULTIVALUED_COLUMNS:
    #spec[multi_category] = tf.VarLenFeature(dtype=tf.int64)
    shape = fixed_shape[:-1]+[len(DOC_CATEGORICAL_MULTIVALUED_COLUMNS[multi_category])]
    spec[multi_category] = tf.FixedLenFeature(shape=shape, dtype=tf.int64)

  metadata = dataset_metadata.DatasetMetadata(dataset_schema.from_feature_spec(spec))
	
  metadata_io.write_metadata(metadata, output_dir)

def tf_log2_1p(x):
  return tf.log1p(x) / tf.log(2.0)

def log2_1p(x):
  return np.log1p(x) / np.log(2.0)

def compute_min_max_logs(rows):
  min_logs = {}
  max_logs = {}
  
  for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM + INT_COLUMNS:
    min_logs[name + '_log_01scaled'] = float("inf")
    max_logs[name + '_log_01scaled'] = float("-inf")

  for row in rows:
    names = CSV_ORDERED_COLUMNS
    columns_dict = dict(zip(names, row))
    for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM:
      nn = name + '_log_01scaled'
      min_logs[nn] = min(min_logs[nn], log2_1p(columns_dict[name] * 1000))
      max_logs[nn] = max(max_logs[nn], log2_1p(columns_dict[name] * 1000))
    for name in INT_COLUMNS:
      nn = name + '_log_01scaled'
      min_logs[nn] = min(min_logs[nn], log2_1p(columns_dict[name]))
      max_logs[nn] = max(max_logs[nn], log2_1p(columns_dict[name]))

  return min_logs, max_logs

def scale_to_0_1(val, minv, maxv):
  return (val - minv) / (maxv - minv)

def create_tf_example(df, min_logs, max_logs):
  result = {}
  result[LABEL_COLUMN] = tf.train.Feature(int64_list=tf.train.Int64List(value=df[LABEL_COLUMN].to_list()))
  result[DISPLAY_ID_COLUMN] = tf.train.Feature(int64_list=tf.train.Int64List(value=df[DISPLAY_ID_COLUMN].to_list()))
  result[IS_LEAK_COLUMN] = tf.train.Feature(int64_list=tf.train.Int64List(value=df[IS_LEAK_COLUMN].to_list()))
  #is_leak = df[IS_LEAK_COLUMN].to_list()
  encoded_value = df[DISPLAY_ID_COLUMN].multiply(10).add(df[IS_LEAK_COLUMN].clip(lower=0)).to_list()
  # * 10 + (0 if is_leak < 0 else is_leak)
  result[DISPLAY_ID_AND_IS_LEAK_ENCODED_COLUMN] = tf.train.Feature(int64_list=tf.train.Int64List(value=encoded_value))
  
  for name in FLOAT_COLUMNS:
    result[name] = tf.train.Feature(float_list=tf.train.FloatList(value=df[name].to_list()))
  for name in FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM:
    #[int(columns_dict[name] * 10)]
    value = df[name].multiply(10).astype('int64').to_list()
    result[name + '_binned'] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
  for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM:
    # [int(log2_1p(columns_dict[name] * 1000))]
    value_prelim = df[name].multiply(1000).apply(np.log1p).multiply(1./np.log(2.0))
    value = value_prelim.astype('int64').to_list()
    result[name + '_binned'] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    nn = name + '_log_01scaled'
    #val = log2_1p(columns_dict[name] * 1000)
    #val = scale_to_0_1(val, min_logs[nn], max_logs[nn])
    value = value_prelim.add(-min_logs[nn]).multiply(1./(max_logs[nn]-min_logs[nn])).to_list()
    result[nn] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
  for name in INT_COLUMNS:
    #[int(log2_1p(columns_dict[name]))]
    value_prelim = df[name].apply(np.log1p).multiply(1./np.log(2.0))
    value = value_prelim.astype('int64').to_list()
    result[name + '_log_int'] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    nn = name + '_log_01scaled'
    #val = log2_1p(columns_dict[name])
    #val = scale_to_0_1(val, min_logs[nn], max_logs[nn])
    value = value_prelim.add(-min_logs[nn]).multiply(1./(max_logs[nn]-min_logs[nn])).to_list()
    result[nn] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
  
  for name in BOOL_COLUMNS + CATEGORICAL_COLUMNS:
    result[name] = tf.train.Feature(int64_list=tf.train.Int64List(value=df[name].to_list()))
  
  for multi_category in DOC_CATEGORICAL_MULTIVALUED_COLUMNS:
    values = []
    for category in DOC_CATEGORICAL_MULTIVALUED_COLUMNS[multi_category]:
      values = values + [df[category].to_numpy()]
    # need to transpose the series so they will be parsed correctly by the FixedLenFeature
    # we can pass in a single series here; they'll be reshaped to [batch_size, num_values]
    # when parsed from the TFRecord
    value = np.stack(values, axis=1).flatten().tolist()
    result[multi_category] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))   

  tf_example = tf.train.Example(features=tf.train.Features(feature=result))

  return tf_example
