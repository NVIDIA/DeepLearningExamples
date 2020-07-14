#!/usr/bin/env python
# coding: utf-8

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


import argparse
import datetime
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import tensorflow as tf
import trainer
from pyspark import TaskContext
from pyspark.context import SparkContext, SparkConf
from pyspark.sql.functions import col, udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import ArrayType, DoubleType
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io
from trainer.features import LABEL_COLUMN, DISPLAY_ID_COLUMN, IS_LEAK_COLUMN, DISPLAY_ID_AND_IS_LEAK_ENCODED_COLUMN, \
    CATEGORICAL_COLUMNS, DOC_CATEGORICAL_MULTIVALUED_COLUMNS, BOOL_COLUMNS, INT_COLUMNS, FLOAT_COLUMNS, \
    FLOAT_COLUMNS_LOG_BIN_TRANSFORM, FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM

evaluation = True
evaluation_verbose = False

OUTPUT_BUCKET_FOLDER = "/outbrain/preprocessed/"
DATA_BUCKET_FOLDER = "/outbrain/orig/"
SPARK_TEMP_FOLDER = "/outbrain/spark-temp/"
LOCAL_DATA_TFRECORDS_DIR = "/outbrain/tfrecords"

TEST_SET_MODE = False

TENSORFLOW_HADOOP = "preproc/data/tensorflow-hadoop-1.5.0.jar"

conf = SparkConf().setMaster('local[*]').set('spark.executor.memory', '40g').set('spark.driver.memory', '200g').set(
    "spark.local.dir", SPARK_TEMP_FOLDER)
conf.set("spark.jars", TENSORFLOW_HADOOP)
conf.set("spark.sql.files.maxPartitionBytes", 805306368)

sc = SparkContext(conf=conf)
spark = SparkSession(sc)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--prebatch_size',
    help='Prebatch size in created tfrecords',
    type=int,
    default=4096)

parser.add_argument(
    '--submission',
    action='store_true',
    default=False
)

args = parser.parse_args()

batch_size = args.prebatch_size

# # Feature Vector export
bool_feature_names = ['event_weekend',
                      'user_has_already_viewed_doc']

int_feature_names = ['user_views',
                     'ad_views',
                     'doc_views',
                     'doc_event_days_since_published',
                     'doc_event_hour',
                     'doc_ad_days_since_published',
                     ]

float_feature_names = [
    'pop_ad_id',
    'pop_ad_id_conf',
    'pop_ad_id_conf_multipl',
    'pop_document_id',
    'pop_document_id_conf',
    'pop_document_id_conf_multipl',
    'pop_publisher_id',
    'pop_publisher_id_conf',
    'pop_publisher_id_conf_multipl',
    'pop_advertiser_id',
    'pop_advertiser_id_conf',
    'pop_advertiser_id_conf_multipl',
    'pop_campain_id',
    'pop_campain_id_conf',
    'pop_campain_id_conf_multipl',
    'pop_doc_event_doc_ad',
    'pop_doc_event_doc_ad_conf',
    'pop_doc_event_doc_ad_conf_multipl',
    'pop_source_id',
    'pop_source_id_conf',
    'pop_source_id_conf_multipl',
    'pop_source_id_country',
    'pop_source_id_country_conf',
    'pop_source_id_country_conf_multipl',
    'pop_entity_id',
    'pop_entity_id_conf',
    'pop_entity_id_conf_multipl',
    'pop_entity_id_country',
    'pop_entity_id_country_conf',
    'pop_entity_id_country_conf_multipl',
    'pop_topic_id',
    'pop_topic_id_conf',
    'pop_topic_id_conf_multipl',
    'pop_topic_id_country',
    'pop_topic_id_country_conf',
    'pop_topic_id_country_conf_multipl',
    'pop_category_id',
    'pop_category_id_conf',
    'pop_category_id_conf_multipl',
    'pop_category_id_country',
    'pop_category_id_country_conf',
    'pop_category_id_country_conf_multipl',
    'user_doc_ad_sim_categories',
    'user_doc_ad_sim_categories_conf',
    'user_doc_ad_sim_categories_conf_multipl',
    'user_doc_ad_sim_topics',
    'user_doc_ad_sim_topics_conf',
    'user_doc_ad_sim_topics_conf_multipl',
    'user_doc_ad_sim_entities',
    'user_doc_ad_sim_entities_conf',
    'user_doc_ad_sim_entities_conf_multipl',
    'doc_event_doc_ad_sim_categories',
    'doc_event_doc_ad_sim_categories_conf',
    'doc_event_doc_ad_sim_categories_conf_multipl',
    'doc_event_doc_ad_sim_topics',
    'doc_event_doc_ad_sim_topics_conf',
    'doc_event_doc_ad_sim_topics_conf_multipl',
    'doc_event_doc_ad_sim_entities',
    'doc_event_doc_ad_sim_entities_conf',
    'doc_event_doc_ad_sim_entities_conf_multipl'
]

# ### Configuring feature vector
category_feature_names_integral = ['ad_advertiser',
                                   'doc_ad_category_id_1',
                                   'doc_ad_category_id_2',
                                   'doc_ad_category_id_3',
                                   'doc_ad_topic_id_1',
                                   'doc_ad_topic_id_2',
                                   'doc_ad_topic_id_3',
                                   'doc_ad_entity_id_1',
                                   'doc_ad_entity_id_2',
                                   'doc_ad_entity_id_3',
                                   'doc_ad_entity_id_4',
                                   'doc_ad_entity_id_5',
                                   'doc_ad_entity_id_6',
                                   'doc_ad_publisher_id',
                                   'doc_ad_source_id',
                                   'doc_event_category_id_1',
                                   'doc_event_category_id_2',
                                   'doc_event_category_id_3',
                                   'doc_event_topic_id_1',
                                   'doc_event_topic_id_2',
                                   'doc_event_topic_id_3',
                                   'doc_event_entity_id_1',
                                   'doc_event_entity_id_2',
                                   'doc_event_entity_id_3',
                                   'doc_event_entity_id_4',
                                   'doc_event_entity_id_5',
                                   'doc_event_entity_id_6',
                                   'doc_event_publisher_id',
                                   'doc_event_source_id',
                                   'event_country',
                                   'event_country_state',
                                   'event_geo_location',
                                   'event_hour',
                                   'event_platform',
                                   'traffic_source']

feature_vector_labels_integral = bool_feature_names \
                                 + int_feature_names \
                                 + float_feature_names \
                                 + category_feature_names_integral

if args.submission:
    train_feature_vector_gcs_folder_name = 'train_feature_vectors_integral'
else:
    train_feature_vector_gcs_folder_name = 'train_feature_vectors_integral_eval'

# ## Exporting integral feature vectors to CSV
train_feature_vectors_exported_df = spark.read.parquet(OUTPUT_BUCKET_FOLDER + train_feature_vector_gcs_folder_name)
train_feature_vectors_exported_df.take(3)

integral_headers = ['label', 'display_id', 'ad_id', 'doc_id', 'doc_event_id',
                    'is_leak'] + feature_vector_labels_integral

CSV_ORDERED_COLUMNS = ['label', 'display_id', 'ad_id', 'doc_id', 'doc_event_id', 'is_leak', 'event_weekend',
                       'user_has_already_viewed_doc', 'user_views', 'ad_views', 'doc_views',
                       'doc_event_days_since_published', 'doc_event_hour', 'doc_ad_days_since_published',
                       'pop_ad_id', 'pop_ad_id_conf',
                       'pop_ad_id_conf_multipl', 'pop_document_id', 'pop_document_id_conf',
                       'pop_document_id_conf_multipl', 'pop_publisher_id', 'pop_publisher_id_conf',
                       'pop_publisher_id_conf_multipl', 'pop_advertiser_id', 'pop_advertiser_id_conf',
                       'pop_advertiser_id_conf_multipl', 'pop_campain_id', 'pop_campain_id_conf',
                       'pop_campain_id_conf_multipl', 'pop_doc_event_doc_ad', 'pop_doc_event_doc_ad_conf',
                       'pop_doc_event_doc_ad_conf_multipl', 'pop_source_id', 'pop_source_id_conf',
                       'pop_source_id_conf_multipl', 'pop_source_id_country', 'pop_source_id_country_conf',
                       'pop_source_id_country_conf_multipl', 'pop_entity_id', 'pop_entity_id_conf',
                       'pop_entity_id_conf_multipl', 'pop_entity_id_country', 'pop_entity_id_country_conf',
                       'pop_entity_id_country_conf_multipl', 'pop_topic_id', 'pop_topic_id_conf',
                       'pop_topic_id_conf_multipl', 'pop_topic_id_country', 'pop_topic_id_country_conf',
                       'pop_topic_id_country_conf_multipl', 'pop_category_id', 'pop_category_id_conf',
                       'pop_category_id_conf_multipl', 'pop_category_id_country', 'pop_category_id_country_conf',
                       'pop_category_id_country_conf_multipl', 'user_doc_ad_sim_categories',
                       'user_doc_ad_sim_categories_conf', 'user_doc_ad_sim_categories_conf_multipl',
                       'user_doc_ad_sim_topics', 'user_doc_ad_sim_topics_conf', 'user_doc_ad_sim_topics_conf_multipl',
                       'user_doc_ad_sim_entities', 'user_doc_ad_sim_entities_conf',
                       'user_doc_ad_sim_entities_conf_multipl',
                       'doc_event_doc_ad_sim_categories', 'doc_event_doc_ad_sim_categories_conf',
                       'doc_event_doc_ad_sim_categories_conf_multipl', 'doc_event_doc_ad_sim_topics',
                       'doc_event_doc_ad_sim_topics_conf', 'doc_event_doc_ad_sim_topics_conf_multipl',
                       'doc_event_doc_ad_sim_entities', 'doc_event_doc_ad_sim_entities_conf',
                       'doc_event_doc_ad_sim_entities_conf_multipl', 'ad_advertiser', 'doc_ad_category_id_1',
                       'doc_ad_category_id_2', 'doc_ad_category_id_3', 'doc_ad_topic_id_1', 'doc_ad_topic_id_2',
                       'doc_ad_topic_id_3', 'doc_ad_entity_id_1', 'doc_ad_entity_id_2', 'doc_ad_entity_id_3',
                       'doc_ad_entity_id_4', 'doc_ad_entity_id_5', 'doc_ad_entity_id_6', 'doc_ad_publisher_id',
                       'doc_ad_source_id', 'doc_event_category_id_1', 'doc_event_category_id_2',
                       'doc_event_category_id_3',
                       'doc_event_topic_id_1', 'doc_event_topic_id_2', 'doc_event_topic_id_3', 'doc_event_entity_id_1',
                       'doc_event_entity_id_2', 'doc_event_entity_id_3', 'doc_event_entity_id_4',
                       'doc_event_entity_id_5',
                       'doc_event_entity_id_6', 'doc_event_publisher_id', 'doc_event_source_id', 'event_country',
                       'event_country_state', 'event_geo_location', 'event_hour', 'event_platform', 'traffic_source']

FEAT_CSV_ORDERED_COLUMNS = ['event_weekend',
                            'user_has_already_viewed_doc', 'user_views', 'ad_views', 'doc_views',
                            'doc_event_days_since_published', 'doc_event_hour', 'doc_ad_days_since_published',
                            'pop_ad_id', 'pop_ad_id_conf',
                            'pop_ad_id_conf_multipl', 'pop_document_id', 'pop_document_id_conf',
                            'pop_document_id_conf_multipl', 'pop_publisher_id', 'pop_publisher_id_conf',
                            'pop_publisher_id_conf_multipl', 'pop_advertiser_id', 'pop_advertiser_id_conf',
                            'pop_advertiser_id_conf_multipl', 'pop_campain_id', 'pop_campain_id_conf',
                            'pop_campain_id_conf_multipl', 'pop_doc_event_doc_ad', 'pop_doc_event_doc_ad_conf',
                            'pop_doc_event_doc_ad_conf_multipl', 'pop_source_id', 'pop_source_id_conf',
                            'pop_source_id_conf_multipl', 'pop_source_id_country', 'pop_source_id_country_conf',
                            'pop_source_id_country_conf_multipl', 'pop_entity_id', 'pop_entity_id_conf',
                            'pop_entity_id_conf_multipl', 'pop_entity_id_country', 'pop_entity_id_country_conf',
                            'pop_entity_id_country_conf_multipl', 'pop_topic_id', 'pop_topic_id_conf',
                            'pop_topic_id_conf_multipl', 'pop_topic_id_country', 'pop_topic_id_country_conf',
                            'pop_topic_id_country_conf_multipl', 'pop_category_id', 'pop_category_id_conf',
                            'pop_category_id_conf_multipl', 'pop_category_id_country', 'pop_category_id_country_conf',
                            'pop_category_id_country_conf_multipl', 'user_doc_ad_sim_categories',
                            'user_doc_ad_sim_categories_conf', 'user_doc_ad_sim_categories_conf_multipl',
                            'user_doc_ad_sim_topics', 'user_doc_ad_sim_topics_conf',
                            'user_doc_ad_sim_topics_conf_multipl',
                            'user_doc_ad_sim_entities', 'user_doc_ad_sim_entities_conf',
                            'user_doc_ad_sim_entities_conf_multipl',
                            'doc_event_doc_ad_sim_categories', 'doc_event_doc_ad_sim_categories_conf',
                            'doc_event_doc_ad_sim_categories_conf_multipl', 'doc_event_doc_ad_sim_topics',
                            'doc_event_doc_ad_sim_topics_conf', 'doc_event_doc_ad_sim_topics_conf_multipl',
                            'doc_event_doc_ad_sim_entities', 'doc_event_doc_ad_sim_entities_conf',
                            'doc_event_doc_ad_sim_entities_conf_multipl', 'ad_advertiser', 'doc_ad_category_id_1',
                            'doc_ad_category_id_2', 'doc_ad_category_id_3', 'doc_ad_topic_id_1', 'doc_ad_topic_id_2',
                            'doc_ad_topic_id_3', 'doc_ad_entity_id_1', 'doc_ad_entity_id_2', 'doc_ad_entity_id_3',
                            'doc_ad_entity_id_4', 'doc_ad_entity_id_5', 'doc_ad_entity_id_6', 'doc_ad_publisher_id',
                            'doc_ad_source_id', 'doc_event_category_id_1', 'doc_event_category_id_2',
                            'doc_event_category_id_3',
                            'doc_event_topic_id_1', 'doc_event_topic_id_2', 'doc_event_topic_id_3',
                            'doc_event_entity_id_1',
                            'doc_event_entity_id_2', 'doc_event_entity_id_3', 'doc_event_entity_id_4',
                            'doc_event_entity_id_5',
                            'doc_event_entity_id_6', 'doc_event_publisher_id', 'doc_event_source_id', 'event_country',
                            'event_country_state', 'event_geo_location', 'event_hour', 'event_platform',
                            'traffic_source']


def to_array(col):
    def to_array_(v):
        return v.toArray().tolist()

    # Important: asNondeterministic requires Spark 2.3 or later
    # It can be safely removed i.e.
    # return udf(to_array_, ArrayType(DoubleType()))(col)
    # but at the cost of decreased performance

    return udf(to_array_, ArrayType(DoubleType())).asNondeterministic()(col)


CONVERT_TO_INT = ['doc_ad_category_id_1',
                  'doc_ad_category_id_2', 'doc_ad_category_id_3', 'doc_ad_topic_id_1', 'doc_ad_topic_id_2',
                  'doc_ad_topic_id_3', 'doc_ad_entity_id_1', 'doc_ad_entity_id_2', 'doc_ad_entity_id_3',
                  'doc_ad_entity_id_4', 'doc_ad_entity_id_5', 'doc_ad_entity_id_6',
                  'doc_ad_source_id', 'doc_event_category_id_1', 'doc_event_category_id_2', 'doc_event_category_id_3',
                  'doc_event_topic_id_1', 'doc_event_topic_id_2', 'doc_event_topic_id_3', 'doc_event_entity_id_1',
                  'doc_event_entity_id_2', 'doc_event_entity_id_3', 'doc_event_entity_id_4', 'doc_event_entity_id_5',
                  'doc_event_entity_id_6']


def format_number(element, name):
    if name in BOOL_COLUMNS + CATEGORICAL_COLUMNS:
        return element.cast("int")
    elif name in CONVERT_TO_INT:
        return element.cast("int")
    else:
        return element


def to_array_with_none(col):
    def to_array_with_none_(v):
        tmp = np.full((v.size,), fill_value=None, dtype=np.float64)
        tmp[v.indices] = v.values
        return tmp.tolist()

    # Important: asNondeterministic requires Spark 2.3 or later
    # It can be safely removed i.e.
    # return udf(to_array_, ArrayType(DoubleType()))(col)
    # but at the cost of decreased performance

    return udf(to_array_with_none_, ArrayType(DoubleType())).asNondeterministic()(col)


@udf
def count_value(x):
    from collections import Counter
    tmp = Counter(x).most_common(2)
    if not tmp or np.isnan(tmp[0][0]):
        return 0
    return float(tmp[0][0])


def replace_with_most_frequent(most_value):
    return udf(lambda x: most_value if not x or np.isnan(x) else x)


train_feature_vectors_integral_csv_rdd_df = train_feature_vectors_exported_df.select('label', 'display_id', 'ad_id',
                                                                                     'document_id', 'document_id_event',
                                                                                     'feature_vector').withColumn(
    'is_leak', F.lit(-1)).withColumn("featvec", to_array("feature_vector")).select(
    ['label'] + ['display_id'] + ['ad_id'] + ['document_id'] + ['document_id_event'] + ['is_leak'] + [
        format_number(element, FEAT_CSV_ORDERED_COLUMNS[index]).alias(FEAT_CSV_ORDERED_COLUMNS[index]) for
        index, element in enumerate([col("featvec")[i] for i in range(len(feature_vector_labels_integral))])]).replace(
    float('nan'), 0)

if args.submission:
    test_validation_feature_vector_gcs_folder_name = 'test_feature_vectors_integral'
else:
    test_validation_feature_vector_gcs_folder_name = 'validation_feature_vectors_integral'

# ## Exporting integral feature vectors
test_validation_feature_vectors_exported_df = spark.read.parquet(
    OUTPUT_BUCKET_FOLDER + test_validation_feature_vector_gcs_folder_name)
test_validation_feature_vectors_exported_df.take(3)

test_validation_feature_vectors_integral_csv_rdd_df = test_validation_feature_vectors_exported_df.select(
    'label', 'display_id', 'ad_id', 'document_id', 'document_id_event',
    'is_leak', 'feature_vector').withColumn("featvec", to_array("feature_vector")).select(
    ['label'] + ['display_id'] + ['ad_id'] + ['document_id'] + ['document_id_event'] + ['is_leak'] + [
        format_number(element, FEAT_CSV_ORDERED_COLUMNS[index]).alias(FEAT_CSV_ORDERED_COLUMNS[index]) for
        index, element in enumerate([col("featvec")[i] for i in range(len(feature_vector_labels_integral))])]).replace(
    float('nan'), 0)


def make_spec(output_dir, batch_size=None):
    fixed_shape = [batch_size, 1] if batch_size is not None else []
    spec = {}
    spec[LABEL_COLUMN] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
    spec[DISPLAY_ID_COLUMN] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
    spec[IS_LEAK_COLUMN] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
    spec[DISPLAY_ID_AND_IS_LEAK_ENCODED_COLUMN] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.int64,
                                                                     default_value=None)
    for name in BOOL_COLUMNS:
        spec[name] = tf.FixedLenFeature(shape=fixed_shape, dtype=tf.int64, default_value=None)
    for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM + FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM:
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
        shape = fixed_shape[:-1] + [len(DOC_CATEGORICAL_MULTIVALUED_COLUMNS[multi_category])]
        spec[multi_category] = tf.FixedLenFeature(shape=shape, dtype=tf.int64)
    metadata = dataset_metadata.DatasetMetadata(dataset_schema.from_feature_spec(spec))
    metadata_io.write_metadata(metadata, output_dir)


# write out tfrecords meta
make_spec(LOCAL_DATA_TFRECORDS_DIR + '/transformed_metadata', batch_size=batch_size)


def log2_1p(x):
    return np.log1p(x) / np.log(2.0)


# calculate min and max stats for the given dataframes all in one go
def compute_min_max_logs(df):
    print(str(datetime.datetime.now()) + '\tComputing min and max')
    min_logs = {}
    max_logs = {}
    float_expr = []
    for name in trainer.features.FLOAT_COLUMNS_LOG_BIN_TRANSFORM + trainer.features.INT_COLUMNS:
        float_expr.append(F.min(name))
        float_expr.append(F.max(name))
    floatDf = all_df.agg(*float_expr).collect()
    for name in trainer.features.FLOAT_COLUMNS_LOG_BIN_TRANSFORM:
        minAgg = floatDf[0]["min(" + name + ")"]
        maxAgg = floatDf[0]["max(" + name + ")"]
        min_logs[name + '_log_01scaled'] = log2_1p(minAgg * 1000)
        max_logs[name + '_log_01scaled'] = log2_1p(maxAgg * 1000)
    for name in trainer.features.INT_COLUMNS:
        minAgg = floatDf[0]["min(" + name + ")"]
        maxAgg = floatDf[0]["max(" + name + ")"]
        min_logs[name + '_log_01scaled'] = log2_1p(minAgg)
        max_logs[name + '_log_01scaled'] = log2_1p(maxAgg)

    return min_logs, max_logs


all_df = test_validation_feature_vectors_integral_csv_rdd_df.union(train_feature_vectors_integral_csv_rdd_df)
min_logs, max_logs = compute_min_max_logs(all_df)

if args.submission:
    train_output_string = '/sub_train'
    eval_output_string = '/test'
else:
    train_output_string = '/train'
    eval_output_string = '/eval'

path = LOCAL_DATA_TFRECORDS_DIR


def create_tf_example_spark(df, min_logs, max_logs):
    result = {}
    result[LABEL_COLUMN] = tf.train.Feature(int64_list=tf.train.Int64List(value=df[LABEL_COLUMN].to_list()))
    result[DISPLAY_ID_COLUMN] = tf.train.Feature(int64_list=tf.train.Int64List(value=df[DISPLAY_ID_COLUMN].to_list()))
    result[IS_LEAK_COLUMN] = tf.train.Feature(int64_list=tf.train.Int64List(value=df[IS_LEAK_COLUMN].to_list()))
    encoded_value = df[DISPLAY_ID_COLUMN].multiply(10).add(df[IS_LEAK_COLUMN].clip(lower=0)).to_list()
    result[DISPLAY_ID_AND_IS_LEAK_ENCODED_COLUMN] = tf.train.Feature(int64_list=tf.train.Int64List(value=encoded_value))
    for name in FLOAT_COLUMNS:
        value = df[name].to_list()
        result[name] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
    for name in FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM:
        value = df[name].multiply(10).astype('int64').to_list()
        result[name + '_binned'] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM:
        value_prelim = df[name].multiply(1000).apply(np.log1p).multiply(1. / np.log(2.0))
        value = value_prelim.astype('int64').to_list()
        result[name + '_binned'] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        nn = name + '_log_01scaled'
        value = value_prelim.add(-min_logs[nn]).multiply(1. / (max_logs[nn] - min_logs[nn])).to_list()
        result[nn] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
    for name in INT_COLUMNS:
        value_prelim = df[name].apply(np.log1p).multiply(1. / np.log(2.0))
        value = value_prelim.astype('int64').to_list()
        result[name + '_log_int'] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        nn = name + '_log_01scaled'
        value = value_prelim.add(-min_logs[nn]).multiply(1. / (max_logs[nn] - min_logs[nn])).to_list()
        result[nn] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
    for name in BOOL_COLUMNS + CATEGORICAL_COLUMNS:
        value = df[name].fillna(0).astype('int64').to_list()
        result[name] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
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


def _transform_to_tfrecords(rdds):
    csv = pd.DataFrame(list(rdds), columns=CSV_ORDERED_COLUMNS)
    num_rows = len(csv.index)
    examples = []
    for start_ind in range(0, num_rows, batch_size if batch_size is not None else 1):  # for each batch
        if start_ind + batch_size - 1 > num_rows:  # if we'd run out of rows
            csv_slice = csv.iloc[start_ind:]
            # drop the remainder
            print("last Example has: ", len(csv_slice))
            examples.append((create_tf_example_spark(csv_slice, min_logs, max_logs), len(csv_slice)))
            return examples
        else:
            csv_slice = csv.iloc[start_ind:start_ind + (batch_size if batch_size is not None else 1)]
        examples.append((create_tf_example_spark(csv_slice, min_logs, max_logs), batch_size))
    return examples


max_partition_num = 30


def _transform_to_slices(rdds):
    taskcontext = TaskContext.get()
    partitionid = taskcontext.partitionId()
    csv = pd.DataFrame(list(rdds), columns=CSV_ORDERED_COLUMNS)
    num_rows = len(csv.index)
    print("working with partition: ", partitionid, max_partition_num, num_rows)
    examples = []
    for start_ind in range(0, num_rows, batch_size if batch_size is not None else 1):  # for each batch
        if start_ind + batch_size - 1 > num_rows:  # if we'd run out of rows
            csv_slice = csv.iloc[start_ind:]
            print("last Example has: ", len(csv_slice), partitionid)
            examples.append((csv_slice, len(csv_slice)))
            return examples
        else:
            csv_slice = csv.iloc[start_ind:start_ind + (batch_size if batch_size is not None else 1)]
        examples.append((csv_slice, len(csv_slice)))
    return examples


def _transform_to_tfrecords_from_slices(rdds):
    examples = []
    for slice in rdds:
        if len(slice[0]) != batch_size:
            print("slice size is not correct, dropping: ", len(slice[0]))
        else:
            examples.append(
                (bytearray((create_tf_example_spark(slice[0], min_logs, max_logs)).SerializeToString()), None))
    return examples


def _transform_to_tfrecords_from_reslice(rdds):
    examples = []
    all_dataframes = pd.DataFrame([])
    for slice in rdds:
        all_dataframes = all_dataframes.append(slice[0])
    num_rows = len(all_dataframes.index)
    examples = []
    for start_ind in range(0, num_rows, batch_size if batch_size is not None else 1):  # for each batch
        if start_ind + batch_size - 1 > num_rows:  # if we'd run out of rows
            csv_slice = all_dataframes.iloc[start_ind:]
            if TEST_SET_MODE:
                remain_len = batch_size - len(csv_slice)
                (m, n) = divmod(remain_len, len(csv_slice))
                print("remainder: ", len(csv_slice), remain_len, m, n)
                if m:
                    for i in range(m):
                        csv_slice = csv_slice.append(csv_slice)
                csv_slice = csv_slice.append(csv_slice.iloc[:n])
                print("after fill remainder: ", len(csv_slice))
                examples.append(
                    (bytearray((create_tf_example_spark(csv_slice, min_logs, max_logs)).SerializeToString()), None))
                return examples
            # drop the remainder
            print("dropping remainder: ", len(csv_slice))
            return examples
        else:
            csv_slice = all_dataframes.iloc[start_ind:start_ind + (batch_size if batch_size is not None else 1)]
            examples.append(
                (bytearray((create_tf_example_spark(csv_slice, min_logs, max_logs)).SerializeToString()), None))
    return examples


TEST_SET_MODE = False
train_features = train_feature_vectors_integral_csv_rdd_df.coalesce(30).rdd.mapPartitions(_transform_to_slices)
cached_train_features = train_features.cache()
cached_train_features.count()
train_full = cached_train_features.filter(lambda x: x[1] == batch_size)
# split out slies where we don't have a full batch so that we can reslice them so we only drop mininal rows
train_not_full = cached_train_features.filter(lambda x: x[1] < batch_size)
train_examples_full = train_full.mapPartitions(_transform_to_tfrecords_from_slices)
train_left = train_not_full.coalesce(1).mapPartitions(_transform_to_tfrecords_from_reslice)
all_train = train_examples_full.union(train_left)

TEST_SET_MODE = True
valid_features = test_validation_feature_vectors_integral_csv_rdd_df.coalesce(30).rdd.mapPartitions(
    _transform_to_slices)
cached_valid_features = valid_features.cache()
cached_valid_features.count()
valid_full = cached_valid_features.filter(lambda x: x[1] == batch_size)
valid_not_full = cached_valid_features.filter(lambda x: x[1] < batch_size)
valid_examples_full = valid_full.mapPartitions(_transform_to_tfrecords_from_slices)
valid_left = valid_not_full.coalesce(1).mapPartitions(_transform_to_tfrecords_from_reslice)
all_valid = valid_examples_full.union(valid_left)

all_train.saveAsNewAPIHadoopFile(LOCAL_DATA_TFRECORDS_DIR + train_output_string,
                                 "org.tensorflow.hadoop.io.TFRecordFileOutputFormat",
                                 keyClass="org.apache.hadoop.io.BytesWritable",
                                 valueClass="org.apache.hadoop.io.NullWritable")

all_valid.saveAsNewAPIHadoopFile(LOCAL_DATA_TFRECORDS_DIR + eval_output_string,
                                 "org.tensorflow.hadoop.io.TFRecordFileOutputFormat",
                                 keyClass="org.apache.hadoop.io.BytesWritable",
                                 valueClass="org.apache.hadoop.io.NullWritable")

spark.stop()
