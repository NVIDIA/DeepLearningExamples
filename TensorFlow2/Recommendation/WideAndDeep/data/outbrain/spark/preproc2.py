#!/usr/bin/env python
# coding: utf-8

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


import datetime
import hashlib
import math
import time

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.context import SparkContext, SparkConf
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructType, StructField, TimestampType, FloatType, ArrayType

OUTPUT_BUCKET_FOLDER = "/tmp/spark/preprocessed/"
DATA_BUCKET_FOLDER = "/outbrain/orig/"
SPARK_TEMP_FOLDER = "/tmp/spark/spark-temp/"

conf = SparkConf().setMaster('local[*]').set('spark.executor.memory', '40g').set('spark.driver.memory', '200g').set(
    "spark.local.dir", SPARK_TEMP_FOLDER)

sc = SparkContext(conf=conf)
spark = SparkSession(sc)

start_time = time.time()


def hashstr(s, nr_bins):
    return int(hashlib.md5(s.encode('utf8')).hexdigest(), 16) % (nr_bins - 1) + 1





# ## UDFs
def date_time_to_unix_epoch(date_time):
    return int(time.mktime(date_time.timetuple()))


def date_time_to_unix_epoch_treated(dt):
    if dt is not None:
        try:
            epoch = date_time_to_unix_epoch(dt)
            return epoch
        except Exception as e:
            print("Error processing dt={}".format(dt), e)
            return 0
    else:
        return 0


timestamp_null_to_zero_int_udf = F.udf(lambda x: date_time_to_unix_epoch_treated(x), IntegerType())

INT_DEFAULT_NULL_VALUE = -1
int_null_to_minus_one_udf = F.udf(lambda x: x if x is not None else INT_DEFAULT_NULL_VALUE, IntegerType())
int_list_null_to_empty_list_udf = F.udf(lambda x: x if x is not None else [], ArrayType(IntegerType()))
float_list_null_to_empty_list_udf = F.udf(lambda x: x if x is not None else [], ArrayType(FloatType()))
str_list_null_to_empty_list_udf = F.udf(lambda x: x if x is not None else [], ArrayType(StringType()))


def truncate_day_from_timestamp(ts):
    return int(ts / 1000 / 60 / 60 / 24)


truncate_day_from_timestamp_udf = F.udf(lambda ts: truncate_day_from_timestamp(ts), IntegerType())

extract_country_udf = F.udf(lambda geo: geo.strip()[:2] if geo is not None else '', StringType())

extract_country_state_udf = F.udf(lambda geo: geo.strip()[:5] if geo is not None else '', StringType())

list_len_udf = F.udf(lambda x: len(x) if x is not None else 0, IntegerType())


def convert_odd_timestamp(timestamp_ms_relative):
    TIMESTAMP_DELTA = 1465876799998
    return datetime.datetime.fromtimestamp((int(timestamp_ms_relative) + TIMESTAMP_DELTA) // 1000)


# # Loading Files

# ## Loading UTC/BST for each country and US / CA states (local time)
country_utc_dst_df = pd.read_csv('data/outbrain/spark/data/country_codes_utc_dst_tz_delta.csv', keep_default_na=False)

countries_utc_dst_dict = dict(
    zip(country_utc_dst_df['country_code'].tolist(), country_utc_dst_df['utc_dst_time_offset_cleaned'].tolist()))
countries_utc_dst_broad = sc.broadcast(countries_utc_dst_dict)

us_states_utc_dst_df = pd.read_csv('data/outbrain/spark/data/us_states_abbrev_bst.csv', keep_default_na=False)

us_states_utc_dst_dict = dict(
    zip(us_states_utc_dst_df['state_abb'].tolist(), us_states_utc_dst_df['utc_dst_time_offset_cleaned'].tolist()))
us_states_utc_dst_broad = sc.broadcast(us_states_utc_dst_dict)

ca_states_utc_dst_df = pd.read_csv('data/outbrain/spark/data/ca_states_abbrev_bst.csv', keep_default_na=False)

ca_countries_utc_dst_dict = dict(
    zip(ca_states_utc_dst_df['state_abb'].tolist(), ca_states_utc_dst_df['utc_dst_time_offset_cleaned'].tolist()))
ca_countries_utc_dst_broad = sc.broadcast(ca_countries_utc_dst_dict)

# ## Loading competition csvs
events_schema = StructType(
    [StructField("display_id", IntegerType(), True),
     StructField("uuid_event", StringType(), True),
     StructField("document_id_event", IntegerType(), True),
     StructField("timestamp_event", IntegerType(), True),
     StructField("platform_event", IntegerType(), True),
     StructField("geo_location_event", StringType(), True)]
)

events_df = spark.read.schema(events_schema) \
    .options(header='true', inferschema='false', nullValue='\\N') \
    .csv(DATA_BUCKET_FOLDER + "events.csv") \
    .withColumn('dummyEvents', F.lit(1)) \
    .withColumn('day_event', truncate_day_from_timestamp_udf('timestamp_event')) \
    .withColumn('event_country', extract_country_udf('geo_location_event')) \
    .withColumn('event_country_state', extract_country_state_udf('geo_location_event')) \
    .alias('events')

events_df.count()

# Drop rows with empty "geo_location"
events_df = events_df.dropna(subset="geo_location_event")
events_df.count()

# Drop rows with empty "platform"
events_df = events_df.dropna(subset="platform_event")
events_df.count()

promoted_content_schema = StructType(
    [StructField("ad_id", IntegerType(), True),
     StructField("document_id_promo", IntegerType(), True),
     StructField("campaign_id", IntegerType(), True),
     StructField("advertiser_id", IntegerType(), True)]
)

promoted_content_df = spark.read.schema(promoted_content_schema) \
    .options(header='true', inferschema='false', nullValue='\\N') \
    .csv(DATA_BUCKET_FOLDER + "promoted_content.csv") \
    .withColumn('dummyPromotedContent', F.lit(1)).alias('promoted_content').cache()

documents_meta_schema = StructType(
    [StructField("document_id_doc", IntegerType(), True),
     StructField("source_id", IntegerType(), True),
     StructField("publisher_id", IntegerType(), True),
     StructField("publish_time", TimestampType(), True)]
)

documents_meta_df = spark.read.schema(documents_meta_schema) \
    .options(header='true', inferschema='false', nullValue='\\N') \
    .csv(DATA_BUCKET_FOLDER + "documents_meta.csv") \
    .withColumn('dummyDocumentsMeta', F.lit(1)).alias('documents_meta').cache()

documents_meta_df.count()

# Drop rows with empty "source_id"
documents_meta_df = documents_meta_df.dropna(subset="source_id")
documents_meta_df.count()

source_publishers_df = documents_meta_df.select(["source_id", "publisher_id"]).dropDuplicates()
source_publishers_df.count()

# get list of source_ids without publisher_id
rows_no_pub = source_publishers_df.filter("publisher_id is NULL")
source_ids_without_publisher = [row['source_id'] for row in rows_no_pub.collect()]
len(source_ids_without_publisher)

# maximum value of publisher_id used so far
max_pub = max(source_publishers_df.select(["publisher_id"]).dropna().collect())['publisher_id']
max_pub

# rows filled with new publisher_ids
new_publishers = [(source, max_pub + 1 + nr) for nr, source in enumerate(source_ids_without_publisher)]
new_publishers_df = spark.createDataFrame(new_publishers, ("source_id", "publisher_id"))
new_publishers_df.take(10)

# old and new publishers merged
fixed_source_publishers_df = source_publishers_df.dropna().union(new_publishers_df)
fixed_source_publishers_df.collect()[-30:]

# update documents_meta with bew publishers
documents_meta_df = documents_meta_df.drop('publisher_id').join(fixed_source_publishers_df, on='source_id')
documents_meta_df.count()

events_joined_df = events_df.join(documents_meta_df
                                  .withColumnRenamed('source_id', 'source_id_doc_event')
                                  .withColumnRenamed('publisher_id', 'publisher_doc_event')
                                  .withColumnRenamed('publish_time', 'publish_time_doc_event'),
                                  on=F.col("document_id_event") == F.col("document_id_doc"), how='left').alias('events').cache()


documents_categories_schema = StructType(
    [StructField("document_id_cat", IntegerType(), True),
     StructField("category_id", IntegerType(), True),
     StructField("confidence_level_cat", FloatType(), True)]
)

documents_categories_df = spark.read.schema(documents_categories_schema) \
    .options(header='true', inferschema='false', nullValue='\\N') \
    .csv(DATA_BUCKET_FOLDER + "documents_categories.csv") \
    .alias('documents_categories').cache()

documents_categories_grouped_df = documents_categories_df.groupBy('document_id_cat') \
    .agg(F.collect_list('category_id').alias('category_id_list'),
         F.collect_list('confidence_level_cat').alias('confidence_level_cat_list')) \
    .withColumn('dummyDocumentsCategory', F.lit(1)) \
    .alias('documents_categories_grouped')

documents_topics_schema = StructType(
    [StructField("document_id_top", IntegerType(), True),
     StructField("topic_id", IntegerType(), True),
     StructField("confidence_level_top", FloatType(), True)]
)

documents_topics_df = spark.read.schema(documents_topics_schema) \
    .options(header='true', inferschema='false', nullValue='\\N') \
    .csv(DATA_BUCKET_FOLDER + "documents_topics.csv") \
    .alias('documents_topics').cache()

documents_topics_grouped_df = documents_topics_df.groupBy('document_id_top') \
    .agg(F.collect_list('topic_id').alias('topic_id_list'),
         F.collect_list('confidence_level_top').alias('confidence_level_top_list')) \
    .withColumn('dummyDocumentsTopics', F.lit(1)) \
    .alias('documents_topics_grouped')

documents_entities_schema = StructType(
    [StructField("document_id_ent", IntegerType(), True),
     StructField("entity_id", StringType(), True),
     StructField("confidence_level_ent", FloatType(), True)]
)

documents_entities_df = spark.read.schema(documents_entities_schema) \
    .options(header='true', inferschema='false', nullValue='\\N') \
    .csv(DATA_BUCKET_FOLDER + "documents_entities.csv") \
    .alias('documents_entities').cache()

documents_entities_grouped_df = documents_entities_df.groupBy('document_id_ent') \
    .agg(F.collect_list('entity_id').alias('entity_id_list'),
         F.collect_list('confidence_level_ent').alias('confidence_level_ent_list')) \
    .withColumn('dummyDocumentsEntities', F.lit(1)) \
    .alias('documents_entities_grouped')

clicks_train_schema = StructType(
    [StructField("display_id", IntegerType(), True),
     StructField("ad_id", IntegerType(), True),
     StructField("clicked", IntegerType(), True)]
)

clicks_train_df = spark.read.schema(clicks_train_schema) \
    .options(header='true', inferschema='false', nullValue='\\N') \
    .csv(DATA_BUCKET_FOLDER + "clicks_train.csv") \
    .withColumn('dummyClicksTrain', F.lit(1)).alias('clicks_train')

clicks_train_joined_df = clicks_train_df \
    .join(promoted_content_df, on='ad_id', how='left') \
    .join(documents_meta_df,
          on=F.col("promoted_content.document_id_promo") == F.col("documents_meta.document_id_doc"),
          how='left') \
    .join(events_joined_df, on='display_id', how='left')
clicks_train_joined_df.createOrReplaceTempView('clicks_train_joined')



# # Spliting Train/validation set | Test set

validation_set_exported_df = spark.read.parquet(
    OUTPUT_BUCKET_FOLDER + "validation_set.parquet") \
    .alias('validation_set')

validation_set_exported_df.select('display_id').distinct() \
    .createOrReplaceTempView("validation_display_ids")

validation_set_df = spark.sql('''
  SELECT * FROM clicks_train_joined t
  WHERE EXISTS (SELECT display_id FROM validation_display_ids
  WHERE display_id = t.display_id)''').alias('clicks') \
    .join(documents_categories_grouped_df,
          on=F.col("document_id_promo") == F.col("documents_categories_grouped.document_id_cat"),
          how='left') \
    .join(documents_topics_grouped_df,
          on=F.col("document_id_promo") == F.col("documents_topics_grouped.document_id_top"),
          how='left') \
    .join(documents_entities_grouped_df,
          on=F.col("document_id_promo") == F.col("documents_entities_grouped.document_id_ent"),
          how='left') \
    .join(documents_categories_grouped_df
          .withColumnRenamed('category_id_list', 'doc_event_category_id_list')
          .withColumnRenamed('confidence_level_cat_list', 'doc_event_confidence_level_cat_list')
          .alias('documents_event_categories_grouped'),
          on=F.col("document_id_event") == F.col("documents_event_categories_grouped.document_id_cat"),
          how='left') \
    .join(documents_topics_grouped_df
          .withColumnRenamed('topic_id_list', 'doc_event_topic_id_list')
          .withColumnRenamed('confidence_level_top_list', 'doc_event_confidence_level_top_list')
          .alias('documents_event_topics_grouped'),
          on=F.col("document_id_event") == F.col("documents_event_topics_grouped.document_id_top"),
          how='left') \
    .join(documents_entities_grouped_df
          .withColumnRenamed('entity_id_list', 'doc_event_entity_id_list')
          .withColumnRenamed('confidence_level_ent_list', 'doc_event_confidence_level_ent_list')
          .alias('documents_event_entities_grouped'),
          on=F.col("document_id_event") == F.col("documents_event_entities_grouped.document_id_ent"),
          how='left') \

# print("validation_set_df.count() =", validation_set_df.count())

# Added to validation set information about the event and the user for statistics of the error (avg ctr)
validation_set_ground_truth_df = validation_set_df.filter('clicked = 1') \
    .select('display_id', 'ad_id', 'platform_event', 'day_event', 'timestamp_event',
            'geo_location_event', 'event_country', 'event_country_state') \
    .withColumnRenamed('ad_id', 'ad_id_gt') \
    .cache()
# print("validation_set_ground_truth_df.count() =", validation_set_ground_truth_df.count())

train_set_df = spark.sql('''
SELECT * FROM clicks_train_joined t
WHERE NOT EXISTS (SELECT display_id FROM validation_display_ids
WHERE display_id = t.display_id)''').cache()
print("train_set_df.count() =", train_set_df.count())


# # Training models


def is_null(value):
    return value is None or len(str(value).strip()) == 0


LESS_SPECIAL_CAT_VALUE = 'less'


def get_category_field_values_counts(field, df, min_threshold=10):
    category_counts = dict(list(filter(lambda x: not is_null(x[0]) and x[1] >= min_threshold,
                                       df.select(field).groupBy(field).count().rdd.map(
                                           lambda x: (x[0], x[1])).collect())))
    # Adding a special value to create a feature for values in this category that are less than min_threshold
    category_counts[LESS_SPECIAL_CAT_VALUE] = -1
    return category_counts


# ## Building category values counters and indexers
event_country_values_counts = get_category_field_values_counts('event_country', events_df, min_threshold=10)
len(event_country_values_counts)
# All non-null categories: 230

event_country_state_values_counts = get_category_field_values_counts('event_country_state', events_df, min_threshold=10)
len(event_country_state_values_counts)

event_geo_location_values_counts = get_category_field_values_counts('geo_location_event', events_df, min_threshold=10)
len(event_geo_location_values_counts)
# All non-null categories: 2988

doc_entity_id_values_counts = get_category_field_values_counts('entity_id', documents_entities_df, min_threshold=10)
len(doc_entity_id_values_counts)


# All non-null categories: 1326009


# ## Processing average CTR by categories
def get_percentiles(df, field, quantiles_levels=None, max_error_rate=0.0):
    if quantiles_levels is None:
        quantiles_levels = np.arange(0.0, 1.1, 0.1).tolist()
    quantiles = df.approxQuantile(field, quantiles_levels, max_error_rate)
    return dict(zip(quantiles_levels, quantiles))


# REG = 10
REG = 0
ctr_udf = F.udf(lambda clicks, views: clicks / float(views + REG), FloatType())

# ### Average CTR by ad_id
ad_id_popularity_df = train_set_df.groupby('ad_id').agg(F.sum('clicked').alias('clicks'),
                                                        F.count('*').alias('views')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))

# ad_id_popularity_df.count()

# get_percentiles(ad_id_popularity_df, 'clicks')

# get_percentiles(ad_id_popularity_df, 'views')

ad_id_popularity = ad_id_popularity_df.filter('views > 5').select('ad_id', 'ctr', 'views') \
    .rdd.map(lambda x: (x['ad_id'], (x['ctr'], x['views'], 1, 1))).collectAsMap()

ad_id_popularity_broad = sc.broadcast(ad_id_popularity)

list(ad_id_popularity.values())[:3]

len(ad_id_popularity)

# get_ad_id_ctr_udf = F.udf(lambda ad_id: ad_id_popularity[ad_id] if ad_id in ad_id_popularity else -1, FloatType())

ad_id_avg_ctr = sum(map(lambda x: x[0], ad_id_popularity.values())) / float(len(ad_id_popularity))
ad_id_avg_ctr

ad_id_weighted_avg_ctr = sum(map(lambda x: x[0] * x[1], ad_id_popularity.values())) / float(
    sum(map(lambda x: x[1], ad_id_popularity.values())))
ad_id_weighted_avg_ctr

ad_id_views_median = np.median(np.array(list(map(lambda x: x[1], ad_id_popularity.values()))))
ad_id_views_median

ad_id_views_mean = sum(map(lambda x: x[1], ad_id_popularity.values())) / float(len(ad_id_popularity))
ad_id_views_mean

# ### Average CTR by document_id (promoted_content)
document_id_popularity_df = train_set_df \
    .groupby('document_id_promo') \
    .agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views'),
         F.countDistinct('ad_id').alias('distinct_ad_ids')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))

document_id_popularity = document_id_popularity_df.filter('views > 5') \
    .select('document_id_promo', 'ctr', 'views', 'distinct_ad_ids') \
    .rdd.map(lambda x: (x['document_id_promo'],
                        (x['ctr'], x['views'], x['distinct_ad_ids'], 1))).collectAsMap()
len(document_id_popularity)

document_id_popularity_broad = sc.broadcast(document_id_popularity)

# document_id_popularity_df.count()

# get_percentiles(document_id_popularity_df, 'clicks')

# get_percentiles(document_id_popularity_df, 'views')

document_id_avg_ctr = sum(map(lambda x: x[0], document_id_popularity.values())) / float(len(document_id_popularity))
document_id_avg_ctr

document_id_weighted_avg_ctr = sum(list(map(lambda x: x[0] * x[1], document_id_popularity.values()))) / float(
    sum(list(map(lambda x: x[1], document_id_popularity.values()))))
document_id_weighted_avg_ctr

document_id_views_median = np.median(np.array(list(map(lambda x: x[1], document_id_popularity.values()))))
document_id_views_median

document_id_views_mean = sum(map(lambda x: x[1], document_id_popularity.values())) / float(len(document_id_popularity))
document_id_views_mean

# ### Average CTR by (doc_event, doc_ad)
doc_event_doc_ad_avg_ctr_df = train_set_df.groupBy('document_id_event', 'document_id_promo') \
    .agg(F.sum('clicked').alias('clicks'),
         F.count('*').alias('views'), F.countDistinct('ad_id').alias('distinct_ad_ids')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))

doc_event_doc_ad_avg_ctr = doc_event_doc_ad_avg_ctr_df.filter('views > 5') \
    .select('document_id_event', 'document_id_promo', 'ctr', 'views', 'distinct_ad_ids') \
    .rdd.map(lambda x: ((x['document_id_event'], x['document_id_promo']),
                        (x['ctr'], x['views'], x['distinct_ad_ids'], 1))).collectAsMap()

len(doc_event_doc_ad_avg_ctr)

doc_event_doc_ad_avg_ctr_broad = sc.broadcast(doc_event_doc_ad_avg_ctr)


# ### Average CTR by source_id
source_id_popularity_df = train_set_df.select('clicked', 'source_id', 'ad_id') \
    .groupby('source_id').agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views'),
                              F.countDistinct('ad_id').alias('distinct_ad_ids')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))

source_id_popularity = source_id_popularity_df \
    .filter('views > 10 and source_id is not null') \
    .select('source_id', 'ctr', 'views', 'distinct_ad_ids') \
    .rdd.map(lambda x: (x['source_id'], (x['ctr'], x['views'], x['distinct_ad_ids'], 1))) \
    .collectAsMap()
len(source_id_popularity)

source_id_popularity_broad = sc.broadcast(source_id_popularity)

# source_id_popularity_df.count()

# get_percentiles(source_id_popularity_df, 'clicks')

# get_percentiles(source_id_popularity_df, 'views')

# source_id_popularity = source_id_popularity_df
# .filter('views > 100 and source_id is not null')
# .select('source_id', 'ctr')
# .rdd.collectAsMap()


# ### Average CTR by publisher_id
publisher_popularity_df = train_set_df.select('clicked', 'publisher_id', 'ad_id') \
    .groupby('publisher_id').agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views'),
                                 F.countDistinct('ad_id').alias('distinct_ad_ids')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))

publisher_popularity = publisher_popularity_df \
    .filter('views > 10 and publisher_id is not null') \
    .select('publisher_id', 'ctr', 'views', 'distinct_ad_ids') \
    .rdd.map(lambda x: (x['publisher_id'], (x['ctr'], x['views'], x['distinct_ad_ids'], 1))) \
    .collectAsMap()
len(publisher_popularity)

publisher_popularity_broad = sc.broadcast(publisher_popularity)

# publisher_popularity_df.count()
# ##863

# get_percentiles(publisher_popularity_df, 'clicks')

# get_percentiles(publisher_popularity_df, 'views')

# publisher_id_popularity = publisher_popularity_df
# .filter('views > 100 and publisher_id is not null')
# .select('publisher_id', 'ctr')
# .rdd.collectAsMap()
# len(publisher_id_popularity)
# ##639


# ### Average CTR by advertiser_id
advertiser_id_popularity_df = train_set_df.select('clicked', 'advertiser_id', 'ad_id') \
    .groupby('advertiser_id').agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views'),
                                  F.countDistinct('ad_id').alias('distinct_ad_ids')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))

advertiser_id_popularity = advertiser_id_popularity_df \
    .filter('views > 10 and advertiser_id is not null') \
    .select('advertiser_id', 'ctr', 'views', 'distinct_ad_ids') \
    .rdd.map(lambda x: (x['advertiser_id'],
                        (x['ctr'], x['views'], x['distinct_ad_ids'], 1))).collectAsMap()
len(advertiser_id_popularity)

advertiser_id_popularity_broad = sc.broadcast(advertiser_id_popularity)

# advertiser_id_popularity_df.count()
# ##4063

# get_percentiles(advertiser_id_popularity_df, 'clicks')

# get_percentiles(advertiser_id_popularity_df, 'views')

# advertiser_id_popularity = advertiser_id_popularity_df
# .filter('views > 100 and advertiser_id is not null')
# .select('advertiser_id', 'ctr')
# .rdd.collectAsMap()
# len(advertiser_id_popularity)
# ##3129


# ### Average CTR by campaign_id
campaign_id_popularity_df = train_set_df.select('clicked', 'campaign_id', 'ad_id') \
    .groupby('campaign_id').agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views'),
                                F.countDistinct('ad_id').alias('distinct_ad_ids')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))

campaign_id_popularity = campaign_id_popularity_df \
    .filter('views > 10 and campaign_id is not null') \
    .select('campaign_id', 'ctr', 'views', 'distinct_ad_ids') \
    .rdd.map(lambda x: (x['campaign_id'], (x['ctr'], x['views'], x['distinct_ad_ids'], 1))) \
    .collectAsMap()
len(campaign_id_popularity)

campaign_id_popularity_broad = sc.broadcast(campaign_id_popularity)

# campaign_id_popularity_df.count()
# ##31390

# get_percentiles(campaign_id_popularity_df, 'clicks')

# get_percentiles(campaign_id_popularity_df, 'views')

# campaign_id_popularity = campaign_id_popularity_df
# .filter('views > 100 and campaign_id is not null')
# .select('campaign_id', 'ctr')
# .rdd.collectAsMap()
# len(campaign_id_popularity)
# ##16097


# ### Average CTR by category

category_id_popularity_df = train_set_df.join(
    documents_categories_df.alias('cat_local'),
    on=F.col("document_id_promo") == F.col("cat_local.document_id_cat"), how='inner') \
    .select('clicked', 'category_id', 'confidence_level_cat', 'ad_id') \
    .groupby('category_id').agg(F.sum('clicked').alias('clicks'),
                                F.count('*').alias('views'),
                                F.mean('confidence_level_cat').alias('avg_confidence_level_cat'),
                                F.countDistinct('ad_id').alias('distinct_ad_ids')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))

category_id_popularity = category_id_popularity_df.filter('views > 10') \
    .select('category_id', 'ctr', 'views', 'avg_confidence_level_cat', 'distinct_ad_ids') \
    .rdd.map(lambda x: (x['category_id'],
                        (x['ctr'], x['views'], x['distinct_ad_ids'], x['avg_confidence_level_cat']))).collectAsMap()
len(category_id_popularity)

category_id_popularity_broad = sc.broadcast(category_id_popularity)

list(category_id_popularity.values())[:10]

np.median(np.array(list(map(lambda x: x[1], category_id_popularity.values()))))

sum(map(lambda x: x[1], category_id_popularity.values())) / float(len(category_id_popularity))

# Parece haver uma hierarquia nas categorias pelo padrão dos códigos...
# category_id_popularity


# ### Average CTR by Topic
topic_id_popularity_df = train_set_df.join(
    documents_topics_df.alias('top_local'),
    on=F.col("document_id_promo") == F.col("top_local.document_id_top"), how='inner') \
    .select('clicked', 'topic_id', 'confidence_level_top', 'ad_id') \
    .groupby('topic_id').agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views'),
                             F.mean('confidence_level_top').alias('avg_confidence_level_top'),
                             F.countDistinct('ad_id').alias('distinct_ad_ids')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))
topic_id_popularity = topic_id_popularity_df.filter('views > 10') \
    .select('topic_id', 'ctr', 'views', 'avg_confidence_level_top', 'distinct_ad_ids') \
    .rdd.map(lambda x: (x['topic_id'],
                        (x['ctr'], x['views'], x['distinct_ad_ids'], x['avg_confidence_level_top']))).collectAsMap()
len(topic_id_popularity)

topic_id_popularity_broad = sc.broadcast(topic_id_popularity)

sum(map(lambda x: x[1], topic_id_popularity.values())) / float(len(topic_id_popularity))

sum(map(lambda x: x[2] * x[1], topic_id_popularity.values())) / float(len(topic_id_popularity))

# ### Average CTR by Entity
entity_id_popularity_df = train_set_df.join(
    documents_entities_df.alias('ent_local'),
    on=F.col("document_id_promo") == F.col("ent_local.document_id_ent"), how='inner') \
    .select('clicked', 'entity_id', 'confidence_level_ent', 'ad_id') \
    .groupby('entity_id').agg(F.sum('clicked').alias('clicks'), F.count('*').alias('views'),
                              F.mean('confidence_level_ent').alias('avg_confidence_level_ent'),
                              F.countDistinct('ad_id').alias('distinct_ad_ids')) \
    .withColumn('ctr', ctr_udf('clicks', 'views'))

entity_id_popularity = entity_id_popularity_df.filter('views > 5') \
    .select('entity_id', 'ctr', 'views', 'avg_confidence_level_ent', 'distinct_ad_ids') \
    .rdd.map(lambda x: (x['entity_id'],
                        (x['ctr'], x['views'], x['distinct_ad_ids'], x['avg_confidence_level_ent']))).collectAsMap()
len(entity_id_popularity)

entity_id_popularity_broad = sc.broadcast(entity_id_popularity)

np.median(np.array(list(map(lambda x: x[1], entity_id_popularity.values()))))

sum(map(lambda x: x[1], entity_id_popularity.values())) / float(len(entity_id_popularity))

# ### Loading # docs by categories, topics, entities


df_filenames_suffix = '_eval'

categories_docs_counts = documents_categories_df.groupBy('category_id').count().rdd.collectAsMap()

topics_docs_counts = documents_topics_df.groupBy('topic_id').count().rdd.collectAsMap()

entities_docs_counts = documents_entities_df.groupBy('entity_id').count().rdd.collectAsMap()

documents_total = documents_meta_df.count()
documents_total

# ## Exploring Publish Time
publish_times_df = train_set_df.filter('publish_time is not null').select('document_id_promo',
                                                                          'publish_time').distinct().select(
    F.col('publish_time').cast(IntegerType()))

publish_time_percentiles = get_percentiles(publish_times_df, 'publish_time', quantiles_levels=[0.5],
                                           max_error_rate=0.001)
publish_time_percentiles

publish_time_median = int(publish_time_percentiles[0.5])
datetime.datetime.utcfromtimestamp(publish_time_median)


def get_days_diff(newer_timestamp, older_timestamp):
    sec_diff = newer_timestamp - older_timestamp
    days_diff = sec_diff / 60 / 60 / 24
    return days_diff


def get_time_decay_factor(timestamp, timestamp_ref=None, alpha=0.001):
    if timestamp_ref is None:
        timestamp_ref = time.time()

    days_diff = get_days_diff(timestamp_ref, timestamp)
    denominator = math.pow(1 + alpha, days_diff)
    if denominator != 0:
        return 1.0 / denominator
    else:
        return 0.0


TIME_DECAY_ALPHA = 0.0005

ref_dates = [
    1476714880,  # 7 days
    1474727680,  # 30 days
    1469370880,  # 90 days
    1461508480,  # 180 days
    1445697280,  # 1 year
    1414161280  # 2 years
]

for d in ref_dates:
    print(datetime.datetime.utcfromtimestamp(d), get_time_decay_factor(d, alpha=TIME_DECAY_ALPHA))

# ### Get local time
DEFAULT_TZ_EST = -4.0


def get_local_utc_bst_tz(event_country, event_country_state):
    local_tz = DEFAULT_TZ_EST
    if len(event_country) > 0:
        if event_country in countries_utc_dst_broad.value:
            local_tz = countries_utc_dst_broad.value[event_country]
            if len(event_country_state) > 2:
                state = event_country_state[3:5]
                if event_country == 'US':
                    if state in us_states_utc_dst_broad.value:
                        local_tz = us_states_utc_dst_broad.value[state]
                elif event_country == 'CA':
                    if state in ca_countries_utc_dst_broad.value:
                        local_tz = ca_countries_utc_dst_broad.value[state]
    return float(local_tz)


hour_bins_dict = {'EARLY_MORNING': 0,
                  'MORNING': 1,
                  'MIDDAY': 2,
                  'AFTERNOON': 3,
                  'EVENING': 4,
                  'NIGHT': 5}

hour_bins_values = sorted(hour_bins_dict.values())


def get_hour_bin(hour):
    if hour >= 5 and hour < 8:
        hour_bin = hour_bins_dict['EARLY_MORNING']
    elif hour >= 8 and hour < 11:
        hour_bin = hour_bins_dict['MORNING']
    elif hour >= 11 and hour < 14:
        hour_bin = hour_bins_dict['MIDDAY']
    elif hour >= 14 and hour < 19:
        hour_bin = hour_bins_dict['AFTERNOON']
    elif hour >= 19 and hour < 22:
        hour_bin = hour_bins_dict['EVENING']
    else:
        hour_bin = hour_bins_dict['NIGHT']
    return hour_bin


def get_local_datetime(dt, event_country, event_country_state):
    local_tz = get_local_utc_bst_tz(event_country, event_country_state)
    tz_delta = local_tz - DEFAULT_TZ_EST
    local_time = dt + datetime.timedelta(hours=tz_delta)
    return local_time


get_local_datetime(datetime.datetime.now(), 'US', 'US>CA')


def is_weekend(dt):
    return dt.weekday() >= 5


is_weekend(datetime.datetime(2016, 6, 14))

# ## Average CTR functions
timestamp_ref = date_time_to_unix_epoch(datetime.datetime(2016, 6, 29, 3, 59, 59))
decay_factor_default = get_time_decay_factor(publish_time_median, timestamp_ref, alpha=TIME_DECAY_ALPHA)
print("decay_factor_default", decay_factor_default)


def get_confidence_sample_size(sample, max_for_reference=100000):
    # Avoiding overflow for large sample size
    if sample >= max_for_reference:
        return 1.0

    ref_log = math.log(1 + max_for_reference,
                       2)  # Curiosly reference in log  with base 2 gives a slightly higher score, so I will keep

    return math.log(1 + sample) / float(ref_log)


for i in [0, 0.5, 1, 2, 3, 4, 5, 10, 20, 30, 100, 200, 300, 1000, 2000, 3000, 10000, 20000, 30000, 50000, 90000, 100000,
          500000, 900000, 1000000, 2171607]:
    print(i, get_confidence_sample_size(i))


def get_popularity(an_id, a_dict):
    return (a_dict[an_id][0], get_confidence_sample_size(a_dict[an_id][1] / float(a_dict[an_id][2])) * a_dict[an_id][
        3]) if an_id in a_dict else (None, None)






def get_popularity_score(event_country, ad_id, document_id, source_id,
                         publisher_id, advertiser_id, campaign_id, document_id_event,
                         category_ids_by_doc, cat_confidence_level_by_doc,
                         topic_ids_by_doc, top_confidence_level_by_doc,
                         entity_ids_by_doc, ent_confidence_level_by_doc,
                         output_detailed_list=False):
    probs = []

    avg_ctr, confidence = get_popularity(ad_id, ad_id_popularity_broad.value)
    if avg_ctr is not None:
        probs.append(('pop_ad_id', avg_ctr))

    avg_ctr, confidence = get_popularity(document_id, document_id_popularity_broad.value)
    if avg_ctr is not None:
        probs.append(('pop_document_id', avg_ctr))


    if source_id != -1:
        avg_ctr = None


        avg_ctr, confidence = get_popularity(source_id, source_id_popularity_broad.value)
        if avg_ctr is not None:
            probs.append(('pop_source_id', avg_ctr))

    if publisher_id is not None:
        avg_ctr, confidence = get_popularity(publisher_id, publisher_popularity_broad.value)
        if avg_ctr is not None:
            probs.append(('pop_publisher_id', avg_ctr))

    if advertiser_id is not None:
        avg_ctr, confidence = get_popularity(advertiser_id, advertiser_id_popularity_broad.value)
        if avg_ctr is not None:
            probs.append(('pop_advertiser_id', avg_ctr))

    if campaign_id is not None:
        avg_ctr, confidence = get_popularity(campaign_id, campaign_id_popularity_broad.value)
        if avg_ctr is not None:
            probs.append(('pop_campain_id', avg_ctr))

    return probs

        # ## Content-Based similarity functions


def cosine_similarity_dicts(dict1, dict2):
    dict1_norm = math.sqrt(sum([v ** 2 for v in dict1.values()]))
    dict2_norm = math.sqrt(sum([v ** 2 for v in dict2.values()]))

    sum_common_aspects = 0.0
    intersections = 0
    for key in dict1:
        if key in dict2:
            sum_common_aspects += dict1[key] * dict2[key]
            intersections += 1

    return sum_common_aspects / (dict1_norm * dict2_norm), intersections





def cosine_similarity_doc_event_doc_ad_aspects(doc_event_aspect_ids, doc_event_aspects_confidence,
                                               doc_ad_aspect_ids, doc_ad_aspects_confidence,
                                               aspect_docs_counts):
    if doc_event_aspect_ids is None or len(doc_event_aspect_ids) == 0 \
            or doc_ad_aspect_ids is None or len(doc_ad_aspect_ids) == 0:
        return None, None

    doc_event_aspects = dict(zip(doc_event_aspect_ids, doc_event_aspects_confidence))
    doc_event_aspects_tfidf_confid = {}
    for key in doc_event_aspect_ids:
        tf = 1.0
        idf = math.log(math.log(documents_total / float(aspect_docs_counts[key])))
        confidence = doc_event_aspects[key]
        doc_event_aspects_tfidf_confid[key] = tf * idf * confidence

    doc_ad_aspects = dict(zip(doc_ad_aspect_ids, doc_ad_aspects_confidence))
    doc_ad_aspects_tfidf_confid = {}
    for key in doc_ad_aspect_ids:
        tf = 1.0
        idf = math.log(math.log(documents_total / float(aspect_docs_counts[key])))
        confidence = doc_ad_aspects[key]
        doc_ad_aspects_tfidf_confid[key] = tf * idf * confidence

    similarity, intersections = cosine_similarity_dicts(doc_event_aspects_tfidf_confid, doc_ad_aspects_tfidf_confid)

    if intersections > 0:
        # P(A intersect B)_intersections = P(A)^intersections * P(B)^intersections
        random_error = math.pow(len(doc_event_aspect_ids) / float(len(aspect_docs_counts)),
                                intersections) * math.pow(len(doc_ad_aspect_ids) / float(len(aspect_docs_counts)),
                                                          intersections)
    else:
        # P(A not intersect B) = 1 - P(A intersect B)
        random_error = 1 - ((len(doc_event_aspect_ids) / float(len(aspect_docs_counts))) *
                            (len(doc_ad_aspect_ids) / float(len(aspect_docs_counts))))

    confidence = 1.0 - random_error

    return similarity, confidence





def get_doc_event_doc_ad_cb_similarity_score(doc_event_category_ids, doc_event_cat_confidence_levels,
                                             doc_event_topic_ids, doc_event_top_confidence_levels,
                                             doc_event_entity_ids, doc_event_ent_confidence_levels,
                                             doc_ad_category_ids, doc_ad_cat_confidence_levels,
                                             doc_ad_topic_ids, doc_ad_top_confidence_levels,
                                             doc_ad_entity_ids, doc_ad_ent_confidence_levels,
                                             output_detailed_list=False):
    # Content-Based
    sims = []

    categories_similarity, cat_sim_confidence = cosine_similarity_doc_event_doc_ad_aspects(
        doc_event_category_ids, doc_event_cat_confidence_levels,
        doc_ad_category_ids, doc_ad_cat_confidence_levels,
        categories_docs_counts)
    if categories_similarity is not None:
        sims.append(('doc_event_doc_ad_sim_categories', categories_similarity, cat_sim_confidence))

    topics_similarity, top_sim_confidence = cosine_similarity_doc_event_doc_ad_aspects(
        doc_event_topic_ids, doc_event_top_confidence_levels,
        doc_ad_topic_ids, doc_ad_top_confidence_levels,
        topics_docs_counts)

    if topics_similarity is not None:
        sims.append(('doc_event_doc_ad_sim_topics', topics_similarity, top_sim_confidence))

    entities_similarity, entity_sim_confid = cosine_similarity_doc_event_doc_ad_aspects(
        doc_event_entity_ids, doc_event_ent_confidence_levels,
        doc_ad_entity_ids, doc_ad_ent_confidence_levels,
        entities_docs_counts)

    if entities_similarity is not None:
        sims.append(('doc_event_doc_ad_sim_entities', entities_similarity, entity_sim_confid))


    return sims



# # Feature Vector export
bool_feature_names = []

int_feature_names = ['ad_views',
                     'campaign_id',
                     'doc_views',
                     'doc_event_days_since_published',
                     'doc_ad_days_since_published',
                     ]

float_feature_names = [
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

TRAFFIC_SOURCE_FV = 'traffic_source'
EVENT_HOUR_FV = 'event_hour'
EVENT_COUNTRY_FV = 'event_country'
EVENT_COUNTRY_STATE_FV = 'event_country_state'
EVENT_GEO_LOCATION_FV = 'event_geo_location'
EVENT_PLATFORM_FV = 'event_platform'
AD_ADVERTISER_FV = 'ad_advertiser'
DOC_AD_SOURCE_ID_FV = 'doc_ad_source_id'
DOC_AD_PUBLISHER_ID_FV = 'doc_ad_publisher_id'
DOC_EVENT_SOURCE_ID_FV = 'doc_event_source_id'
DOC_EVENT_PUBLISHER_ID_FV = 'doc_event_publisher_id'
DOC_AD_CATEGORY_ID_FV = 'doc_ad_category_id'
DOC_AD_TOPIC_ID_FV = 'doc_ad_topic_id'
DOC_AD_ENTITY_ID_FV = 'doc_ad_entity_id'
DOC_EVENT_CATEGORY_ID_FV = 'doc_event_category_id'
DOC_EVENT_TOPIC_ID_FV = 'doc_event_topic_id'
DOC_EVENT_ENTITY_ID_FV = 'doc_event_entity_id'

# ### Configuring feature vector
category_feature_names_integral = ['ad_advertiser',
                                   'doc_ad_publisher_id',
                                   'doc_ad_source_id',
                                   'doc_event_publisher_id',
                                   'doc_event_source_id',
                                   'event_country',
                                   'event_country_state',
                                   'event_geo_location',
                                   'event_platform',
                                   'traffic_source']

feature_vector_labels_integral = bool_feature_names \
                                 + int_feature_names \
                                 + float_feature_names \
                                 + category_feature_names_integral
print(feature_vector_labels_integral)
feature_vector_labels_integral_dict = dict([(key, idx) for idx, key in enumerate(feature_vector_labels_integral)])

with open(OUTPUT_BUCKET_FOLDER + 'feature_vector_labels_integral.txt', 'w') as output:
    output.writelines('\n'.join(feature_vector_labels_integral))


# ### Building feature vectors
def set_feature_vector_cat_value_integral(field_name, field_value, feature_vector):
    if not is_null(field_value):  # and str(field_value) != '-1':
        feature_vector[feature_vector_labels_integral_dict[field_name]] = float(field_value)


def set_feature_vector_cat_top_multi_values_integral(
        field_name, values, confidences, feature_vector, top=5):
    top_values = list(filter(lambda z: z != -1,
                             map(lambda y: y[0], sorted(zip(values, confidences), key=lambda x: -x[1]))))[:top]
    for idx, field_value in list(enumerate(top_values)):
        set_feature_vector_cat_value_integral(
            '{}_{}'.format(field_name, idx + 1), field_value, feature_vector)


def get_ad_feature_vector_integral(
        event_country, event_country_state,
        ad_id, document_id, source_id, doc_ad_publish_time, timestamp_event, platform_event,
        geo_location_event,
        doc_event_source_id, doc_event_publisher_id, doc_event_publish_time,
        advertiser_id, publisher_id,
        campaign_id, document_id_event,
        doc_ad_category_ids, doc_ad_cat_confidence_levels,
        doc_ad_topic_ids, doc_ad_top_confidence_levels,
        doc_ad_entity_ids, doc_ad_ent_confidence_levels,
        doc_event_category_ids, doc_event_cat_confidence_levels,
        doc_event_topic_ids, doc_event_top_confidence_levels,
        doc_event_entity_ids, doc_event_ent_confidence_levels):
    try:

        feature_vector = {}
        feature_vector[feature_vector_labels_integral_dict['campaign_id']] = campaign_id
        if ad_id in ad_id_popularity_broad.value:
            feature_vector[feature_vector_labels_integral_dict['ad_views']] = float(
                ad_id_popularity_broad.value[ad_id][1])

        if document_id in document_id_popularity_broad.value:
            feature_vector[feature_vector_labels_integral_dict['doc_views']] = float(
                document_id_popularity_broad.value[document_id][1])

        if timestamp_event > -1:
            dt_timestamp_event = convert_odd_timestamp(timestamp_event)
            if doc_ad_publish_time is not None:
                delta_days = (dt_timestamp_event - doc_ad_publish_time).days
                if 0 <= delta_days <= 365 * 10:  # 10 years
                    feature_vector[feature_vector_labels_integral_dict['doc_ad_days_since_published']] = float(
                        delta_days)

            if doc_event_publish_time is not None:
                delta_days = (dt_timestamp_event - doc_event_publish_time).days
                if 0 <= delta_days <= 365 * 10:  # 10 years
                    feature_vector[feature_vector_labels_integral_dict['doc_event_days_since_published']] = float(
                        delta_days)


        # Setting Popularity fields
        pop_scores = get_popularity_score(event_country, ad_id, document_id, source_id,
                                          publisher_id, advertiser_id, campaign_id, document_id_event,
                                          doc_ad_category_ids, doc_ad_cat_confidence_levels,
                                          doc_ad_topic_ids, doc_ad_top_confidence_levels,
                                          doc_ad_entity_ids, doc_ad_ent_confidence_levels,
                                          output_detailed_list=True)

        for score in pop_scores:
            feature_vector[feature_vector_labels_integral_dict[score[0]]] = score[1]


        # Setting Doc_event-doc_ad CB Similarity fields
        doc_event_doc_ad_cb_sim_scores = get_doc_event_doc_ad_cb_similarity_score(
            doc_event_category_ids, doc_event_cat_confidence_levels,
            doc_event_topic_ids, doc_event_top_confidence_levels,
            doc_event_entity_ids, doc_event_ent_confidence_levels,
            doc_ad_category_ids, doc_ad_cat_confidence_levels,
            doc_ad_topic_ids, doc_ad_top_confidence_levels,
            doc_ad_entity_ids, doc_ad_ent_confidence_levels,
            output_detailed_list=True)

        for score in doc_event_doc_ad_cb_sim_scores:
            feature_vector[feature_vector_labels_integral_dict[score[0]]] = score[1]

        # Process code for event_country
        if event_country in event_country_values_counts:
            event_country_code = event_country_values_counts[event_country]
        else:
            event_country_code = event_country_values_counts[LESS_SPECIAL_CAT_VALUE]
        set_feature_vector_cat_value_integral(EVENT_COUNTRY_FV, event_country_code, feature_vector)

        # Process code for event_country_state
        if event_country_state in event_country_state_values_counts:
            event_country_state_code = event_country_state_values_counts[event_country_state]
        else:
            event_country_state_code = event_country_state_values_counts[LESS_SPECIAL_CAT_VALUE]
        set_feature_vector_cat_value_integral(EVENT_COUNTRY_STATE_FV, event_country_state_code, feature_vector)

        # Process code for geo_location_event
        if geo_location_event in event_geo_location_values_counts:
            geo_location_event_code = event_geo_location_values_counts[geo_location_event]
        else:
            geo_location_event_code = event_geo_location_values_counts[LESS_SPECIAL_CAT_VALUE]

        # -1 to traffic_source and platform_event
        if platform_event is not None:
            feature_vector[feature_vector_labels_integral_dict[EVENT_PLATFORM_FV]] = int(platform_event - 1)

        set_feature_vector_cat_value_integral(EVENT_GEO_LOCATION_FV, geo_location_event_code, feature_vector)

        # set_feature_vector_cat_value_integral(EVENT_PLATFORM_FV, platform_event - 1, feature_vector)
        set_feature_vector_cat_value_integral(AD_ADVERTISER_FV, advertiser_id, feature_vector)
        set_feature_vector_cat_value_integral(DOC_AD_SOURCE_ID_FV, source_id, feature_vector)
        set_feature_vector_cat_value_integral(DOC_AD_PUBLISHER_ID_FV, publisher_id, feature_vector)
        set_feature_vector_cat_value_integral(DOC_EVENT_SOURCE_ID_FV, doc_event_source_id, feature_vector)
        set_feature_vector_cat_value_integral(DOC_EVENT_PUBLISHER_ID_FV, doc_event_publisher_id, feature_vector)




    except Exception as e:
        raise Exception("[get_ad_feature_vector_integral] ERROR PROCESSING FEATURE VECTOR! Params: {}"
                        .format([event_country, event_country_state,
                                 ad_id, document_id, source_id, doc_ad_publish_time, timestamp_event, platform_event,
                                 geo_location_event,
                                 doc_event_source_id, doc_event_publisher_id, doc_event_publish_time,
                                 advertiser_id, publisher_id,
                                 campaign_id, document_id_event,
                                 doc_ad_category_ids, doc_ad_cat_confidence_levels,
                                 doc_ad_topic_ids, doc_ad_top_confidence_levels,
                                 doc_ad_entity_ids, doc_ad_ent_confidence_levels,
                                 doc_event_category_ids, doc_event_cat_confidence_levels,
                                 doc_event_topic_ids, doc_event_top_confidence_levels,
                                 doc_event_entity_ids, doc_event_ent_confidence_levels]),
                        e)

    return SparseVector(len(feature_vector_labels_integral_dict), feature_vector)


get_ad_feature_vector_integral_udf = F.udf(
    lambda event_country, event_country_state, ad_id, document_id, source_id,
           doc_ad_publish_time, timestamp_event, platform_event,
           geo_location_event,
           doc_event_source_id, doc_event_publisher_id, doc_event_publish_time,
           advertiser_id, publisher_id,
           campaign_id, document_id_event,
           category_ids_by_doc, cat_confidence_level_by_doc,
           topic_ids_by_doc, top_confidence_level_by_doc,
           entity_ids_by_doc, ent_confidence_level_by_doc,
           doc_event_category_id_list, doc_event_confidence_level_cat_list,
           doc_event_topic_id_list, doc_event_confidence_level_top,
           doc_event_entity_id_list, doc_event_confidence_level_ent:
    get_ad_feature_vector_integral(event_country, event_country_state,
                                   ad_id, document_id, source_id, doc_ad_publish_time, timestamp_event,
                                   platform_event,
                                   geo_location_event,
                                   doc_event_source_id, doc_event_publisher_id, doc_event_publish_time,
                                   advertiser_id, publisher_id,
                                   campaign_id, document_id_event,
                                   category_ids_by_doc, cat_confidence_level_by_doc,
                                   topic_ids_by_doc, top_confidence_level_by_doc,
                                   entity_ids_by_doc, ent_confidence_level_by_doc,
                                   doc_event_category_id_list, doc_event_confidence_level_cat_list,
                                   doc_event_topic_id_list, doc_event_confidence_level_top,
                                   doc_event_entity_id_list, doc_event_confidence_level_ent),
    VectorUDT())

# ## Export Train set feature vectors
train_set_enriched_df = train_set_df \
    .join(documents_categories_grouped_df,
          on=F.col("document_id_promo") == F.col("documents_categories_grouped.document_id_cat"),
          how='left') \
    .join(documents_topics_grouped_df,
          on=F.col("document_id_promo") == F.col("documents_topics_grouped.document_id_top"),
          how='left') \
    .join(documents_entities_grouped_df,
          on=F.col("document_id_promo") == F.col("documents_entities_grouped.document_id_ent"),
          how='left') \
    .join(documents_categories_grouped_df
          .withColumnRenamed('category_id_list', 'doc_event_category_id_list')
          .withColumnRenamed('confidence_level_cat_list', 'doc_event_confidence_level_cat_list')
          .alias('documents_event_categories_grouped'),
          on=F.col("document_id_event") == F.col("documents_event_categories_grouped.document_id_cat"),
          how='left') \
    .join(documents_topics_grouped_df
          .withColumnRenamed('topic_id_list', 'doc_event_topic_id_list')
          .withColumnRenamed('confidence_level_top_list', 'doc_event_confidence_level_top_list')
          .alias('documents_event_topics_grouped'),
          on=F.col("document_id_event") == F.col("documents_event_topics_grouped.document_id_top"),
          how='left') \
    .join(documents_entities_grouped_df
          .withColumnRenamed('entity_id_list', 'doc_event_entity_id_list')
          .withColumnRenamed('confidence_level_ent_list', 'doc_event_confidence_level_ent_list')
          .alias('documents_event_entities_grouped'),
          on=F.col("document_id_event") == F.col("documents_event_entities_grouped.document_id_ent"),
          how='left') \
    .select('display_id', 'uuid_event', 'event_country', 'event_country_state', 'platform_event',
            'source_id_doc_event', 'publisher_doc_event', 'publish_time_doc_event',
            'publish_time', 'ad_id', 'document_id_promo', 'clicked',
            'geo_location_event', 'advertiser_id', 'publisher_id',
            'campaign_id', 'document_id_event',
            int_list_null_to_empty_list_udf('doc_event_category_id_list')
            .alias('doc_event_category_id_list'),
            float_list_null_to_empty_list_udf('doc_event_confidence_level_cat_list')
            .alias('doc_event_confidence_level_cat_list'),
            int_list_null_to_empty_list_udf('doc_event_topic_id_list')
            .alias('doc_event_topic_id_list'),
            float_list_null_to_empty_list_udf('doc_event_confidence_level_top_list')
            .alias('doc_event_confidence_level_top_list'),
            str_list_null_to_empty_list_udf('doc_event_entity_id_list')
            .alias('doc_event_entity_id_list'),
            float_list_null_to_empty_list_udf('doc_event_confidence_level_ent_list')
            .alias('doc_event_confidence_level_ent_list'),
            int_null_to_minus_one_udf('source_id').alias('source_id'),
            int_null_to_minus_one_udf('timestamp_event').alias('timestamp_event'),
            int_list_null_to_empty_list_udf('category_id_list').alias('category_id_list'),
            float_list_null_to_empty_list_udf('confidence_level_cat_list')
            .alias('confidence_level_cat_list'),
            int_list_null_to_empty_list_udf('topic_id_list').alias('topic_id_list'),
            float_list_null_to_empty_list_udf('confidence_level_top_list')
            .alias('confidence_level_top_list'),
            str_list_null_to_empty_list_udf('entity_id_list').alias('entity_id_list'),
            float_list_null_to_empty_list_udf('confidence_level_ent_list')
            .alias('confidence_level_ent_list')) \

train_set_feature_vectors_df = train_set_enriched_df \
    .withColumn('feature_vector',
                get_ad_feature_vector_integral_udf(
                    'event_country',
                    'event_country_state',
                    'ad_id',
                    'document_id_promo',
                    'source_id',
                    'publish_time',
                    'timestamp_event',
                    'platform_event',
                    'geo_location_event',
                    'source_id_doc_event',
                    'publisher_doc_event',
                    'publish_time_doc_event',
                    'advertiser_id',
                    'publisher_id',
                    'campaign_id',
                    'document_id_event',
                    'category_id_list',
                    'confidence_level_cat_list',
                    'topic_id_list',
                    'confidence_level_top_list',
                    'entity_id_list',
                    'confidence_level_ent_list',
                    'doc_event_category_id_list',
                    'doc_event_confidence_level_cat_list',
                    'doc_event_topic_id_list',
                    'doc_event_confidence_level_top_list',
                    'doc_event_entity_id_list',
                    'doc_event_confidence_level_ent_list')) \
    .select(F.col('uuid_event').alias('uuid'), 'display_id', 'ad_id', 'document_id_event',
            F.col('document_id_promo').alias('document_id'), F.col('clicked').alias('label'),
            'feature_vector')


train_feature_vector_gcs_folder_name = 'train_feature_vectors_integral_eval'


train_set_feature_vectors_df.write.parquet(OUTPUT_BUCKET_FOLDER + train_feature_vector_gcs_folder_name,
                                           mode='overwrite')


# # Export Validation/Test set feature vectors




data_df = validation_set_df


test_validation_set_enriched_df = data_df.select(
    'display_id', 'uuid_event', 'event_country', 'event_country_state', 'platform_event',
    'source_id_doc_event', 'publisher_doc_event', 'publish_time_doc_event',
    'publish_time',
    'ad_id', 'document_id_promo', 'clicked',
    'geo_location_event', 'advertiser_id', 'publisher_id',
    'campaign_id', 'document_id_event',
    int_list_null_to_empty_list_udf('doc_event_category_id_list')
        .alias('doc_event_category_id_list'),
    float_list_null_to_empty_list_udf('doc_event_confidence_level_cat_list')
        .alias('doc_event_confidence_level_cat_list'),
    int_list_null_to_empty_list_udf('doc_event_topic_id_list')
        .alias('doc_event_topic_id_list'),
    float_list_null_to_empty_list_udf('doc_event_confidence_level_top_list')
        .alias('doc_event_confidence_level_top_list'),
    str_list_null_to_empty_list_udf('doc_event_entity_id_list')
        .alias('doc_event_entity_id_list'),
    float_list_null_to_empty_list_udf('doc_event_confidence_level_ent_list')
        .alias('doc_event_confidence_level_ent_list'),
    int_null_to_minus_one_udf('source_id')
        .alias('source_id'),
    int_null_to_minus_one_udf('timestamp_event').alias('timestamp_event'),
    int_list_null_to_empty_list_udf('category_id_list').alias('category_id_list'),
    float_list_null_to_empty_list_udf('confidence_level_cat_list')
        .alias('confidence_level_cat_list'),
    int_list_null_to_empty_list_udf('topic_id_list').alias('topic_id_list'),
    float_list_null_to_empty_list_udf('confidence_level_top_list')
        .alias('confidence_level_top_list'),
    str_list_null_to_empty_list_udf('entity_id_list').alias('entity_id_list'),
    float_list_null_to_empty_list_udf('confidence_level_ent_list')
        .alias('confidence_level_ent_list')) \


test_validation_set_feature_vectors_df = test_validation_set_enriched_df \
    .withColumn('feature_vector',
                get_ad_feature_vector_integral_udf(
                    'event_country',
                    'event_country_state',
                    'ad_id',
                    'document_id_promo',
                    'source_id',
                    'publish_time',
                    'timestamp_event',
                    'platform_event',
                    'geo_location_event',
                    'source_id_doc_event',
                    'publisher_doc_event',
                    'publish_time_doc_event',
                    'advertiser_id',
                    'publisher_id',
                    'campaign_id',
                    'document_id_event',
                    'category_id_list',
                    'confidence_level_cat_list',
                    'topic_id_list',
                    'confidence_level_top_list',
                    'entity_id_list',
                    'confidence_level_ent_list',
                    'doc_event_category_id_list',
                    'doc_event_confidence_level_cat_list',
                    'doc_event_topic_id_list',
                    'doc_event_confidence_level_top_list',
                    'doc_event_entity_id_list',
                    'doc_event_confidence_level_ent_list')) \
    .select(F.col('uuid_event').alias('uuid'), 'display_id', 'ad_id', 'document_id_event',
            F.col('document_id_promo').alias('document_id'), F.col('clicked').alias('label'),
            'feature_vector')


test_validation_feature_vector_gcs_folder_name = 'validation_feature_vectors_integral'


test_validation_set_feature_vectors_df.write.parquet(
    OUTPUT_BUCKET_FOLDER + test_validation_feature_vector_gcs_folder_name, mode='overwrite')

spark.stop()
