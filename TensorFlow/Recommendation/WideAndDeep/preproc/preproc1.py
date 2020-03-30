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

OUTPUT_BUCKET_FOLDER = "/outbrain/preprocessed/"
DATA_BUCKET_FOLDER = "/outbrain/orig/"
SPARK_TEMP_FOLDER = "/outbrain/spark-temp/"

from pyspark.sql.types import IntegerType, StringType, StructType, StructField
import pyspark.sql.functions as F

from pyspark.context import SparkContext, SparkConf
from pyspark.sql.session import SparkSession

conf = SparkConf().setMaster('local[*]').set('spark.executor.memory', '256g').set('spark.driver.memory', '126g').set("spark.local.dir", SPARK_TEMP_FOLDER)

sc = SparkContext(conf=conf)
spark = SparkSession(sc)

print('Loading data...')

truncate_day_from_timestamp_udf = F.udf(lambda ts: int(ts / 1000 / 60 / 60 / 24), IntegerType())

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
  .withColumn('day_event', truncate_day_from_timestamp_udf('timestamp_event')) \
  .alias('events')   

events_df.count()

print('Drop rows with empty "geo_location"...')
events_df = events_df.dropna(subset="geo_location_event")
events_df.count()

print('Drop rows with empty "platform"...')
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
  .csv(DATA_BUCKET_FOLDER+"promoted_content.csv") \
  .alias('promoted_content')

clicks_train_schema = StructType(
  [StructField("display_id", IntegerType(), True),
  StructField("ad_id", IntegerType(), True),                    
  StructField("clicked", IntegerType(), True)]
  )

clicks_train_df = spark.read.schema(clicks_train_schema) \
  .options(header='true', inferschema='false', nullValue='\\N') \
  .csv(DATA_BUCKET_FOLDER+"clicks_train.csv") \
  .alias('clicks_train')


clicks_train_joined_df = clicks_train_df \
  .join(promoted_content_df, on='ad_id', how='left') \
  .join(events_df, on='display_id', how='left')
clicks_train_joined_df.createOrReplaceTempView('clicks_train_joined')

validation_display_ids_df = clicks_train_joined_df.select('display_id','day_event') \
  .distinct() \
  .sampleBy("day_event", fractions={0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, \
  5: 0.2, 6: 0.2, 7: 0.2, 8: 0.2, 9: 0.2, 10: 0.2, 11: 1.0, 12: 1.0}, seed=0)
validation_display_ids_df.createOrReplaceTempView("validation_display_ids")
validation_set_df = spark.sql('''SELECT display_id, ad_id, uuid_event, day_event, 
  timestamp_event, document_id_promo, platform_event, geo_location_event 
  FROM clicks_train_joined t
    WHERE EXISTS (SELECT display_id FROM validation_display_ids 
      WHERE display_id = t.display_id)''')

validation_set_gcs_output = "validation_set.parquet"
validation_set_df.write.parquet(OUTPUT_BUCKET_FOLDER+validation_set_gcs_output, mode='overwrite')

print(validation_set_df.take(5))

spark.stop()
