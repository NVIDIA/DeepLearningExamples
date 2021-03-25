# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
import sys

from argparse import ArgumentParser
from collections import OrderedDict
from contextlib import contextmanager
from operator import itemgetter
from time import time

from pyspark import broadcast
from pyspark.sql import Row, SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *


LABEL_COL = 0
INT_COLS = list(range(1, 14))
CAT_COLS = list(range(14, 40))


def get_column_counts_with_frequency_limit(df, frequency_limit = None):
    cols = ['_c%d' % i for i in CAT_COLS]
    df = (df
        .select(posexplode(array(*cols)))
        .withColumnRenamed('pos', 'column_id')
        .withColumnRenamed('col', 'data')
        .filter('data is not null')
        .groupBy('column_id', 'data')
        .count())

    if frequency_limit:
        frequency_limit = frequency_limit.split(",")
        exclude = []
        default_limit = None
        for fl in frequency_limit:
            frequency_pair = fl.split(":")
            if len(frequency_pair) == 1:
                default_limit = int(frequency_pair[0])
            elif len(frequency_pair) == 2:
                df = df.filter((col('column_id') != int(frequency_pair[0]) - CAT_COLS[0]) | (col('count') >= int(frequency_pair[1])))
                exclude.append(int(frequency_pair[0]))
        if default_limit:
            remain = [x - CAT_COLS[0] for x in CAT_COLS if x not in exclude]
            df = df.filter((~col('column_id').isin(remain)) | (col('count') >= default_limit))
            # for comparing isin and separate filter
            # for i in remain:
            #     df = df.filter((col('column_id') != i - CAT_COLS[0]) | (col('count') >= default_limit))
    return df


def assign_id_with_window(df):
    windowed = Window.partitionBy('column_id').orderBy(desc('count'))
    return (df
            .withColumn('id', row_number().over(windowed))
            .withColumnRenamed('count', 'model_count'))


def assign_low_mem_partial_ids(df):
    # To avoid some scaling issues with a simple window operation, we use a more complex method
    # to compute the same thing, but in a more distributed spark specific way
    df = df.orderBy(asc('column_id'), desc('count'))
    # The monotonically_increasing_id is the partition id in the top 31 bits and the rest
    # is an increasing count of the rows within that partition.  So we split it into two parts,
    # the partion id part_id and the count mono_id
    df = df.withColumn('part_id', spark_partition_id())
    return df.withColumn('mono_id', monotonically_increasing_id() - shiftLeft(col('part_id'), 33))


def assign_low_mem_final_ids(df):
    # Now we can find the minimum and maximum mono_ids within a given column/partition pair
    sub_model = df.groupBy('column_id', 'part_id').agg(max('mono_id').alias('top'), min('mono_id').alias('bottom'))
    sub_model = sub_model.withColumn('diff', col('top') - col('bottom') + 1)
    sub_model = sub_model.drop('top')
    # This window function is over aggregated column/partition pair table. It will do a running sum of the rows
    # within that column
    windowed = Window.partitionBy('column_id').orderBy('part_id').rowsBetween(Window.unboundedPreceding, -1)
    sub_model = sub_model.withColumn('running_sum', sum('diff').over(windowed)).na.fill(0, ["running_sum"])

    joined = df.withColumnRenamed('column_id', 'i_column_id')
    joined = joined.withColumnRenamed('part_id', 'i_part_id')
    joined = joined.withColumnRenamed('count', 'model_count')

    # Then we can join the original input with the pair it is a part of
    joined = joined.join(sub_model, (col('i_column_id') == col('column_id')) & (col('part_id') == col('i_part_id')))

    # So with all that we can subtract bottom from mono_id makeing it start at 0 for each partition
    # and then add in the running_sum so the id is contiguous and unique for the entire column. + 1 to make it match the 1 based indexing
    # for row_number
    ret = joined.select(col('column_id'),
                        col('data'),
                        (col('mono_id') - col('bottom') + col('running_sum') + 1).cast(IntegerType()).alias('id'),
                        col('model_count'))
    return ret


def get_column_models(combined_model):
    for i in CAT_COLS:
        model = (combined_model
            .filter('column_id == %d' % (i - CAT_COLS[0]))
            .drop('column_id'))
        yield i, model


def col_of_rand_long():
    return (rand() * (1 << 52)).cast(LongType())

def skewed_join(df, model, col_name, cutoff):
    # Most versions of spark don't have a good way
    # to deal with a skewed join out of the box.
    # Some do and if you want to replace this with
    # one of those that would be great.
    
    # Because we have statistics about the skewedness
    # that we can used we divide the model up into two parts
    # one part is the highly skewed part and we do a
    # broadcast join for that part, but keep the result in
    # a separate column
    b_model = broadcast(model.filter(col('model_count') >= cutoff)
            .withColumnRenamed('data', col_name)
            .drop('model_count'))
    
    df = (df
            .join(b_model, col_name, how='left')
            .withColumnRenamed('id', 'id_tmp'))
    
    # We also need to spread the skewed data that matched
    # evenly.  We will use a source of randomness for this
    # but use a -1 for anything that still needs to be matched
    if 'ordinal' in df.columns:
        rand_column = col('ordinal')
    else:
        rand_column = col_of_rand_long()

    df = df.withColumn('join_rand',
            # null values are not in the model, they are filtered out
            # but can be a source of skewedness so include them in
            # the even distribution
            when(col('id_tmp').isNotNull() | col(col_name).isNull(), rand_column)
            .otherwise(lit(-1)))
    
    # Null out the string data that already matched to save memory
    df = df.withColumn(col_name,
            when(col('id_tmp').isNotNull(), None)
            .otherwise(col(col_name)))
    
    # Now do the second join, which will be a non broadcast join.
    # Sadly spark is too smart for its own good and will optimize out
    # joining on a column it knows will always be a constant value.
    # So we have to make a convoluted version of assigning a -1 to the
    # randomness column for the model itself to work around that.
    nb_model = (model
            .withColumn('join_rand', when(col('model_count') < cutoff, lit(-1)).otherwise(lit(-2)))
            .filter(col('model_count') < cutoff)
            .withColumnRenamed('data', col_name)
            .drop('model_count'))
    
    df = (df
            .join(nb_model, ['join_rand', col_name], how='left')
            .drop(col_name, 'join_rand')
            # Pick either join result as an answer
            .withColumn(col_name, coalesce(col('id'), col('id_tmp')))
            .drop('id', 'id_tmp'))

    return df


def apply_models(df, models, broadcast_model = False, skew_broadcast_pct = 1.0):
    # sort the models so broadcast joins come first. This is
    # so we reduce the amount of shuffle data sooner than later
    # If we parsed the string hex values to ints early on this would
    # not make a difference.
    models = sorted(models, key=itemgetter(3), reverse=True)
    for i, model, original_rows, would_broadcast in models:
        col_name = '_c%d' % i
        if not (would_broadcast or broadcast_model):
            # The data is highly skewed so we need to offset that
            cutoff = int(original_rows * skew_broadcast_pct/100.0)
            df = skewed_join(df, model, col_name, cutoff)
        else:
            # broadcast joins can handle skewed data so no need to
            # do anything special
            model = (model.drop('model_count')
                          .withColumnRenamed('data', col_name))
            model = broadcast(model) if broadcast_model else model
            df = (df
                .join(model, col_name, how='left')
                .drop(col_name)
                .withColumnRenamed('id', col_name))
    return df.fillna(0, ['_c%d' % i for i in CAT_COLS])


def transform_log(df, transform_log = False):
    cols = ['_c%d' % i for i in INT_COLS]
    if transform_log:
        for col_name in cols:
            df = df.withColumn(col_name, log(df[col_name] + 3))
    return df.fillna(0, cols)


def would_broadcast(spark, str_path):
    sc = spark.sparkContext
    config = sc._jsc.hadoopConfiguration()
    path = sc._jvm.org.apache.hadoop.fs.Path(str_path)
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(config)
    stat = fs.listFiles(path, True)
    sum = 0
    while stat.hasNext():
       sum = sum + stat.next().getLen()
    sql_conf = sc._jvm.org.apache.spark.sql.internal.SQLConf()
    cutoff = sql_conf.autoBroadcastJoinThreshold() * sql_conf.fileCompressionFactor()
    return sum <= cutoff

def delete_data_source(spark, path):
    sc = spark.sparkContext
    config = sc._jsc.hadoopConfiguration()
    path = sc._jvm.org.apache.hadoop.fs.Path(path)
    sc._jvm.org.apache.hadoop.fs.FileSystem.get(config).delete(path, True)


def load_raw(spark, folder, day_range):
    label_fields = [StructField('_c%d' % LABEL_COL, IntegerType())]
    int_fields = [StructField('_c%d' % i, IntegerType()) for i in INT_COLS]
    str_fields = [StructField('_c%d' % i, StringType()) for i in CAT_COLS]

    schema = StructType(label_fields + int_fields + str_fields)
    paths = [os.path.join(folder, 'day_%d' % i) for i in day_range]
    return (spark
        .read
        .schema(schema)
        .option('sep', '\t')
        .csv(paths))

def rand_ordinal(df):
    # create a random long from the double precision float.  
    # The fraction part of a double is 52 bits, so we try to capture as much
    # of that as possible
    return df.withColumn('ordinal', col_of_rand_long())

def day_from_ordinal(df, num_days):
    return df.withColumn('day', (col('ordinal') % num_days).cast(IntegerType()))

def day_from_input_file(df):
    return df.withColumn('day', substring_index(input_file_name(), '_', -1).cast(IntegerType()))

def psudo_sort_by_day_plus(spark, df, num_days):
    # Sort is very expensive because it needs to calculate the partitions
    # which in our case may involve rereading all of the data.  In some cases
    # we can avoid this by repartitioning the data and sorting within a single partition
    shuffle_parts = int(spark.conf.get('spark.sql.shuffle.partitions'))
    extra_parts = int(shuffle_parts/num_days)
    if extra_parts <= 0:
        df = df.repartition('day')
    else:
        #We want to spread out the computation to about the same amount as shuffle_parts
        divided = (col('ordinal') / num_days).cast(LongType())
        extra_ident = divided % extra_parts
        df = df.repartition(col('day'), extra_ident)
    return df.sortWithinPartitions('day', 'ordinal')


def load_combined_model(spark, model_folder):
    path = os.path.join(model_folder, 'combined.parquet')
    return spark.read.parquet(path)


def save_combined_model(df, model_folder, mode=None):
    path = os.path.join(model_folder, 'combined.parquet')
    df.write.parquet(path, mode=mode)


def delete_combined_model(spark, model_folder):
    path = os.path.join(model_folder, 'combined.parquet')
    delete_data_source(spark, path)


def load_low_mem_partial_ids(spark, model_folder):
    path = os.path.join(model_folder, 'partial_ids.parquet')
    return spark.read.parquet(path)


def save_low_mem_partial_ids(df, model_folder, mode=None):
    path = os.path.join(model_folder, 'partial_ids.parquet')
    df.write.parquet(path, mode=mode)


def delete_low_mem_partial_ids(spark, model_folder):
    path = os.path.join(model_folder, 'partial_ids.parquet')
    delete_data_source(spark, path)


def load_column_models(spark, model_folder, count_required):
    for i in CAT_COLS:
        path = os.path.join(model_folder, '%d.parquet' % i)
        df = spark.read.parquet(path)
        if count_required:
            values = df.agg(sum('model_count').alias('sum'), count('*').alias('size')).collect()
        else:
            values = df.agg(sum('model_count').alias('sum')).collect()
        yield i, df, values[0], would_broadcast(spark, path)

def save_column_models(column_models, model_folder, mode=None):
    for i, model in column_models:
        path = os.path.join(model_folder, '%d.parquet' % i)
        model.write.parquet(path, mode=mode)


def save_model_size(model_size, path, write_mode):
    if os.path.exists(path) and write_mode == 'errorifexists':
        print('Error: model size file %s exists' % path)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as fp:
        json.dump(model_size, fp, indent=4)


_benchmark = {}


@contextmanager
def _timed(step):
    start = time()
    yield
    end = time()
    _benchmark[step] = end - start


def _parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--mode',
        required=True,
        choices=['generate_models', 'transform'])

    parser.add_argument('--days', required=True)
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--output_folder')
    parser.add_argument('--model_size_file')
    parser.add_argument('--model_folder', required=True)
    parser.add_argument(
        '--write_mode',
        choices=['overwrite', 'errorifexists'],
        default='errorifexists')

    parser.add_argument('--frequency_limit')
    parser.add_argument('--no_numeric_log_col', action='store_true')
    #Support for running in a lower memory environment
    parser.add_argument('--low_mem', action='store_true')
    parser.add_argument(
        '--output_ordering',
        choices=['total_random', 'day_random', 'any', 'input'],
        default='total_random')

    parser.add_argument(
        '--output_partitioning',
        choices=['day', 'none'],
        default='none')

    parser.add_argument('--dict_build_shuffle_parallel_per_day', type=int, default=2)
    parser.add_argument('--apply_shuffle_parallel_per_day', type=int, default=25)
    parser.add_argument('--skew_broadcast_pct', type=float, default=1.0)

    parser.add_argument('--debug_mode', action='store_true')

    args = parser.parse_args()

    start, end = args.days.split('-')
    args.day_range = list(range(int(start), int(end) + 1))
    args.days = len(args.day_range)

    return args


def _main():
    args = _parse_args()
    spark = SparkSession.builder.getOrCreate()

    df = load_raw(spark, args.input_folder, args.day_range)

    if args.mode == 'generate_models':
        spark.conf.set('spark.sql.shuffle.partitions', args.days * args.dict_build_shuffle_parallel_per_day)
        with _timed('generate models'):
            col_counts = get_column_counts_with_frequency_limit(df, args.frequency_limit)
            if args.low_mem:
                # in low memory mode we have to save an intermediate result
                # because if we try to do it in one query spark ends up assigning the
                # partial ids in two different locations that are not guaranteed to line up
                # this prevents that from happening by assigning the partial ids
                # and then writeing them out.
                save_low_mem_partial_ids(
                        assign_low_mem_partial_ids(col_counts),
                        args.model_folder,
                        args.write_mode)
                save_combined_model(
                        assign_low_mem_final_ids(load_low_mem_partial_ids(spark, args.model_folder)),
                        args.model_folder,
                        args.write_mode)
                if not args.debug_mode:
                    delete_low_mem_partial_ids(spark, args.model_folder)

            else:
                save_combined_model(
                        assign_id_with_window(col_counts),
                        args.model_folder,
                        args.write_mode)
            save_column_models(
                get_column_models(load_combined_model(spark, args.model_folder)),
                args.model_folder,
                args.write_mode)
            if not args.debug_mode:
                delete_combined_model(spark, args.model_folder)

    if args.mode == 'transform':
        spark.conf.set('spark.sql.shuffle.partitions', args.days * args.apply_shuffle_parallel_per_day)
        with _timed('transform'):
            if args.output_ordering == 'total_random':
                df = rand_ordinal(df)
                if args.output_partitioning == 'day':
                    df = day_from_ordinal(df, args.days)
            elif args.output_ordering == 'day_random':
                df = rand_ordinal(df)
                df = day_from_input_file(df)
            elif args.output_ordering == 'input':
                df = df.withColumn('ordinal', monotonically_increasing_id())
                if args.output_partitioning == 'day':
                    df = day_from_input_file(df)
            else: # any ordering
                if args.output_partitioning == 'day':
                    df = day_from_input_file(df)

            models = list(load_column_models(spark, args.model_folder, bool(args.model_size_file)))
            if args.model_size_file:
                save_model_size(
                    OrderedDict(('_c%d' % i, agg.size) for i, _, agg, _ in models),
                    args.model_size_file,
                    args.write_mode)
            models = [(i, df, agg.sum, flag) for i, df, agg, flag in models]

            df = apply_models(
                df,
                models,
                not args.low_mem,
                args.skew_broadcast_pct)
            df = transform_log(df, not args.no_numeric_log_col)


            if args.output_partitioning == 'day':
                partitionBy = 'day'
            else:
                partitionBy = None

            if args.output_ordering == 'total_random':
                if args.output_partitioning == 'day':
                    df = psudo_sort_by_day_plus(spark, df, args.days)
                else: # none
                    # Don't do a full sort it is expensive. Order is random so
                    # just make it random
                    df = df.repartition('ordinal').sortWithinPartitions('ordinal')

                df = df.drop('ordinal')
            elif args.output_ordering == 'day_random':
                df = psudo_sort_by_day_plus(spark, df, args.days)
                df = df.drop('ordinal')
                if args.output_partitioning != 'day':
                    df = df.drop('day')
            elif args.output_ordering == 'input':
                if args.low_mem:
                    # This is the slowest option. We totally messed up the order so we have to put
                    # it back in the correct order
                    df = df.orderBy('ordinal')
                else:
                    # Applying the dictionary happened within a single task so we are already really
                    # close to the correct order, just need to sort within the partition
                    df = df.sortWithinPartitions('ordinal')
                df = df.drop('ordinal')
                if args.output_partitioning != 'day':
                    df = df.drop('day')
            # else: any ordering so do nothing the ordering does not matter

            df.write.parquet(
                args.output_folder,
                mode=args.write_mode,
                partitionBy=partitionBy)

    print('=' * 100)
    print(_benchmark)


if __name__ == '__main__':
    _main()
