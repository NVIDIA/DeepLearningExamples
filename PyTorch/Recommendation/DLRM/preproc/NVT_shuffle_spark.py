# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
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


from argparse import ArgumentParser

from pyspark.sql import Row, SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *


LABEL_COL = 0
INT_COLS = list(range(1, 14))
CAT_COLS = list(range(14, 40))


def col_of_rand_long():
    return (rand() * (1 << 52)).cast(LongType())

def rand_ordinal(df):
    return df.withColumn('ordinal', col_of_rand_long())

def _parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path')
    args = parser.parse_args()
    return args


def _main():
    args = _parse_args()
    spark = SparkSession.builder.getOrCreate()

    df = rand_ordinal(spark.read.load(args.input_path + "/*"))
    df = df.repartition('ordinal').sortWithinPartitions('ordinal')
    df = df.drop('ordinal')

    df.write.parquet(
        args.output_path,
        mode='overwrite'
    ) 
    

if __name__ == '__main__':
    _main()
