#!/bin/bash

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

#########################################################################
# File Name: run_NVTabular.sh

set -e

# the data path including 1TB criteo data, day_0, day_1, ...
export INPUT_PATH=${1:-'/data/dlrm/criteo'}

# the output path, use for generating the dictionary and the final dataset
# the output folder should have more than 300GB
export OUTPUT_PATH=${2:-'/data/dlrm/output'}

export FREQUENCY_LIMIT=${3:-'15'}

export CRITEO_PARQUET=${4:-'/data/dlrm/criteo_parquet'}

if [ "$DGX_VERSION" = "DGX-2" ]; then
    export DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
else
    export DEVICES=0,1,2,3,4,5,6,7
fi

echo "Preprocessing data"
python preproc_NVTabular.py $INPUT_PATH $OUTPUT_PATH --devices $DEVICES --intermediate_dir $CRITEO_PARQUET --freq_threshold $FREQUENCY_LIMIT

echo "Shuffling"

source ${DGX_VERSION}_config.sh

export SPARK_HOME=/opt/spark
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
export MASTER=spark://$HOSTNAME:7077
export SPARK_LOCAL_DIRS='/data/dlrm/spark/tmp'
mkdir -p $SPARK_LOCAL_DIRS

echo "Starting spark standalone"
start-master.sh
start-slave.sh $MASTER

spark-submit --master $MASTER \
    --driver-memory "${DRIVER_MEMORY}G" \
    --executor-cores $NUM_EXECUTOR_CORES \
    --executor-memory "${EXECUTOR_MEMORY}G" \
    --conf spark.cores.max=$TOTAL_CORES \
    --conf spark.task.cpus=1 \
    --conf spark.sql.files.maxPartitionBytes=1073741824 \
    --conf spark.sql.shuffle.partitions=1200 \
    --conf spark.driver.maxResultSize=2G \
    --conf spark.locality.wait=0s \
    --conf spark.network.timeout=1800s \
    --conf spark.task.resource.gpu.amount=0.01 \
    --conf spark.executor.resource.gpu.amount=1 \
    --conf spark.plugins=com.nvidia.spark.SQLPlugin \
    --conf spark.rapids.sql.concurrentGpuTasks=2 \
    --conf spark.rapids.sql.reader.batchSizeRows=4000000 \
    --conf spark.rapids.memory.pinnedPool.size=16g \
    --conf spark.rapids.sql.explain=ALL \
    --conf spark.sql.autoBroadcastJoinThreshold=1GB \
    --conf spark.rapids.sql.incompatibleOps.enabled=true \
    --conf spark.driver.maxResultSize=2G \
    --conf spark.executor.extraJavaOptions="-Dai.rapids.cudf.prefer-pinned=true\ -Djava.io.tmpdir=$SPARK_LOCAL_DIRS" \
    NVT_shuffle_spark.py --input_path $OUTPUT_PATH/train --output_path $OUTPUT_PATH/shuffled_train

stop-master.sh
stop-slave.sh

rm -rf $OUTPUT_PATH/train
mv $OUTPUT_PATH/shuffled_train $OUTPUT_PATH/train



