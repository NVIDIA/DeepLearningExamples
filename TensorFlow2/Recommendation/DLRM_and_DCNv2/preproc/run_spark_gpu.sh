#!/bin/bash

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

#########################################################################
# File Name: run_spark_gpu.sh

set -e

# the data path including 1TB criteo data, day_0, day_1, ...
export INPUT_PATH=${1:-'/data/dlrm/criteo'}

# the output path, use for generating the dictionary and the final dataset
# the output folder should have more than 300GB
export OUTPUT_PATH=${2:-'/data/dlrm/spark/output'}

export FREQUENCY_LIMIT=${3:-'15'}

export HARDWARE_PLATFORM=${4:-'DGX2'}

# spark local dir should have about 3TB
# the temporary path used for spark shuffle write
export SPARK_LOCAL_DIRS='/data/dlrm/spark/tmp'

if [[ $HARDWARE_PLATFORM == DGX2 ]]; then
    source dgx2_config.sh
else
    echo "Unknown hardware platform ${HARDWARE_PLATFORM}"
    exit 1
fi

OPTS="--frequency_limit $FREQUENCY_LIMIT"

export SPARK_HOME=/opt/spark
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

# we use spark standalone to run the job
export MASTER=spark://$HOSTNAME:7077

echo "Starting spark standalone"
start-master.sh
start-slave.sh $MASTER

echo "Generating the dictionary..."
spark-submit --master $MASTER \
    	--driver-memory "${DRIVER_MEMORY}G" \
    	--executor-cores $NUM_EXECUTOR_CORES \
    	--executor-memory "${EXECUTOR_MEMORY}G" \
    	--conf spark.cores.max=$TOTAL_CORES \
        --conf spark.task.cpus=1 \
        --conf spark.sql.files.maxPartitionBytes=1073741824 \
    	--conf spark.sql.shuffle.partitions=600 \
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
        --conf spark.executor.extraJavaOptions="-Dcom.nvidia.cudf.prefer-pinned=true\ -Djava.io.tmpdir=$SPARK_LOCAL_DIRS" \
    	spark_data_utils.py --mode generate_models \
    	$OPTS \
    	--input_folder $INPUT_PATH \
    	--days 0-23 \
    	--model_folder $OUTPUT_PATH/models \
    	--write_mode overwrite --low_mem 2>&1 | tee submit_dict_log.txt

echo "Transforming the train data from day_0 to day_22..."
spark-submit --master $MASTER \
    	--driver-memory "${DRIVER_MEMORY}G" \
    	--executor-cores $NUM_EXECUTOR_CORES \
    	--executor-memory "${EXECUTOR_MEMORY}G" \
    	--conf spark.cores.max=$TOTAL_CORES \
        --conf spark.task.cpus=3 \
        --conf spark.sql.files.maxPartitionBytes=1073741824 \
    	--conf spark.sql.shuffle.partitions=600 \
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
        --conf spark.executor.extraJavaOptions="-Dcom.nvidia.cudf.prefer-pinned=true\ -Djava.io.tmpdir=$SPARK_LOCAL_DIRS" \
    	spark_data_utils.py --mode transform \
    	--input_folder $INPUT_PATH \
    	--days 0-22 \
    	--output_folder $OUTPUT_PATH/train \
        --model_size_file $OUTPUT_PATH/model_size.json \
    	--model_folder $OUTPUT_PATH/models \
    	--write_mode overwrite --low_mem 2>&1 | tee submit_train_log.txt

echo "Splitting the last day into 2 parts of test and validation..."
last_day=$INPUT_PATH/day_23
temp_test=$OUTPUT_PATH/temp/test
temp_validation=$OUTPUT_PATH/temp/validation
mkdir -p $temp_test $temp_validation

lines=`wc -l $last_day | awk '{print $1}'`
former=$((lines / 2))
latter=$((lines - former))

head -n $former $last_day > $temp_test/day_23
tail -n $latter $last_day > $temp_validation/day_23

echo "Transforming the test data in day_23..."
spark-submit --master $MASTER \
    	--driver-memory "${DRIVER_MEMORY}G" \
    	--executor-cores $NUM_EXECUTOR_CORES \
    	--executor-memory "${EXECUTOR_MEMORY}G" \
    	--conf spark.cores.max=$TOTAL_CORES \
        --conf spark.task.cpus=1 \
        --conf spark.sql.files.maxPartitionBytes=1073741824 \
    	--conf spark.sql.shuffle.partitions=30 \
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
        --conf spark.executor.extraJavaOptions="-Dcom.nvidia.cudf.prefer-pinned=true\ -Djava.io.tmpdir=$SPARK_LOCAL_DIRS" \
    	spark_data_utils.py --mode transform \
    	--input_folder $temp_test \
    	--days 23-23 \
    	--output_folder $OUTPUT_PATH/test \
    	--output_ordering input \
    	--model_folder $OUTPUT_PATH/models \
    	--write_mode overwrite --low_mem 2>&1 | tee submit_test_log.txt

echo "Transforming the validation data in day_23..."
spark-submit --master $MASTER \
    	--driver-memory "${DRIVER_MEMORY}G" \
    	--executor-cores $NUM_EXECUTOR_CORES \
    	--executor-memory "${EXECUTOR_MEMORY}G" \
    	--conf spark.cores.max=$TOTAL_CORES \
        --conf spark.task.cpus=1 \
        --conf spark.sql.files.maxPartitionBytes=1073741824 \
    	--conf spark.sql.shuffle.partitions=30 \
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
        --conf spark.executor.extraJavaOptions="-Dcom.nvidia.cudf.prefer-pinned=true\ -Djava.io.tmpdir=$SPARK_LOCAL_DIRS" \
    	spark_data_utils.py --mode transform \
    	--input_folder $temp_validation \
    	--days 23-23 \
    	--output_folder $OUTPUT_PATH/validation \
    	--output_ordering input \
    	--model_folder $OUTPUT_PATH/models \
    	--write_mode overwrite --low_mem 2>&1 | tee submit_validation_log.txt

rm -r $temp_test $temp_validation
stop-master.sh
stop-slave.sh
