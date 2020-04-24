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
# File Name: run-spark.sh

#!/bin/bash

set -e

# the environment variables to run spark job
# should modify below environment variables

# the data path including 1TB criteo data, day_0, day_1, ...
export INPUT_PATH=${1:-'/data/dlrm/criteo'}

# the output path, use for generating the dictionary and the final dataset
# the output folder should have more than 300GB
export OUTPUT_PATH=${2:-'/data/dlrm/spark/output'}

# spark local dir should have about 3TB
# the temporary path used for spark shuffle write
export SPARK_LOCAL_DIRS='/data/dlrm/spark/tmp'

# below numbers should be adjusted according to the resource of your running environment
# set the total number of CPU cores, spark can use
export TOTAL_CORES=80

# set the number of executors
export NUM_EXECUTORS=8

# the cores for each executor, it'll be calculated
export NUM_EXECUTOR_CORES=$((${TOTAL_CORES}/${NUM_EXECUTORS}))

# unit: GB,  set the max memory you want to use
export TOTAL_MEMORY=800

# unit: GB, set the memory for driver
export DRIVER_MEMORY=32

# the memory per executor
export EXECUTOR_MEMORY=$(((${TOTAL_MEMORY}-${DRIVER_MEMORY})/${NUM_EXECUTORS}))

# use frequency_limit=15 or not
# by default use a frequency limit of 15
USE_FREQUENCY_LIMIT=1
OPTS=""
if [[ $USE_FREQUENCY_LIMIT == 1 ]]; then
    OPTS="--frequency_limit 15"
fi

export SPARK_HOME=/opt/spark-2.4.5-bin-hadoop2.7
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
    	--conf spark.task.cpus=1 \
        --conf spark.sql.files.maxPartitionBytes=1073741824 \
    	--conf spark.sql.shuffle.partitions=600 \
    	--conf spark.driver.maxResultSize=2G \
    	--conf spark.locality.wait=0s \
    	--conf spark.network.timeout=1800s \
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
    	spark_data_utils.py --mode transform \
    	--input_folder $temp_validation \
    	--days 23-23 \
    	--output_folder $OUTPUT_PATH/validation \
    	--output_ordering input \
    	--model_folder $OUTPUT_PATH/models \
    	--write_mode overwrite --low_mem 2>&1 | tee submit_validation_log.txt

rm -r $temp_test $temp_validation
