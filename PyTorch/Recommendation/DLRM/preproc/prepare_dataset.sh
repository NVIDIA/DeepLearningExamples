#! /bin/bash

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

# Examples:
# to run on GPU with a frequency limit of 3 using NVTabular:
#   ./prepare_dataset.sh 3 GPU NVTabular
#
# to run on GPU with a frequency limit of 15 using Spark GPU:
#   ./prepare_dataset.sh 15 GPU Spark
#
# to run on CPU with a frequency limit of 15 using Spark CPU:
#   ./prepare_dataset.sh 15 CPU

set -e
set -x

ls -ltrash


rm -rf /data/dlrm/spark
rm -rf /data/dlrm/intermediate_binary
rm -rf /data/dlrm/output
rm -rf /data/dlrm/criteo_parquet
rm -rf /data/dlrm/binary_dataset


download_dir=${download_dir:-'/data/dlrm/criteo'}
./verify_criteo_downloaded.sh ${download_dir}

output_path=${output_path:-'/data/dlrm/output'}


if [ "$3" = "NVTabular" ]; then
    echo "Performing NVTabular preprocessing"
    ./run_NVTabular.sh ${download_dir} ${output_path} $1
    preprocessing_version=NVTabular
else
    if [ -f ${output_path}/train/_SUCCESS ] \
        && [ -f ${output_path}/validation/_SUCCESS ] \
        && [ -f ${output_path}/test/_SUCCESS ]; then

        echo "Spark preprocessing already carried out"
    else
        echo "Performing spark preprocessing"
        ./run_spark.sh $2 ${download_dir} ${output_path} $1
    fi
    preprocessing_version=Spark
fi

conversion_intermediate_dir=${conversion_intermediate_dir:-'/data/dlrm/intermediate_binary'}
final_output_dir=${final_output_dir:-'/data/dlrm/binary_dataset'}

source ${DGX_VERSION}_config.sh

if [ -d ${final_output_dir}/train ] \
   && [ -d ${final_output_dir}/validation ] \
   && [ -d ${final_output_dir}/test ] \
   && [ -f ${final_output_dir}/feature_spec.yaml ]; then

    echo "Final conversion already done"
else
    echo "Performing final conversion to a custom data format"
    python parquet_to_binary.py --parallel_jobs ${TOTAL_CORES} --src_dir ${output_path} \
                                --intermediate_dir  ${conversion_intermediate_dir} \
                                --dst_dir ${final_output_dir}

    cp "${output_path}/model_size.json" "${final_output_dir}/model_size.json"

    python split_dataset.py --dataset "${final_output_dir}" --output "${final_output_dir}/split"
    rm ${final_output_dir}/train_data.bin
    rm ${final_output_dir}/validation_data.bin
    rm ${final_output_dir}/test_data.bin
    rm ${final_output_dir}/model_size.json

    mv ${final_output_dir}/split/* ${final_output_dir}
    rm -rf ${final_output_dir}/split
fi

echo "Done preprocessing the Criteo Kaggle Dataset"
