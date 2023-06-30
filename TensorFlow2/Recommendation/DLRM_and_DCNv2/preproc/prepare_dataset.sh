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
# to run on a DGX2 with a frequency limit of 3 (will need 8xV100-32GB to fit the model in GPU memory)
# ./prepare_dataset.sh DGX2 3
#
# to run on a DGX2 with a frequency limit of 15 (should fit on a single V100-32GB):
# ./prepare_dataset.sh DGX2 15
#
# to run on CPU with a frequency limit of 15:
# ./prepare_dataset.sh CPU 15



set -e
set -x

ls -ltrash

download_dir=${download_dir:-'/data/criteo_orig'}
./verify_criteo_downloaded.sh ${download_dir}

spark_output_path=${spark_output_path:-'/data/spark/output'}


if [ -f ${spark_output_path}/train/_SUCCESS ] \
   && [ -f ${spark_output_path}/validation/_SUCCESS ] \
   && [ -f ${spark_output_path}/test/_SUCCESS ]; then

   echo "Spark preprocessing already carried out"
else
   echo "Performing spark preprocessing"
   ./run_spark.sh $1 ${download_dir} ${spark_output_path} $2
fi

conversion_intermediate_dir=${conversion_intermediate_dir:-'/data/intermediate_binary'}
final_output_dir=${final_output_dir:-'/data/preprocessed'}


if [ -d ${final_output_dir}/train ] \
   && [ -d ${final_output_dir}/validation ] \
   && [ -d ${final_output_dir}/test ] \
   && [ -f ${final_output_dir}/feature_spec.yaml ]; then

    echo "Final conversion already done"
else
    echo "Performing final conversion to a custom data format"
    python parquet_to_binary.py --parallel_jobs 40 --src_dir ${spark_output_path} \
                                --intermediate_dir  ${conversion_intermediate_dir} \
                                --dst_dir ${final_output_dir}

    cp "${spark_output_path}/model_size.json" "${final_output_dir}/model_size.json"

    python split_dataset.py --dataset "${final_output_dir}" --output "${final_output_dir}/split"
    rm ${final_output_dir}/train_data.bin
    rm ${final_output_dir}/validation_data.bin
    rm ${final_output_dir}/test_data.bin
    rm ${final_output_dir}/model_size.json

    mv ${final_output_dir}/split/* ${final_output_dir}
    rm -rf ${final_output_dir}/split
fi

echo "Done preprocessing the Criteo Kaggle Dataset"
