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

#! /bin/bash

set -e
set -x

ls -ltrash

download_dir=${download_dir:-'/data/dlrm/criteo'}
./verify_criteo_downloaded.sh ${download_dir}

spark_output_path=${spark_output_path:-'/data/dlrm/spark/output'}


if [ -f ${spark_output_path}/train/_SUCCESS ] \
   && [ -f ${spark_output_path}/validation/_SUCCESS ] \
   && [ -f ${spark_output_path}/test/_SUCCESS ]; then

   echo "Spark preprocessing already carried done"
else
   echo "Performing spark preprocessing"
   ./run_spark.sh ${download_dir} ${spark_output_path}
fi

conversion_intermediate_dir=${conversion_intermediate_dir:-'/data/dlrm/intermediate_binary'}
final_output_dir=${final_output_dir:-'/data/dlrm/binary_dataset'}


if [ -f ${final_output_dir}/train_data.bin ] \
   && [ -f ${final_output_dir}/val_data.bin ] \
   && [ -f ${final_output_dir}/test_data.bin ] \
   && [ -f ${final_output_dir}/model_sizes.json ]; then

    echo "Final conversion already done"
else
    echo "Performing final conversion to a custom data format"
    python parquet_to_binary.py --parallel_jobs 40 --src_dir ${spark_output_path} \
                                --intermediate_dir  ${conversion_intermediate_dir} \
                                --dst_dir ${final_output_dir}

    cp "${spark_output_path}/model_size.json" "${final_output_dir}/model_size.json"
fi

echo "Done preprocessing the Criteo Kaggle Dataset"
echo "You can now start the training with: "
echo "python -m dlrm.scripts.main --mode train --dataset  /data/dlrm/binary_dataset/ --model_config dlrm/config/default.json"