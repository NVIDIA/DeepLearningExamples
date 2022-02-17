#!/bin/bash
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DATAPATH='/data'

declare -A URLS=( ['electricity']='https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'
                  ['traffic']='https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip'
                )

mkdir -p ${DATAPATH}/raw
mkdir -p ${DATAPATH}/processed

for DS in electricity traffic
do
	DS_PATH=${DATAPATH}/raw/${DS}
	ZIP_FNAME=${DS_PATH}.zip
    if [ ! -d ${DS_PATH} ]
    then
        wget "${URLS[${DS}]}" -O ${ZIP_FNAME}
        unzip ${ZIP_FNAME} -d ${DS_PATH}
    fi
	python -c "from data_utils import standarize_${DS} as standarize; standarize(\"${DS_PATH}\")"
	python -c "from data_utils import preprocess; \
               from configuration import ${DS^}Config as Config; \
               preprocess(\"${DS_PATH}/standarized.csv\", \"${DATAPATH}/processed/${DS}_bin\", Config())" 
done


