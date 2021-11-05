 # Copyright (c) 2018, deepakn94, robieta. All rights reserved.
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
#
# -----------------------------------------------------------------------
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


#!/bin/bash
set -e
set -x

DATASET_NAME=${1:-'ml-20m'}
RAW_DATADIR=${2:-"/data/${DATASET_NAME}"}
CACHED_DATADIR=${3:-"$/data/cache/${DATASET_NAME}"}

# you can add another option to this case in order to support other datasets
case ${DATASET_NAME} in
    'ml-20m')
	ZIP_PATH=${RAW_DATADIR}/'ml-20m.zip'
	RATINGS_PATH=${RAW_DATADIR}'/ml-20m/ratings.csv'
	;;
    'ml-1m')
	ZIP_PATH=${RAW_DATADIR}/'ml-1m.zip'
	RATINGS_PATH=${RAW_DATADIR}'/ml-1m/ratings.dat'
	;;
	*)
	echo "Unsupported dataset name: $DATASET_NAME"
	exit 1
esac

if [ ! -d ${RAW_DATADIR} ]; then
    mkdir -p ${RAW_DATADIR}
fi

if [ ! -d ${CACHED_DATADIR} ]; then
    mkdir -p ${CACHED_DATADIR}
fi

if [ -f log ]; then
    rm -f log
fi

if [ ! -f ${ZIP_PATH} ]; then
    echo "Dataset not found. Please download it from: https://grouplens.org/datasets/movielens/20m/ and put it in ${ZIP_PATH}"
    exit 1
fi

if [ ! -f ${RATINGS_PATH} ]; then
    unzip -u ${ZIP_PATH}  -d ${RAW_DATADIR}
fi

for test_name in more_pos less_pos less_user less_item more_user more_item other_names;
do
    NEW_DIR=${CACHED_DATADIR}/${test_name}

    if [ ! -d ${NEW_DIR} ]; then
    mkdir -p ${NEW_DIR}
    fi

    python convert_test.py --path ${RATINGS_PATH} --output $NEW_DIR --test ${test_name}
    echo "Generated testing for $test_name"
done

for test_sample in '0' '10' '200';
do
    NEW_DIR=${CACHED_DATADIR}/sample_${test_name}

    if [ ! -d ${NEW_DIR} ]; then
    mkdir -p ${NEW_DIR}
    fi

    python convert_test.py --path ${RATINGS_PATH} --output $NEW_DIR --valid_negative $test_sample
    echo "Generated testing for $test_name"
done

echo "Dataset $DATASET_NAME successfully prepared at: $CACHED_DATADIR"
echo "You can now run the training with: python -m torch.distributed.launch --nproc_per_node=<number_of_GPUs> --use_env ncf.py --data ${CACHED_DATADIR}"


