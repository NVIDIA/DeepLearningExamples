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
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
RAW_DATADIR=${2:-'/data'}
CACHED_DATADIR=${3:-"${RAW_DATADIR}/cache/${DATASET_NAME}"}

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
    echo 'Dataset not found, downloading...'
    ./download_dataset.sh ${DATASET_NAME} ${RAW_DATADIR}
fi

if [ ! -f ${RATINGS_PATH} ]; then
    unzip -u ${ZIP_PATH}  -d ${RAW_DATADIR}
fi

if [ ! -f ${CACHED_DATADIR}/train_ratings.pt ]; then
    echo "preprocessing ${RATINGS_PATH} and save to disk"
    t0=$(date +%s)
    python convert.py --path ${RATINGS_PATH} --output ${CACHED_DATADIR}
    t1=$(date +%s)
    delta=$(( $t1 - $t0 ))
    echo "Finish preprocessing in $delta seconds"
else
    echo 'Using cached preprocessed data'
fi

echo "Dataset $DATASET_NAME successfully prepared at: $CACHED_DATADIR\n"
echo "You can now run the training with: python -m torch.distributed.launch --nproc_per_node=<number_of_GPUs> --use_env ncf.py --data ${CACHED_DATADIR}"


