#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#### input arguments
ARCH=${ARCH:-75}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/checkpoints}
CHECKPOINT=${CHECKPOINT:-"jasper_fp16.pt"}
PRECISION=${PRECISION:-fp16}
MAX_SEQUENCE_LENGTH_FOR_ENGINE=${MAX_SEQUENCE_LENGTH_FOR_ENGINE:-3600}
GPU=${GPU:-0}
####

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=${SCRIPT_DIR}/../..
if [ -f /.dockerenv ]; then # inside docker
	CUDA_VISIBLE_DEVICES=${GPU} CHECKPOINT=${CHECKPOINT} CHECKPOINT_DIR=${CHECKPOINT_DIR} PRECISION=${PRECISION} ARCH=${ARCH} MAX_SEQUENCE_LENGTH_FOR_ENGINE=${MAX_SEQUENCE_LENGTH_FOR_ENGINE} ${PROJECT_DIR}/trtis/scripts/export_model_helper.sh || exit 1
else
	set -x
	PROGRAM_PATH="/jasper/trtis/scripts/export_model_helper.sh"  \
	EXTRA_JASPER_ENV="-e PRECISION=${PRECISION} -e CHECKPOINT=${CHECKPOINT} -e CHECKPOINT_DIR=/checkpoints -e ARCH=${ARCH} -e MAX_SEQUENCE_LENGTH_FOR_ENGINE=${MAX_SEQUENCE_LENGTH_FOR_ENGINE} -e CUDA_VISIBLE_DEVICES=${GPU}" \
	CHECKPOINT_DIR=${CHECKPOINT_DIR} DATA_DIR= RESULT_DIR= \
	${PROJECT_DIR}/trtis/scripts/docker/launch.sh || exit 1
	set +x
fi
