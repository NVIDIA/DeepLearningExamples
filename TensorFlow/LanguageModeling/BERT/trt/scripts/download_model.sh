#!/bin/bash

# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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


# Setup default parameters (if no command-line parameters given)
MODEL='large'
FT_PRECISION='fp16'
SEQ_LEN='128'

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname ${SCRIPT})
TENSORRT_DIR=${SCRIPT_DIR}/../../../

while test $# -gt 0
do
    case "$1" in
        -h) echo "Usage: sh download_model.sh [base|large] [fp16|fp32] [128|384]"
            exit 0
            ;;
        base) MODEL='base'
            ;;
        large) MODEL='large'
            ;;
        fp16) FT_PRECISION='fp16'
            ;;
        fp32) FT_PRECISION='fp32'
            ;;
        128) SEQ_LEN='128'
            ;;
        384) SEQ_LEN='384'
            ;;
        *) echo "Invalid argument $1...exiting"
            exit 0
            ;;
    esac
    shift
done

# Download the BERT fine-tuned model
echo "Downloading BERT-${MODEL} with fine-tuned precision ${FT_PRECISION} and sequence length ${SEQ_LEN} from NGC"
mkdir -p /workspace/bert/models/fine-tuned
cd /workspace/bert/models/fine-tuned
ngc registry model download-version nvidia/bert_tf_v2_${MODEL}_${FT_PRECISION}_${SEQ_LEN}:2
