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
# ==============================================================================

INIT_CKPT=${1}

if [ ! -f "$INIT_CKPT" ]; then
    echo "$INIT_CKPT does not exist. Cannot run inference without a valid checkpoint"
    exit -1
fi

PRED_BS=${2:-96}
NUM_GPU=${3:-8}
PRECISION=${4:-fp16}
EVAL_BEAMS=${5:-4}
MAX_SOURCE_LEN=${6:-1024}
MAX_TARGET_LEN=${7:-142}

DATA_DIR=${8:-data/cnn_dm}
CONFIG_PATH=${9:-"configs/config.json"}
PRELN=${10:-true}

printf -v TAG "bart_pyt_inference"
DATESTAMP=`date +'%y%m%d%H%M%S'`
RESULTS_DIR=${RESULTS_DIR:-results/${TAG}_${DATESTAMP}}
mkdir -p $RESULTS_DIR

if [ "$PRECISION" = "fp16" ] ; then
    echo "fp16 activated!"
    USE_FP16="--fp16"
elif [ "$PRECISION" = "bf16" ] ; then
    echo "bf16 activated!"
    USE_FP16="--bf16"
else
    echo "fp32/tf32 activated!"
    USE_FP16=""
fi

if [ "$PRELN" = "true" ] ; then
    echo "Use PreLN"
    USE_FP16="--pre_ln $USE_FP16"
else
    echo "Use PostLN"
fi

python -m torch.distributed.launch --nproc_per_node=$NUM_GPU run_eval.py \
    --task summarization \
    --bs ${PRED_BS} --max_source_length=${MAX_SOURCE_LEN} --max_target_length=${MAX_TARGET_LEN} \
    --eval_max_gen_length=${MAX_TARGET_LEN} --eval_beams=${EVAL_BEAMS} ${USE_FP16} \
    ${INIT_CKPT} ${CONFIG_PATH} ${DATA_DIR} ${RESULTS_DIR} |& tee -a ${RESULTS_DIR}/joblog.log
