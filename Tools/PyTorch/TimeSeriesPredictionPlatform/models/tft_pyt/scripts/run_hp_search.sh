# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

: ${SEED:=1}
: ${LR:=1e-3}
: ${BATCH_SIZE:=1024}
: ${H_SIZE:=128}
: ${N_HEADS:=1}
: ${NGPU:=8}
: ${EPOCHS:=20}
: ${DATASET:="electricity"}
: ${DROPOUT:=0.1}
: ${PREC:="amp"}

SAMPLES=""
GNORMS=(0.01 0.1 1.0)
[ ${DATASET} = "volatility" ] || SAMPLES="--sample 450000 50000"
[ ${DATASET} = "traffic" ] && GNORMS=(0.1 1.0 10.0 100.0)
[ ${DATASET} = "favorita" ] && GNORMS=(0.1 1.0 10.0 100.0)

[ ${PREC} = "amp" ] && AMP="--use_amp"
[ ${PREC} = "fp32" ] || [ ${PREC} = "tf32" ] && AMP=""

for MAX_GNORM in "${GNORMS[@]}"
do
    for EMA_DECAY in 0.25 0.5 0.75 0.9 0.95
    do
        for SEED in {1..30}
        do
            EXP_NAME=TFT_${DATASET}_bs${NGPU}x${BATCH_SIZE}_HSIZE${H_SIZE}_NHEADS${N_HEADS}_LR${LR}_CLIP_GRAD${MAX_GNORM}_DROPOUT_${DROPOUT}_EMA${EMA_DECAY}_${PREC}
            
            for RETRY in {1..3}
            do
                python -m torch.distributed.run --nproc_per_node=${NGPU} train.py \
                        --dataset ${DATASET} \
                        --data_path /ws/datasets/${DATASET}_bin \
                        --batch_size=${BATCH_SIZE} \
                        --lr ${LR} \
                        --epochs ${EPOCHS} \
                        ${SAMPLES} \
                        --seed ${SEED} \
                        ${AMP} \
                        --clip_grad ${MAX_GNORM} \
                        --ema_decay ${EMA_DECAY} \
                        --overwrite_config "{\"dropout\":$DROPOUT, \"hidden_size\":$H_SIZE, \"n_head\":$N_HEADS}" \
                        --results /results/${EXP_NAME}/seed_${SEED}

                if [ -f /results/${EXP_NAME}/seed_${SEED}/dllogger.json ]
                then
                    LAST_LINE=$( tail -n 1 /results/${EXP_NAME}/seed_${SEED}/dllogger.json | grep  "\"step\": \[\]" )
                    [[ $LAST_LINE = "" ]] || break
                fi
                echo RETRYING ...
                rm -r /results/${EXP_NAME}/seed_${SEED}

            done
        done
    done
done
