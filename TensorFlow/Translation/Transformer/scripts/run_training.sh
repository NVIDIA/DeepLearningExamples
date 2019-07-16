#!/bin/bash

set -e

NUM_GPU=${1:-"8"}
LR=${2:-1.0}
STEPS=${3:-60000}
WARMUP_STEPS=${4:-6000}
PRECISION=${5:-"fp32"}
ENABLE_XLA=${6:-"false"}
QUALITY=${7:-28.0}
SEED=${8:-1}
cd /research/transformer

results_dir="/results"
checkpoints_dir="/results"

enable_amp_tag=""
if [ "$PRECISION" = "fp16" ] ; then
        echo "AMP activated!"
        #export TF_ENABLE_AUTO_MIXED_PRECISION=1
        export TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=1
        enable_amp_tag="--enable_amp"
fi
enable_xla_tag=""
if [ "$ENABLE_XLA" = "true" ] ; then
    enable_xla_tag="--enable_xla"
    echo "XLA activated"
fi


export PYTHONPATH=/research/transformer/transformer:${PYTHONPATH}
# Add compliance to PYTHONPATH
# export PYTHONPATH=/mlperf/training/compliance:${PYTHONPATH}

mpiexec --allow-run-as-root --bind-to socket -np $NUM_GPU \
   python3 transformer/transformer_main.py \
   --random_seed=${SEED} \
   --data_dir=data/processed_data/ \
   --model_dir=${checkpoints_dir} \
   --enable_horovod \
   --params=big \
   --bleu_threshold ${QUALITY} \
   --bleu_source=data/newstest2014.en \
   --bleu_ref=data/newstest2014.de \
   --train_steps=${STEPS} \
   --steps_between_eval=5000 \
   --warmup_steps ${WARMUP_STEPS} \
   --learning_rate=${LR} \
   --sentencepiece \
   --report_loss \
   $enable_amp_tag $enable_xla_tag |& tee /results/output_${NUM_GPU}_${LR}_${WARMUP_STEPS}_$(date +'%m-%d-%H:%M').log
