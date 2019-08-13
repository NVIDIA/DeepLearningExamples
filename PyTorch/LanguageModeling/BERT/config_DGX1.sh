#!/bin/bash

## DL params
BATCHSIZE=16
LEARNING_RATE=6e-3
WARMUP_UPDATES=0.2843
EXTRA_PARAMS="--input_dir=workspace/data --do_train --config_file=bert_config.json --max_seq_length=128 --max_predictions_per_seq=20 --output_dir /results --fp16 --max_steps=7508 --num_steps_per_checkpoint=200"

## System run parms
DGXNNODES=1
DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=00:15:00
DEADLINE=$(date -d '+168 hours' '+%FT%T')
SLURM_EMAIL_TYPE="END"

## System config params
DGXNGPU=8
DGXSOCKETCORES=20
DGXNSOCKET=2
DGXHT=2         # HT is on is 2, HT off is 1
DGXIBDEVICES=''