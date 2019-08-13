#!/bin/bash

## DL params
BATCHSIZE=32
LEARNING_RATE="4e-3"
WARMUP_UPDATES=0.128
EXTRA_PARAMS="--input_dir=/workspace/data_phase2 --do_train --config_file=bert_config.json --max_seq_length=512 --max_predictions_per_seq=80 --output_dir /checkpoints --max_steps=1563 --num_steps_per_checkpoint=1000 --log_freq=1 --phase2 --gradient_accumulation_steps=8 --resume_from_checkpoint"

## System run parms
DGXNNODES=64
DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=04:00:00
DEADLINE=$(date -d '+168 hours' '+%FT%T')
SLURM_EMAIL_TYPE="END"

## System config params
DGXNGPU=16
DGXSOCKETCORES=24
DGXNSOCKET=2
DGXHT=2 	# HT is on is 2, HT off is 1
DGXIBDEVICES='--device=/dev/infiniband/rdma_cm --device=/dev/infiniband/ucm10 --device=/dev/infiniband/ucm9 --device=/dev/infiniband/ucm8 --device=/dev/infiniband/ucm7 --device=/dev/infiniband/ucm4 --device=/dev/infiniband/ucm3 --device=/dev/infiniband/ucm2 --device=/dev/infiniband/ucm1 --device=/dev/infiniband/uverbs10 --device=/dev/infiniband/uverbs9 --device=/dev/infiniband/uverbs8 --device=/dev/infiniband/uverbs7 --device=/dev/infiniband/uverbs4 --device=/dev/infiniband/uverbs3 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/issm10 --device=/dev/infiniband/umad10 --device=/dev/infiniband/issm9 --device=/dev/infiniband/umad9 --device=/dev/infiniband/issm8 --device=/dev/infiniband/umad8 --device=/dev/infiniband/issm7 --device=/dev/infiniband/umad7 --device=/dev/infiniband/issm4 --device=/dev/infiniband/umad4 --device=/dev/infiniband/issm3 --device=/dev/infiniband/umad3 --device=/dev/infiniband/issm2 --device=/dev/infiniband/umad2 --device=/dev/infiniband/issm1 --device=/dev/infiniband/umad1'