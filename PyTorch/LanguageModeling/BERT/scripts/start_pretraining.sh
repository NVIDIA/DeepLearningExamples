#!/bin/bash
# purpose: for multinode training on slurm clusters
node_type=${1:-"dgx1"}
num_nodes=${2:-1}
partition=${3:-"default"}
wall_time=${4:-"12:00:00"}
job_name=${5:-"pyt_bert"}
root_dir=${6:-"$PWD"}
train_batch_size=${7:-4}
eval_batch_size=${8:-4}
train_steps=${9:-1000000}
warmup_proportion=${10:-0.01}
learning_rate=${11:-1e-4}
precision=${12:-"fp16"}
save_checkpoint_steps=${13:-5000}
results_dir=${14:-"$root_dir/results"}
checkpoints_dir=${15:-"$root_dir/checkpoints"}

CONT=${CONT:-"gitlab-master.nvidia.com:5005/dl/dgx/pytorch:19.02-py3-devel"}

BENCHMARK=${BENCHMARK:-"bert"}
BENCHMARK_NAME="bert"

if [ "$node_type" = "dgx1" ] ; then
   echo "Running on dgx1 systems"
   DGXSYSTEM="DGX1"
   DGXNGPU=8
   DGXSOCKETCORES=20
   DGXNSOCKET=2
   DGXHT=2
   DGXIBDEVICES='--device=/dev/infiniband --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/ucm3 --device=/dev/infiniband/ucm2 --device=/dev/infiniband/ucm1 --device=/dev/infiniband/ucm0 --device=/dev/infiniband/uverbs3 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/issm3 --device=/dev/infiniband/umad3 --device=/dev/infiniband/issm2 --device=/dev/infiniband/umad2 --device=/dev/infiniband/issm1 --device=/dev/infiniband/umad1 --device=/dev/infiniband/issm0 --device=/dev/infiniband/umad0'
elif [ "$node_type" = "dgx2h" ] ; then
   echo "Running on dgx2h systems"
   DGXSYSTEM="DGX2H"
   DGXNGPU=16
   DGXSOCKETCORES=24
   DGXNSOCKET=2
   DGXHT=2         # HT is on is 2, HT off is 1
   DGXIBDEVICES='--device=/dev/infiniband/rdma_cm --device=/dev/infiniband/ucm10 --device=/dev/infiniband/ucm9 --device=/dev/infiniband/ucm8 --device=/dev/infiniband/ucm7 --device=/dev/infiniband/ucm4 --device=/dev/infiniband/ucm3 --device=/dev/infiniband/ucm2 --device=/dev/infiniband/ucm1 --device=/dev/infiniband/uverbs10 --device=/dev/infiniband/uverbs9 --device=/dev/infiniband/uverbs8 --device=/dev/infiniband/uverbs7 --device=/dev/infiniband/uverbs4 --device=/dev/infiniband/uverbs3 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/issm10 --device=/dev/infiniband/umad10 --device=/dev/infiniband/issm9 --device=/dev/infiniband/umad9 --device=/dev/infiniband/issm8 --device=/dev/infiniband/umad8 --device=/dev/infiniband/issm7 --device=/dev/infiniband/umad7 --device=/dev/infiniband/issm4 --device=/dev/infiniband/umad4 --device=/dev/infiniband/issm3 --device=/dev/infiniband/umad3 --device=/dev/infiniband/issm2 --device=/dev/infiniband/umad2 --device=/dev/infiniband/issm1 --device=/dev/infiniband/umad1'
else
   echo "Unknown <node_type>, must be either dgx1 or dgx2"
   exit -1
fi

printf -v EXTRA_PARAMS "%d %d %e %s 1 %d %d %d false" $train_batch_size $eval_batch_size $learning_rate "$precision" $warmup_proportion $train_steps $save_checkpoint_steps

export ROOTDIR=$root_dir
export DATA_DIR=${DATA_DIR:-$CODEDIR/data/hdf5/books_wiki_en_corpus}

VOLS="-v $ROOTDIR:/workspace/bert"
VOLS+=" -v $DATA_DIR:/workspace/bert/data/wikipedia_corpus/pyt_hdf5_shards"
# VOLS+=" -v $BOOKS_DIR:/workspace/bert/data/bookcorpus/final_tfrecord_sharded"
VOLS+=" -v $results_dir:/results"
VOLS+=" -v $checkpoints_dir:/checkpoints"

export VOLS
export CONT
export DGXSYSTEM
export DGXNGPU
export DGXIBDEVICES
export EXTRA_PARAMS

set -x
cd $CODEDIR
pwd

PART=""
if [ "$partition" != "default" ] ; then
   printf -v PART "%s" "-p $partition"
fi

export GBS=$(expr $num_nodes \* $batch_size \* $DGXNGPU)
printf -v TAG "%s_%dn_%s_gbs%d" "$job_name" $num_nodes "$precision" $GBS
export DATESTAMP=`date +'%y%m%d%H%M%S'`

sbatch $PART \
        -N $num_nodes \
        -t $wall_time \
        -J $job_name \
        --exclusive \
        --mem=0 \
        --mail-type=FAIL \
        --ntasks-per-node=$DGXNGPU \
        --threads-per-core=$DGXHT \
        --cores-per-socket=$DGXSOCKETCORES \
        --output=$LOGDIR/$TAG.$DATESTAMP.log \
        $CODEDIR/scripts/run.sub
set +x

