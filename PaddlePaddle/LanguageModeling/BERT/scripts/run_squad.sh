# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

echo "Container nvidia build = " $NVIDIA_BUILD_ID

init_checkpoint=${1:-"checkpoints/squad"}
epochs=${2:-"2"}
batch_size=${3:-"32"}
learning_rate=${4:-"4.6e-5"}
warmup_proportion=${5:-"0.2"}
precision=${6:-"amp"}
num_gpus=${7:-"8"}
seed=${8:-"1"}
squad_dir=${9:-"$BERT_PREP_WORKING_DIR/download/squad/v1.1"}
vocab_file=${10:-"vocab/bert-large-uncased-vocab.txt"}
OUT_DIR=${11:-"/results"}
mode=${12:-"train_eval"}
CONFIG_FILE=${13:-"None"}
max_steps=${14:-"-1"} 
enable_benchmark=${15:-"false"}
benchmark_steps=${16:-"100"}
benchmark_warmup_steps=${17:-"100"}
fuse_mha=${18:-"true"}


echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

amp=""
FUSE_MHA=""
if [ "$precision" = "amp" ] ; then
  echo "amp activated!"
  amp=" --amp --use-dynamic-loss-scaling --scale-loss=128.0"
  if [ "$fuse_mha" = "true" ] ; then
    FUSE_MHA="--fuse-mha"
  fi
fi

CONFIG=""
if [ "$CONFIG_FILE" != "None" ] ; then
  CONFIG="--config-file=$CONFIG_FILE"
fi

BENCH=""
if [ "$enable_benchmark" = "true" ] ; then
  BENCH="--benchmark --benchmark-steps=$benchmark_steps --benchmark-warmup-steps=$benchmark_warmup_steps"
fi

unset CUDA_VISIBLE_DEVICES
if [ "$num_gpus" = "1" ] ; then
  CMD="python -m paddle.distributed.launch --gpus=0"
elif [ "$num_gpus" = "2" ] ; then
  CMD="python -m paddle.distributed.launch --gpus=0,1"
elif [ "$num_gpus" = "3" ] ; then
  CMD="python -m paddle.distributed.launch --gpus=0,1,2"
elif [ "$num_gpus" = "4" ] ; then
  CMD="python -m paddle.distributed.launch --gpus=0,1,2,3"
elif [ "$num_gpus" = "5" ] ; then
  CMD="python -m paddle.distributed.launch --gpus=0,1,2,3,4"
elif [ "$num_gpus" = "6" ] ; then
  CMD="python -m paddle.distributed.launch --gpus=0,1,2,3,4,5"
elif [ "$num_gpus" = "7" ] ; then
  CMD="python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6"
elif [ "$num_gpus" = "8" ] ; then
  CMD="python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7"
else
  echo "Wrong number of gpus"
  exit 2
fi

CMD+=" run_squad.py "
CMD+="--from-pretrained-params=$init_checkpoint "
if [ "$mode" = "train" ] ; then
  CMD+="--do-train "
  CMD+="--train-file=$squad_dir/train-v1.1.json "
  CMD+="--train-batch-size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  CMD+="--do-predict "
  CMD+="--predict-file=$squad_dir/dev-v1.1.json "
  CMD+="--predict-batch-size=$batch_size "
  CMD+="--eval-script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do-eval "
elif [ "$mode" = "prediction" ] ; then
  CMD+="--do-predict "
  CMD+="--predict-file=$squad_dir/dev-v1.1.json "
  CMD+="--predict-batch-size=$batch_size "
else
  CMD+=" --do-train "
  CMD+=" --train-file=$squad_dir/train-v1.1.json "
  CMD+=" --train-batch-size=$batch_size "
  CMD+="--do-predict "
  CMD+="--predict-file=$squad_dir/dev-v1.1.json "
  CMD+="--predict-batch-size=$batch_size "
  CMD+="--eval-script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do-eval "
fi

CMD+=" --do-lower-case "
CMD+=" --bert-model=bert-large-uncased "
CMD+=" --learning-rate=$learning_rate "
CMD+=" --seed=$seed "
CMD+=" --epochs=$epochs "
CMD+=" --max-seq-length=384 "
CMD+=" --doc-stride=128 "
CMD+=" --output-dir=$OUT_DIR "
CMD+=" --vocab-file=$vocab_file "
CMD+=" $CONFIG "
CMD+=" --max-steps=$max_steps "
CMD+=" --optimizer=AdamW "
CMD+=" --log-freq=100 "
CMD+=" $amp "
CMD+=" $FUSE_MHA "
CMD+=" $BENCH "
CMD+=" --report-file $OUT_DIR/dllogger_${num_gpus}_${precision}.json "

LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE
