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

dgxa100-80g_8gpu_amp ()
{
    init_checkpoint="results/bert-large-uncased/phase2/1563"
    epochs="2"
    batch_size="32"
    learning_rate="4.6e-5"
    warmup_proportion="0.2"
    precision="amp"
    num_gpu="8"
    seed="1"
    squad_dir="$BERT_PREP_WORKING_DIR/download/squad/v1.1"
    vocab_file="vocab/bert-large-uncased-vocab.txt"
    CODEDIR=/workspace/bert
    OUT_DIR="$CODEDIR/results"
    mode="train_eval"
    CONFIG_FILE="bert_configs/bert-large-uncased.json"
    max_steps="-1"
    enable_benchmark="false"
    benchmark_steps="100"  # It takes effect only after the enable_benchmark is set to true
    benchmark_warmup_steps="100" # It takes effect only after the enable_benchmark is set to true
    echo $init_checkpoint $epochs $batch_size $learning_rate $warmup_proportion \
     $precision $num_gpu $seed $squad_dir $vocab_file $OUT_DIR $mode $CONFIG_FILE \
     $max_steps $enable_benchmark $benchmark_steps $benchmark_warmup_steps
}

dgxa100-80g_8gpu_tf32 ()
{
    init_checkpoint="results/bert-large-uncased/phase2/1563"
    epochs="2"
    batch_size="32"
    learning_rate="4.6e-5"
    warmup_proportion="0.2"
    precision="amp"
    num_gpu="8"
    seed="1"
    squad_dir="$BERT_PREP_WORKING_DIR/download/squad/v1.1"
    vocab_file="vocab/bert-large-uncased-vocab.txt"
    CODEDIR=/workspace/bert
    OUT_DIR="$CODEDIR/results"
    mode="train_eval"
    CONFIG_FILE="bert_configs/bert-large-uncased.json"
    max_steps="-1"
    enable_benchmark="false"
    benchmark_steps="100"  # It takes effect only after the enable_benchmark is set to true
    benchmark_warmup_steps="100" # It takes effect only after the enable_benchmark is set to true
    echo $init_checkpoint $epochs $batch_size $learning_rate $warmup_proportion \
     $precision $num_gpu $seed $squad_dir $vocab_file $OUT_DIR $mode $CONFIG_FILE \
     $max_steps $enable_benchmark $benchmark_steps $benchmark_warmup_steps
}
