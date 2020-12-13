# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

logdir="bert-base-log-int8"
mkdir ${logdir}
all_log="${logdir}/all-log.log"
echo -e "| <Batch_size, Seq_len> | TF (ms) | FT-INT8v1-OP (ms) | FT-INT8v1-OP Speedup | FT-INT8v1-CPP (ms) | FT-INT8v1-CPP Speedup | FT-INT8v2-OP (ms) | FT-INT8v2-OP Speedup | FT-INT8v2-CPP (ms) | FT-INT8v2-CPP Speedup |" > $all_log
echo -e "|:---------------------:|:-------:|:---------------:|:------------------:|:----------------:|:-------------------:|:-----------------:|:--------------------:|:------------------:|:---------------------:| " >> $all_log

for batch_size in 1 8 32 64 128 ;
do
for seq_len in 32 64 128 ;
do

    s_t=5
    if [ ${batch_size} -ge 32 ]; then
      s_t=$(echo "scale=0; $s_t * $batch_size * $seq_len / 1024" | bc)
    fi

    tmp_int8v1_log_cpp=${logdir}/batchsize-${batch_size}-seq-${seq_len}-int8v1-cpp-log.log
    tmp_int8v2_log_cpp=${logdir}/batchsize-${batch_size}-seq-${seq_len}-int8v2-cpp-log.log
    ./bin/encoder_gemm ${batch_size} ${seq_len} 12 64 1 0
    sleep ${s_t}
    ./bin/encoder_gemm ${batch_size} ${seq_len} 12 64 1 1
    sleep ${s_t}
    ./bin/encoder_sample ${batch_size} 12 ${seq_len} 12 64 1 0 1 2>&1 | tee $tmp_int8v1_log_cpp
    sleep ${s_t}
    ./bin/encoder_sample ${batch_size} 12 ${seq_len} 12 64 1 0 2 2>&1 | tee $tmp_int8v2_log_cpp
    sleep ${s_t}

    tmp_log_tf_int8v1=${logdir}/batchsize-${batch_size}-seq-${seq_len}-all-tf-int8v1-log.log
    python tensorflow/encoder_sample_int8.py -batch ${batch_size} -s ${seq_len} -time 1 -d fp16 -int8_mode 1 2>&1 | tee $tmp_log_tf_int8v1
    sleep ${s_t}
    
    tmp_log_tf_int8v2=${logdir}/batchsize-${batch_size}-seq-${seq_len}-all-tf-int8v2-log.log
    python tensorflow/encoder_sample_int8.py -batch ${batch_size} -s ${seq_len} -time 1 -d fp16 -int8_mode 2 2>&1 | tee $tmp_log_tf_int8v2
    sleep ${s_t}
    
    ft_int8v1_c_time=`tail -n 1 ${tmp_int8v1_log_cpp} | awk '{print $9}'`
    ft_int8v2_c_time=`tail -n 1 ${tmp_int8v2_log_cpp} | awk '{print $9}'`
    tf_time=`tail -n 2 ${tmp_log_tf_int8v1} | head -n 1 | awk '{print $9}'`
    ft_int8v1_o_time=`tail -n 1 ${tmp_log_tf_int8v1} | head -n 1 | awk '{print $9}'`
    ft_int8v2_o_time=`tail -n 1 ${tmp_log_tf_int8v2} | head -n 1 | awk '{print $9}'`
    
    ft_int8v1_o_speedup=$(echo "scale=2; $tf_time / $ft_int8v1_o_time" | bc)
    ft_int8v2_o_speedup=$(echo "scale=2; $tf_time / $ft_int8v2_o_time" | bc)
    ft_int8v1_c_speedup=$(echo "scale=2; $tf_time / $ft_int8v1_c_time" | bc)
    ft_int8v2_c_speedup=$(echo "scale=2; $tf_time / $ft_int8v2_c_time" | bc)
    tail -n 1 ${tmp_int8v1_log_cpp} | awk -v tf_time=$tf_time \
                        -v ft_int8v1_o_time=$ft_int8v1_o_time -v ft_int8v1_o_speedup=$ft_int8v1_o_speedup \
                        -v ft_int8v1_c_time=$ft_int8v1_c_time -v ft_int8v1_c_speedup=$ft_int8v1_c_speedup \
                        -v ft_int8v2_o_time=$ft_int8v2_o_time -v ft_int8v2_o_speedup=$ft_int8v2_o_speedup \
                        -v ft_int8v2_c_time=$ft_int8v2_c_time -v ft_int8v2_c_speedup=$ft_int8v2_c_speedup \
                        '{print "| <" $3 ", " $5 "> | " tf_time " | " \
                         ft_int8v1_o_time " | " ft_int8v1_o_speedup " | " ft_int8v1_c_time " | " ft_int8v1_c_speedup " | " \
                         ft_int8v2_o_time " | " ft_int8v2_o_speedup " | " ft_int8v2_c_time " | " ft_int8v2_c_speedup " | " }' >> $all_log
done
done 
