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

logdir="effective-transformer-log-int8"
mkdir ${logdir}
all_log="${logdir}/all-log.log"
echo -e "| <Batch_size, Max_seq_len, Avg_seq_len> | TF (ms) | Effective int8v1 FT op (ms) | Effective int8v2 FT op (ms) | Effective int8v1 FT cpp (ms) | Effective int8v2 FT cpp (ms) |" > $all_log
echo -e "|:---------------------:|:-------:|:-----------------:|:-----------------:| :-----------------:|:-----------------:| " >> $all_log

for batch_size in 1 8 32 64 128 ;
do
for seq_len in 32 64 128 ;
do

    s_t=5
    if [ ${batch_size} -ge 32 ]; then
            s_t=$(echo "scale=0; $s_t * $batch_size * $seq_len / 1024" | bc)
    fi
    echo ${s_t}

    ./bin/encoder_gemm ${batch_size} ${seq_len} 12 64 1 1
    sleep ${s_t}
    
    tmp_log_tf_int8v1=${logdir}/batchsize-${batch_size}-seq-${seq_len}-int8v1-tf-log.log
    tmp_log_tf_int8v2=${logdir}/batchsize-${batch_size}-seq-${seq_len}-int8v2-tf-log.log

    tmp_log_cpp_int8v1=${logdir}/batchsize-${batch_size}-seq-${seq_len}-int8v1-cpp-log.log
    tmp_log_cpp_int8v2=${logdir}/batchsize-${batch_size}-seq-${seq_len}-int8v2-cpp-log.log
    avg_seq_len=$(echo "scale=0; $seq_len / 2" | bc)
    
    python tensorflow/encoder_sample_int8.py -batch ${batch_size} -s ${seq_len} -time 1 -remove_padding True --avg_seq_len ${avg_seq_len} -d fp16 -int8_mode 1 2>&1 | tee $tmp_log_tf_int8v1
    sleep ${s_t}
    
    python tensorflow/encoder_sample_int8.py -batch ${batch_size} -s ${seq_len} -time 1 -remove_padding True --avg_seq_len ${avg_seq_len} -d fp16 -int8_mode 2 2>&1 | tee $tmp_log_tf_int8v2
    sleep ${s_t}

    ./bin/encoder_sample ${batch_size} 12 ${seq_len} 12 64 1 1 1 2>&1 | tee $tmp_log_cpp_int8v1
    sleep ${s_t}

    ./bin/encoder_sample ${batch_size} 12 ${seq_len} 12 64 1 1 2 2>&1 | tee $tmp_log_cpp_int8v2
    sleep ${s_t}

    tf_o_time=`tail -n 2 ${tmp_log_tf_int8v1} | head -n 1 | awk '{print $9}'`
    eff_int8v1_time=`tail -n 1 ${tmp_log_tf_int8v1} | head -n 1 | awk '{print $9}'`
    eff_int8v2_time=`tail -n 1 ${tmp_log_tf_int8v2} | head -n 1 | awk '{print $9}'`
    eff_cpp_int8v1_time=`tail -n 1 ${tmp_log_cpp_int8v1} | head -n 1 | awk '{print $9}'`
    eff_cpp_int8v2_time=`tail -n 1 ${tmp_log_cpp_int8v2} | head -n 1 | awk '{print $9}'`

    tail -n 1 ${tmp_log_tf_int8v1} | awk -v batch_size=${batch_size} -v seq_len=${seq_len} -v avg_seq_len=${avg_seq_len} \
                        -v tf_o_time=$tf_o_time -v eff_int8v1_time=$eff_int8v1_time \
                        -v eff_int8v2_time=$eff_int8v2_time -v eff_cpp_int8v1_time=$eff_cpp_int8v1_time -v eff_cpp_int8v2_time=$eff_cpp_int8v2_time\
                        '{print "| <" batch_size ", " seq_len ", " avg_seq_len "> | " tf_o_time " | " \
                        eff_int8v1_time " | " eff_int8v2_time " | " eff_cpp_int8v1_time " | " eff_cpp_int8v2_time " | "}' >> $all_log
done
done 

