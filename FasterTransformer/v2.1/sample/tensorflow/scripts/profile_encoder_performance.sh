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

for precision in fp32 fp16;
do

if [ "$precision" = "fp16" ]; then
    echo "Using fp16."
    precision_num=1
else
    echo "Using fp32"
    precision_num=0
fi

logdir="bert-base-log-${precision}"
mkdir ${logdir}
all_log="${logdir}/all-log.log"
echo -e "| <Batch_size, Seq_len> | TF (ms) | FT-OP (ms) | FT-OP Speedup | FT-CPP (ms) | FT-CPP Speedup | " > $all_log
echo -e "|:---------------------:|:-------:|:----------:|:-------------:|:-----------:|:--------------:| " >> $all_log

for batch_size in 1 8 32 64 128 ;
do
for seq_len in 32 64 128 ;
do
    tmp_log_cpp=${logdir}/batchsize-${batch_size}-seq-${seq_len}-${precision}-cpp-log.log
    ./bin/encoder_gemm ${batch_size} ${seq_len} 12 64 ${precision_num}
    ./bin/encoder_sample ${batch_size} 12 ${seq_len} 12 64 ${precision_num} 0 2>&1 | tee $tmp_log_cpp

    tmp_log_tf=${logdir}/batchsize-${batch_size}-seq-${seq_len}-${precision}-tf-log.log
    python tensorflow/encoder_sample.py -batch ${batch_size} -s ${seq_len} -time 1 -d ${precision} 2>&1 | tee $tmp_log_tf
    
    ft_c_time=`tail -n 1 ${tmp_log_cpp} | awk '{print $9}'`
    ft_o_time=`tail -n 1 ${tmp_log_tf} | awk '{print $9}'`
    tf_time=`tail -n 2 ${tmp_log_tf} | head -n 1 | awk '{print $9}'`
    ft_o_speedup=$(echo "scale=2; $tf_time / $ft_o_time" | bc)
    ft_c_speedup=$(echo "scale=2; $tf_time / $ft_c_time" | bc)
    tail -n 1 ${tmp_log_cpp} | awk -v tf_time=$tf_time -v ft_o_time=$ft_o_time \
                        -v ft_c_time=$ft_c_time -v ft_o_speedup=$ft_o_speedup -v ft_c_speedup=$ft_c_speedup \
                        '{print "| <" $3 ", " $5 "> | " tf_time " | " \
                        ft_o_time " | " ft_o_speedup " | " ft_c_time " | " ft_c_speedup " | "  }' >> $all_log
done
done 
done