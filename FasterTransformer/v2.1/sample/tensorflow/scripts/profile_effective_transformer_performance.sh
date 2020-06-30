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

logdir="effective-transformer-log-${precision}"
mkdir ${logdir}
all_log="${logdir}/all-log.log"
echo -e "| <Batch_size, Max_seq_len, Avg_seq_len> | TF (ms) | FT-OP (ms) | Effective FT (ms) | TF Speedup (ms) | FT-OP Speedup | " > $all_log
echo -e "|:---------------------:|:-------:|:----------:|:-----------------:|:---------------:|:-------------:| " >> $all_log

for batch_size in 1 8 32 64 128 ;
do
for seq_len in 32 64 128 ;
do
    ./bin/encoder_gemm ${batch_size} ${seq_len} 12 64 ${precision_num}

    tmp_log_tf=${logdir}/batchsize-${batch_size}-seq-${seq_len}-${precision}-tf-log.log
    tmp_log_tf_2=${logdir}/batchsize-${batch_size}-seq-${seq_len}-${precision}-eff-log.log
    python tensorflow/encoder_sample.py -batch ${batch_size} -s ${seq_len} -time 1 -d ${precision} 2>&1 | tee $tmp_log_tf
    avg_seq_len=$(echo "scale=0; $seq_len / 2" | bc)
    python tensorflow/encoder_sample.py -batch ${batch_size} -s ${seq_len} --avg_seq_len ${avg_seq_len} -remove_padding True -time 1 -d ${precision} 2>&1 | tee $tmp_log_tf_2
    
    ft_o_time=`tail -n 1 ${tmp_log_tf} | awk '{print $9}'`
    tf_time=`tail -n 2 ${tmp_log_tf} | head -n 1 | awk '{print $9}'`
    eff_time=`tail -n 1 ${tmp_log_tf_2} | awk '{print $9}'`
    eff_tf_speedup=$(echo "scale=2; $tf_time / $eff_time" | bc)
    eff_ft_speedup=$(echo "scale=2; $ft_o_time / $eff_time" | bc)

    tail -n 1 ${tmp_log_tf_2} | awk -v batch_size=${batch_size} -v seq_len=${seq_len} -v avg_seq_len=${avg_seq_len} \
                        -v tf_time=$tf_time -v ft_o_time=$ft_o_time -v eff_time=$eff_time \
                        -v eff_tf_speedup=$eff_tf_speedup -v eff_ft_speedup=$eff_ft_speedup \
                        '{print "| <" batch_size ", " seq_len ", " avg_seq_len "> | " tf_time " | " \
                        ft_o_time " | " eff_time " | " eff_tf_speedup " | " eff_ft_speedup " | "  }' >> $all_log
done
done 
done