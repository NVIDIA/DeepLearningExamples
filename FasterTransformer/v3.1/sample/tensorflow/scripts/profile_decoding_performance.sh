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

logdir="decoding-log-${precision}"
mkdir ${logdir}
all_log="${logdir}/all-log.log"
echo -e "| <Batch_size, beam_width, Seq_len> | TF (ms) | FT-OP (ms) | FT-OP Speedup | FT-CPP (ms) | FT-CPP Speedup | " > $all_log
echo -e "|:---------------------------------:|:-------:|:----------:|:-------------:|:-----------:|:--------------:| " >> $all_log

for batch_size in 1 8 32 64 128 ;
do
for beam_width in 1 4 ;
do
for seq_len in 32 64 128 ;
do
    tmp_log_cpp=${logdir}/batchsize-${batch_size}-beamwidth-${beam_width}-seq-${seq_len}-${precision}-cpp-log.log
    tmp_log_tf=${logdir}/batchsize-${batch_size}-beamwidth-${beam_width}-seq-${seq_len}-${precision}-tf-log.log

    ./bin/decoding_gemm ${batch_size} ${beam_width} 8 64 30000 ${seq_len} 512 ${precision_num}
    ./bin/decoding_beamsearch_sample ${batch_size} ${beam_width} 8 64 30000 ${seq_len} 6 512 ${precision_num} 2>&1 | tee ${tmp_log_cpp}
    python tensorflow/decoding_sample.py \
            --batch_size ${batch_size} \
            --beam_width ${beam_width} \
            --max_seq_len ${seq_len} \
            --head_number 8 \
            --size_per_head 64 \
            --memory_hidden_dim 512 \
            --num_layer 6 \
            --data_type ${precision} \
            --test_time 01 2>&1 | tee ${tmp_log_tf}
    ft_c_time=`tail -n 1 ${tmp_log_cpp} | awk '{print $17}'`
    ft_o_time=`tail -n 1 ${tmp_log_tf} | awk '{print $17}'`
    tf_time=`tail -n 2 ${tmp_log_tf} | head -n 1 | awk '{print $17}'`
    ft_o_speedup=$(echo "scale=2; $tf_time / $ft_o_time" | bc)
    ft_c_speedup=$(echo "scale=2; $tf_time / $ft_c_time" | bc)
    tail -n 1 ${tmp_log_cpp} | awk -v tf_time=$tf_time -v ft_o_time=$ft_o_time \
                        -v ft_c_time=$ft_c_time -v ft_o_speedup=$ft_o_speedup -v ft_c_speedup=$ft_c_speedup \
                        '{print "| <" $3 ", " $5 ", " $11 "> | " tf_time " | " \
                        ft_o_time " | " ft_o_speedup " | " ft_c_time " | " ft_c_speedup " | "  }' >> $all_log
done # for seq_len
done # for beam_width
done # for batch_size
done # for precision