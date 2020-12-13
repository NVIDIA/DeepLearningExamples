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

# apt-get update
# apt-get install bc
pip install transformers==2.5.1

logdir="bert-base-log-int8-eff"
mkdir ${logdir}
all_log="${logdir}/all-log.log"
echo -e "| <batch_size, seq_len> | PyTorch (ms) | TorchScript (ms) | FT-EFF-int8v1-OP (ms) | FT-EFF-int8v2-OP (ms) | " > $all_log
echo -e "|:---------------------:|:------:|:------:|:------:|:--------:| " >> $all_log

for batch_size in 1 8 32 64 128 ;
do
for seq_len in 32 64 128 ;
do

    s_t=5
    if [ ${batch_size} -ge 32 ]; then
            s_t=$(echo "scale=0; $s_t * $batch_size * $seq_len / 1024" | bc)
    fi

    ./bin/encoder_gemm ${batch_size} ${seq_len} 12 64 1 0
    sleep ${s_t}

    ./bin/encoder_gemm ${batch_size} ${seq_len} 12 64 1 1
    sleep ${s_t}

    tmp_log_pt=${logdir}/batchsize-${batch_size}-seq-${seq_len}-fp16-log.log
    tmp_log_pt_int8v1_eff=${logdir}/batchsize-${batch_size}-seq-${seq_len}-int8v1-eff-log.log
    tmp_log_pt_int8v2_eff=${logdir}/batchsize-${batch_size}-seq-${seq_len}-int8v2-eff-log.log

    python pytorch/encoder_sample.py ${batch_size} 12 ${seq_len} 12 64 --fp16 --time 2>&1 | tee $tmp_log_pt
    sleep ${s_t}
    
    python pytorch/encoder_sample.py ${batch_size} 12 ${seq_len} 12 64 --fp16 --remove_padding --ths --time --int8_mode 1 2>&1 | tee $tmp_log_pt_int8v1_eff
    sleep ${s_t}
    
    python pytorch/encoder_sample.py ${batch_size} 12 ${seq_len} 12 64 --fp16 --remove_padding --ths --time --int8_mode 2 2>&1 | tee $tmp_log_pt_int8v2_eff
    sleep ${s_t}

    pt_time=`tail -n 2 ${tmp_log_pt} | head -n 1 | awk '{print $5}'`
    ths_time=`tail -n 2 ${tmp_log_pt_int8v1_eff} | head -n 1 | awk '{print $5}'`
    int8v1_eff_o_time=`tail -n 1 ${tmp_log_pt_int8v1_eff} | awk '{print $5}'`
    int8v2_eff_o_time=`tail -n 1 ${tmp_log_pt_int8v2_eff} | awk '{print $5}'`

    echo ' ' | awk -v batch_size=$batch_size -v seq_len=$seq_len -v pt_time=$pt_time -v ths_time=$ths_time \
                        -v int8v1_eff_o_time=$int8v1_eff_o_time -v int8v2_eff_o_time=$int8v2_eff_o_time \
                        '{print "| <" batch_size ", " seq_len "> | " pt_time " | " \
                        ths_time " | " int8v1_eff_o_time " | " int8v2_eff_o_time " | "}' >> $all_log

done
done

