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

logdir="bert-base-log-int8"
mkdir ${logdir}
all_log="${logdir}/all-log.log"
echo -e "| <batch_size, seq_len> | PyTorch (ms) | TorchScript (ms) | CustomExt int8v1 (ms) | CustomExt int8v2 (ms) | Speedup int8v1 (w/ PyTorch) | Speedup int8v2 (w/ PyTorch) | Speedup int8v1 (w/ TorchScript) | Speedup int8v2 (w/ TorchScript) |" > $all_log
echo -e "|:---------------------:|:------:|:------:|:------:|:--------:|:--------:|:------:|:------:|:------:| " >> $all_log

for batch_size in 1 8 32 64 128 ;
do
for seq_len in 32 64 128 ;
do
    ./bin/encoder_gemm ${batch_size} ${seq_len} 12 64 1 1
    sleep 60
    ./bin/encoder_gemm ${batch_size} ${seq_len} 12 64 1 0
    sleep 60
    tmp_log_int8v1_pt=${logdir}/batchsize-${batch_size}-seq-${seq_len}-int8v1-pt-log.log
    tmp_log_int8v2_pt=${logdir}/batchsize-${batch_size}-seq-${seq_len}-int8v2-pt-log.log
    
    python pytorch/encoder_sample.py ${batch_size} 12 ${seq_len} 12 64 --fp16 --time --int8_mode 1 2>&1 | tee $tmp_log_int8v1_pt
    sleep 60
    
    python pytorch/encoder_sample.py ${batch_size} 12 ${seq_len} 12 64 --fp16 --time --int8_mode 2 2>&1 | tee $tmp_log_int8v2_pt
    sleep 60
    
    pt_time=`tail -n 2 ${tmp_log_int8v1_pt} | head -n 1 | awk '{print $5}'`
    ft_int8v1_o_time=`tail -n 1 ${tmp_log_int8v1_pt} | awk '{print $5}'`
    ft_int8v2_o_time=`tail -n 1 ${tmp_log_int8v2_pt} | awk '{print $5}'`

    tmp_log_ths=${logdir}/batchsize-${batch_size}-seq-${seq_len}-fp16-ths-log.log
    python pytorch/encoder_sample.py ${batch_size} 12 ${seq_len} 12 64 --fp16 --ths --time 2>&1 | tee $tmp_log_ths
    sleep 60

    ths_time=`tail -n 2 ${tmp_log_ths} | head -n 1 | awk '{print $5}'`

    speedup_int8v1_pt=$(echo "scale=2; $pt_time / $ft_int8v1_o_time" | bc)
    speedup_int8v2_pt=$(echo "scale=2; $pt_time / $ft_int8v2_o_time" | bc)
    speedup_int8v1_ths=$(echo "scale=2; $ths_time / $ft_int8v1_o_time" | bc)
    speedup_int8v2_ths=$(echo "scale=2; $ths_time / $ft_int8v2_o_time" | bc)
    echo ' ' | awk -v batch_size=$batch_size -v seq_len=$seq_len -v pt_time=$pt_time -v ths_time=$ths_time \
                        -v ft_int8v1_o_time=$ft_int8v1_o_time -v ft_int8v2_o_time=$ft_int8v2_o_time -v speedup_int8v1_pt=$speedup_int8v1_pt -v speedup_int8v2_pt=$speedup_int8v2_pt -v speedup_int8v1_ths=$speedup_int8v1_ths -v speedup_int8v2_ths=$speedup_int8v2_ths \
                        '{print "| <" batch_size ", " seq_len "> | " pt_time " | " \
                        ths_time " | " ft_int8v1_o_time " | " ft_int8v2_o_time " | " speedup_int8v1_pt " | " speedup_int8v2_pt " | " speedup_int8v1_ths " | " speedup_int8v2_ths " | "  }' >> $all_log
done
done

