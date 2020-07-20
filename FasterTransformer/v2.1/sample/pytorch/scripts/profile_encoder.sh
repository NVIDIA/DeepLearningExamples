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
echo -e "| <batch_size, seq_len> | PyTorch (ms) | TorchScript (ms) | CustomExt (ms) | Speedup (w/ PyTorch) | Speedup (w/ TorchScript) | " > $all_log
echo -e "|:---------------------:|:------:|:------:|:------:|:--------:|:--------:| " >> $all_log

for batch_size in 1 8 32 64 128 ;
do
for seq_len in 32 64 128 ;
do
    ./bin/encoder_gemm ${batch_size} ${seq_len} 12 64 ${precision_num}

    tmp_log_pt=${logdir}/batchsize-${batch_size}-seq-${seq_len}-${precision}-pt-log.log
    if [ "$precision" = "fp16" ]; then
        python pytorch/encoder_sample.py ${batch_size} 12 ${seq_len} 12 64 --fp16 --time 2>&1 | tee $tmp_log_pt
    else
        python pytorch/encoder_sample.py ${batch_size} 12 ${seq_len} 12 64 --time 2>&1 | tee $tmp_log_pt
    fi
    pt_time=`tail -n 2 ${tmp_log_pt} | head -n 1 | awk '{print $5}'`
    ft_o_time=`tail -n 1 ${tmp_log_pt} | awk '{print $5}'`

    tmp_log_ths=${logdir}/batchsize-${batch_size}-seq-${seq_len}-${precision}-ths-log.log
    if [ "$precision" = "fp16" ]; then
        python pytorch/encoder_sample.py ${batch_size} 12 ${seq_len} 12 64 --fp16 --ths --time 2>&1 | tee $tmp_log_ths
    else
        python pytorch/encoder_sample.py ${batch_size} 12 ${seq_len} 12 64 --ths --time 2>&1 | tee $tmp_log_ths
    fi
    ths_time=`tail -n 2 ${tmp_log_ths} | head -n 1 | awk '{print $5}'`

    speedup_pt=$(echo "scale=2; $pt_time / $ft_o_time" | bc)
    speedup_ths=$(echo "scale=2; $ths_time / $ft_o_time" | bc)
    echo ' ' | awk -v batch_size=$batch_size -v seq_len=$seq_len -v pt_time=$pt_time -v ths_time=$ths_time \
                        -v ft_o_time=$ft_o_time -v speedup_pt=$speedup_pt -v speedup_ths=$speedup_ths \
                        '{print "| <" batch_size ", " seq_len "> | " pt_time " | " \
                        ths_time " | " ft_o_time " | " speedup_pt " | " speedup_ths " | "  }' >> $all_log
done
done
done
