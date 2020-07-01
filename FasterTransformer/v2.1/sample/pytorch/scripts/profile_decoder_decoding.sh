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
pip install opennmt-py==1.1.1

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
mkdir -p ${logdir}
all_log="${logdir}/all-log.log"
echo -e "| <batch_size, seq_len, beam_size> | PyTorch (ms) | Decoder (ms) | Decoding (ms) | Decoder Speedup | Decoding Speedup | " > $all_log
echo -e "|:-----------------------:|:------:|:------:|:------:|:---------:|:---------:| " >> $all_log

for beam_size in 1 4 ;
do
for batch_size in 1 8 32 64 128 ;
do
for seq_len in 32 64 128 ;
do
    ./bin/decoding_gemm ${batch_size} ${beam_size} 8 64 31538 ${seq_len} 512 ${precision_num}
    tmp_log=${logdir}/beamsize-${beam_size}-batchsize-${batch_size}-seq-${seq_len}-${precision}-log.log
    if [ "$precision" = "fp16" ]; then
        python pytorch/decoding_sample.py ${batch_size} 6 ${seq_len} 8 64 ${beam_size} 31538 --fp16 --time 2>&1 | tee $tmp_log
    else
        python pytorch/decoding_sample.py ${batch_size} 6 ${seq_len} 8 64 ${beam_size} 31538 --time 2>&1 | tee $tmp_log
    fi
    pt_time=`tail -n 3 ${tmp_log} | head -n 1 | awk '{print $5}'`
    decoder_time=`tail -n 2 ${tmp_log} | head -n 1 | awk '{print $7}'`
    decoding_o_time=`tail -n 1 ${tmp_log} | awk '{print $5}'`

    speedup_decoder=$(echo "scale=2; $pt_time / $decoder_time" | bc)
    speedup_decoding=$(echo "scale=2; $pt_time / $decoding_o_time" | bc)
    echo ' ' | awk -v batch_size=$batch_size -v seq_len=$seq_len -v beam_size=$beam_size \
                        -v pt_time=$pt_time -v decoder_time=$decoder_time \
                        -v decoding_o_time=$decoding_o_time -v speedup_decoder=$speedup_decoder -v speedup_decoding=$speedup_decoding \
                        '{print "| <" batch_size ", " seq_len ", " beam_size "> | " pt_time " | " \
                        decoder_time " | " decoding_o_time " | " speedup_decoder " | " speedup_decoding " | "  }' >> $all_log
done
done
done
done
