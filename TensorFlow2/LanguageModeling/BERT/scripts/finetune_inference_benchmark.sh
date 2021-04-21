#!/bin/bash

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# ==============================================================================
LOGFILE="/results/squad_inference_benchmark_bert.log"
tmp_file="/tmp/squad_inference_benchmark.log"
for s in base large; do
  for precision in fp16 fp32; do
    for slen in 128 384; do
      len=""
      if [ "$slen" = 128 ] ; then
        len="_seq128"
      fi
      for i in 1 2 4 8; do
        scripts/run_inference_benchmark$len.sh $s $i $precision true |& tee $tmp_file

        perf=`cat $tmp_file | grep -F 'Throughput Average (sentences/sec) =' | tail -1 | awk -F'= ' '{print $2}'`
        la=`cat $tmp_file | grep -F 'Latency Average (ms)' | awk -F'= ' '{print $2}'`
        l50=`cat $tmp_file | grep -F 'Latency Confidence Level 50 (ms)' | awk -F'= ' '{print $2}'`
        l90=`cat $tmp_file | grep -F 'Latency Confidence Level 90 (ms)' | awk -F'= ' '{print $2}'`
        l95=`cat $tmp_file | grep -F 'Latency Confidence Level 95 (ms)' | awk -F'= ' '{print $2}'`
        l99=`cat $tmp_file | grep -F 'Latency Confidence Level 99 (ms)' | awk -F'= ' '{print $2}'`
        l100=`cat $tmp_file | grep -F 'Latency Confidence Level 100 (ms)' | awk -F'= ' '{print $2}'`

        echo "$s $slen $i $precision $perf $la $l50 $l90 $l95 $l99 $l100" >> $LOGFILE
      done
    done
  done
done
