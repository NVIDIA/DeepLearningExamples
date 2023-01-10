#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


# Copyright 2020 Chengxi Zang
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


gpus=${1:-1}
prec=${2:-amp}
flags="${@:3}"


if [[ "${gpus}" == "1" ]]; then
    cmd="python"
else
    cmd="torchrun --nproc_per_node=${gpus}"
fi

cmd="${cmd} \
    /workspace/moflow_pyt/moflow/runtime/train.py \
    --cuda_graph \
    ${flags} \
    "

eval_cmd="python \
    /workspace/moflow_pyt/moflow/runtime/evaluate.py \
    --steps 1000 \
    --jit \
    ${flags} \
    "

if [ $prec == "amp" ]; then
    cmd="${cmd} --amp"
    eval_cmd="${eval_cmd} --amp"
fi

if [[ $gpus == 1 ]]; then
    cmd="${cmd} --learning_rate 0.0001"
fi

set -x
bash -c "${cmd} && ${eval_cmd}"
