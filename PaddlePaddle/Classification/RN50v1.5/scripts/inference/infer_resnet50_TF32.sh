# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

python inference.py \
    --trt-inference-dir ./inference_tf32 \
    --trt-precision FP32 \
    --dali-num-threads 8 \
    --batch-size 256 \
    --benchmark-steps 1024 \
    --benchmark-warmup-steps 16 \
    --trt-use-synthetic True
