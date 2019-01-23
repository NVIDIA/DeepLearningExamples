# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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


# This script launches ResNet50 inference benchmark in FP16 on 1 GPU with 1,2,4,64,128,192,208 batch size
# Usage ./INFER_BENCHMARK_FP16.sh <additionals flags>

python benchmark.py -n 1 -b 1,2,4,64,128,192,208 --only-inference -e 3 -w 1 -i 100 -o report.json $@
