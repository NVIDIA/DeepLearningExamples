# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#! /bin/bash


bind_cpu_cores=([0]="48-51,176-179" [1]="60-63,188-191" [2]="16-19,144-147" [3]="28-31,156-159"
                [4]="112-115,240-243" [5]="124-127,252-255" [6]="80-83,208-211" [7]="92-95,220-223")

bind_mem=([0]="3" [1]="3" [2]="1" [3]="1"
          [4]="7" [5]="7" [6]="5" [7]="5")
