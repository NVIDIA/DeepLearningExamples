#!/bin/bash -e

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


mTotal=$(cat /proc/meminfo | grep "MemTotal:" | tr -s ' ' | cut -d' ' -f2)
mFOffset=$(cat /proc/meminfo | grep "MemAvailable:" | tr -s ' ' | cut -d' ' -f2)
minFreeMem=$mFOffset
while true; do 
	mF=$(cat /proc/meminfo | grep "MemAvailable:" | tr -s ' ' | cut -d' ' -f2)	
	if [ $minFreeMem -gt $mF ]
	then
		minFreeMem=$mF
		memConsumed=$((mFOffset - mF))
		echo $memConsumed > mem_consumption.txt
	fi
       	sleep 1
done
