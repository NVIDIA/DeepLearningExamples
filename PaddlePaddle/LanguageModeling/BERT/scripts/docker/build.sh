#!/bin/bash

# Copyright (c) 2023 NVIDIA Corporation.  All rights reserved.
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

URL=${1:-"bert"}
PUSH=${2:-"none"}  # 'push' or 'none'

set -e

docker build \
  --network=host \
  --rm \
  --pull \
  --no-cache \
  -t ${URL} \
  .

if [ "${PUSH}" == "push" ]; then
  docker push ${URL}
elif [ "${PUSH}" == "none" ]; then
  echo "Keep the built image locally."
else
  echo "Invalid \${PUSH} option: ${PUSH} !"
  exit 1
fi
