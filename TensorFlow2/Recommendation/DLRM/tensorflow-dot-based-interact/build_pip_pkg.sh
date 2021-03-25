#!/usr/bin/env bash

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

set -e

DEST=$(readlink -f "artifacts")
mkdir -p "${DEST}"
TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

cp setup.py "${TMPDIR}"
cp MANIFEST.in "${TMPDIR}"
cp LICENSE "${TMPDIR}"
rsync -avm -L --exclude='*_test.py' --exclude='*/cc/*' --exclude='*/__pycache__/*' ${PIP_FILE_PREFIX}tensorflow_dot_based_interact "${TMPDIR}"
pushd ${TMPDIR}
python3 setup.py bdist_wheel > /dev/null
cp dist/*.whl "${DEST}"
popd
rm -rf ${TMPDIR}
