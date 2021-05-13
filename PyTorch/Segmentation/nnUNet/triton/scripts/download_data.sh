#!/usr/bin/env bash
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
if [ -f "${CHECKPOINT_DIR}/nnunet_pyt_ckpt_3d_fold2_amp_21.02.0.zip" ]; then
  echo "Checkpoint already downloaded."
else
  echo "Downloading checkpoint ..."
  wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/nnunet_pyt_ckpt_3d_fold2_amp/versions/21.02.0/zip -O nnunet_pyt_ckpt_3d_fold2_amp_21.02.0.zip || {
    echo "ERROR: Failed to download checkpoint from NGC"
    exit 1
  }
  unzip nnunet_pyt_ckpt_3d_fold2_amp_21.02.0.zip -d ${CHECKPOINT_DIR}
  rm nnunet_pyt_ckpt_3d_fold2_amp_21.02.0.zip
  echo "ok"
fi

