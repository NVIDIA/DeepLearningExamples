#!/usr/bin/env bash

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

mkdir -p weights/
cd weights

# DOWNLOAD CHECKPOINTS

## Mask RCNN
## ====================== Mask RCNN ====================== ##
BASE_URL="https://storage.googleapis.com/cloud-tpu-checkpoints/mask-rcnn/1555659850"
DEST_DIR="mask-rcnn/1555659850"

wget -N ${BASE_URL}/saved_model.pb -P ${DEST_DIR}
wget -N ${BASE_URL}/variables/variables.data-00000-of-00001 -P ${DEST_DIR}/variables
wget -N ${BASE_URL}/variables/variables.index -P ${DEST_DIR}/variables

## ====================== resnet-nhwc-2018-02-07 ====================== ##
BASE_URL="https://storage.googleapis.com/cloud-tpu-artifacts/resnet/resnet-nhwc-2018-02-07"
DEST_DIR="resnet/resnet-nhwc-2018-02-07"

wget -N ${BASE_URL}/checkpoint -P ${DEST_DIR}
wget -N ${BASE_URL}/model.ckpt-112603.data-00000-of-00001 -P ${DEST_DIR}
wget -N ${BASE_URL}/model.ckpt-112603.index  -P ${DEST_DIR}
wget -N ${BASE_URL}/model.ckpt-112603.meta -P ${DEST_DIR}

## ====================== resnet-nhwc-2018-10-14 ====================== ##
BASE_URL="https://storage.googleapis.com/cloud-tpu-artifacts/resnet/resnet-nhwc-2018-10-14"
DEST_DIR="resnet/resnet-nhwc-2018-10-14"

wget -N ${BASE_URL}/model.ckpt-112602.data-00000-of-00001 -P ${DEST_DIR}
wget -N ${BASE_URL}/model.ckpt-112602.index -P ${DEST_DIR}
wget -N ${BASE_URL}/model.ckpt-112602.meta -P ${DEST_DIR}

# VERIFY CHECKPOINTS
echo "Verifying and Processing Checkpoints..."

python pb_to_ckpt.py \
    --frozen_model_filename=mask-rcnn/1555659850/ \
    --output_filename=mask-rcnn/1555659850/ckpt/model.ckpt

python extract_RN50_weights.py \
    --checkpoint_dir=mask-rcnn/1555659850/ckpt/model.ckpt \
    --save_to=resnet/extracted_from_maskrcnn

echo "Generating list of tensors and their shape..."

python inspect_checkpoint.py --file_name=mask-rcnn/1555659850/ckpt/model.ckpt \
    > mask-rcnn/1555659850/tensors_and_shape.txt

python inspect_checkpoint.py --file_name=resnet/resnet-nhwc-2018-02-07/model.ckpt-112603 \
    > resnet/resnet-nhwc-2018-02-07/tensors_and_shape.txt

python inspect_checkpoint.py --file_name=resnet/resnet-nhwc-2018-10-14/model.ckpt-112602 \
    > resnet/resnet-nhwc-2018-10-14/tensors_and_shape.txt

python inspect_checkpoint.py --file_name=resnet/extracted_from_maskrcnn/resnet50.ckpt \
    > resnet/extracted_from_maskrcnn/tensors_and_shape.txt

echo "Script Finished with Success"
