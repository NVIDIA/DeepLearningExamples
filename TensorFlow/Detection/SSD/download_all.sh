#!/usr/bin/env bash

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

# Get COCO 2017 data sets
if [ -z $1 ]; then echo "Docker container name is missing" && exit 1; fi
CONTAINER=$1
COCO_DIR=${2:-"/data"}
CHECKPOINT_DIR=${3:-"/checkpoints"}
mkdir $COCO_DIR 2> /dev/null;
chmod 777 $COCO_DIR
cd $COCO_DIR
curl -O http://images.cocodataset.org/zips/train2017.zip; unzip train2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip; unzip val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip annotations_trainval2017.zip
# Download backbone checkpoint
mkdir $CHECKPOINT_DIR 2> /dev/null;
chmod 777 $CHECKPOINT_DIR
cd $CHECKPOINT_DIR
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
tar -xzf resnet_v1_50_2016_08_28.tar.gz
mkdir resnet_v1_50
mv resnet_v1_50.ckpt resnet_v1_50/model.ckpt
nvidia-docker run --rm -it -u 123 -v $COCO_DIR:/data $CONTAINER bash -c '
cd /data
# Create TFRecords
python /workdir/models/research/object_detection/dataset_tools/create_coco_tf_record.py \
    --train_image_dir=`pwd`"/train2017" \
    --val_image_dir=`pwd`"/val2017" \
    --val_annotations_file=`pwd`"/annotations/instances_val2017.json" \
    --train_annotations_file=`pwd`"/annotations/instances_train2017.json" \
    --testdev_annotations_file=`pwd`"/annotations/instances_val2017.json" \
    --test_image_dir=`pwd`"/val2017" \
    --output_dir=`pwd`'
