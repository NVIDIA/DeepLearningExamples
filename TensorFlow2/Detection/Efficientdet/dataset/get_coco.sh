#!/bin/bash

# Download and extract coco 2017
mkdir -p /workspace/coco
cd /workspace/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# Convert to tfrecord format
cd /workspace/effdet-tf2
python dataset/create_coco_tfrecord.py --image_dir=/workspace/coco/train2017 \
      --caption_annotations_file=/workspace/coco/annotations/captions_train2017.json \
      --output_file_prefix=/workspace/coco/train --num_shards=256
python dataset/create_coco_tfrecord.py --image_dir=/workspace/coco/val2017 \
      --caption_annotations_file=/workspace/coco/annotations/captions_val2017.json \
      --output_file_prefix=/workspace/coco/val --num_shards=32

# Clean up
rm /workspace/coco/*.zip
rm -rf /workspace/coco/train2017
rm -rf /workspace/coco/val2017


