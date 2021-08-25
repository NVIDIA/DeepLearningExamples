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

DATA_DIR=$1

wget -c https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

if md5sum -c hashes.md5
then
	echo "DOWNLOAD PASSED"
	# mkdir $DATA_DIR
	mv coco_annotations_minival.tgz $DATA_DIR
	mv train2017.zip $DATA_DIR
	mv val2017.zip $DATA_DIR
	mv annotations_trainval2017.zip $DATA_DIR

	cd $DATA_DIR
	dtrx --one=here coco_annotations_minival.tgz
	dtrx --one=here annotations_trainval2017.zip
	mv annotations.1/* annotations/

	dtrx train2017.zip
	dtrx val2017.zip

	echo "EXTRACTION COMPLETE"
else
	echo "DOWNLOAD FAILED HASHCHECK"
fi