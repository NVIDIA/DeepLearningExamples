  GNU nano 6.2                                  test_download.sh                                            
#!/bin/bash
#
# script to fully prepare ImageNet dataset

## 1. Download the data
IMAGENET_DIR=${1:-"/imagenet"}
dir=$(pwd)
mkdir $IMAGENET_DIR; cd $IMAGENET_DIR
# get ILSVRC2012_img_val.tar (about 6.3 GB). MD5: 29b22e2961454d5413ddabcf34fc5622
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
# get ILSVRC2012_img_train.tar (about 138 GB). MD5: 1d675b47d978889d74fa0da5fadfb00e
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

## 2. Extract the training data:
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; >
cd ..

## 3. Extract the validation data and move images to subfolders:
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

cd ..
rm *.tar
cd $dir

