#!/bin/bash

PATH_TO_COCO=$1
MOUNT_LOCATION='/datasets/data'
NAME='detectron2_interactive'


docker run --runtime=nvidia --cap-add=SYS_PTRACE --cap-add SYS_ADMIN --cap-add DAC_READ_SEARCH --security-opt seccomp=unconfined -v /efficientdet-pytorch:/workspace/object_detection -v /effdet/backbone_checkpoints:/backbone_checkpoints -v /effdet/checkpoints:/checkpoints -v /coco2017/:/workspace/object_detection/datasets/coco -v /waymo_2D_object_detection/raw/:/workspace/object_detection/datasets/waymo --rm --name=$NAME --shm-size=30g --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -t -i nvcr.io/nvidia/effdet:21.06-py3-stage bash
