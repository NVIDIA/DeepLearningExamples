#!/bin/bash

USEFP16=false
INSTANCE=2
WORKSPACE=/workspace/rn50/resnet50v1.5

if $USEFP16; then
  TRTUSEFP='--trt-fp16'
  MODELUSEFP='--fp16'
else
  TRTUSEFP=''
  MODELUSEFP=''
fi

#echo $TRTUSEFP $MODELUSEFP

python -m triton.deployer --ts-script $TRTUSEFP --triton-model-name res-trt-16 --triton-max-batch-size 64 \
 --save-dir $WORKSPACE/triton_config --triton-no-cuda --triton-engine-count $INSTANCE -- \
 --config resnet50 --checkpoint $WORKSPACE/model_ckpt/nv_resnet50_weight.pth \
 --batch_size 64 $MODELUSEFP