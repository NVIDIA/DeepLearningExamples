python -m triton.deployer --trt --triton-model-name sernxt-trt-16 \
  --triton-max-batch-size 64 --save-dir /repository --triton-no-cuda -- \
  --config se-resnext101-32x4d --checkpoint \
  /repository/nvidia_se_weight.pth --batch_size 64