docker run -it --rm \
  --shm-size=1g --ulimit memlock=-1 \
  --ulimit stack=67108864 --net=host \
  -v /Users/gingfungyeung/Dev/DeepLearningExamples/PyTorch/Classification/ConvNets:/workspace/rn50 \
  nv-conv-pytorch-exp bash
