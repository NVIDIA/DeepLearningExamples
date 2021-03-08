docker run -it --rm \
  --shm-size=1g --ulimit memlock=-1 \
  --ulimit stack=67108864 --net=host \
  -v $(pwd):/workspace/rn50 \
  nv-conv-pytorch-exp bash
