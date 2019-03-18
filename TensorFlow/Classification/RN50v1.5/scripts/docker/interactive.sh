#!/bin/bash

nvidia-docker run -it --rm --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/rn50v15_tf/ rn50v15_tf bash
