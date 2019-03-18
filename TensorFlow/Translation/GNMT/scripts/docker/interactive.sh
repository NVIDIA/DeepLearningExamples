#!/bin/bash

nvidia-docker run -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/gnmt gnmt_tf bash
