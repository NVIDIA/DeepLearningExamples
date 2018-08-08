#!/bin/bash

nvidia-docker run -it --rm --ipc=host -v $PWD:/workspace/gnmt/ gnmt bash
