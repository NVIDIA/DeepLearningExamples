#!/bin/bash

nvidia-docker run --init -it --rm --ipc=host -v $PWD:/workspace/gnmt/ gnmt bash
