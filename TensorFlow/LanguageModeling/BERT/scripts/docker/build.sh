#!/bin/bash

docker pull nvcr.io/nvidia/tritonserver:20.06-v1-py3

docker build . --rm -t bert
