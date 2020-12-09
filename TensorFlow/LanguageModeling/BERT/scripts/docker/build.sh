#!/bin/bash

docker pull nvcr.io/nvidia/tritonserver:20.09-py3

docker build . --rm -t tf_bert_20.06-py3
