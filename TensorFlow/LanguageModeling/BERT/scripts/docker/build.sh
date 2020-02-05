#!/bin/bash

docker pull nvcr.io/nvidia/tensorrtserver:19.08-py3

docker build . --rm -t bert
