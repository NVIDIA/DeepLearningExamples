#!/bin/bash

docker build --rm -t nvcr.io/nvidia/effdet:21.06-py3-stage . -f Dockerfile
