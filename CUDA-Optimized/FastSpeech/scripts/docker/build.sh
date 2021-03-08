#!/bin/bash

docker build . --rm -t fastspeech
# docker build . --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg UNAME=$(id -un) --rm -t fastspeech