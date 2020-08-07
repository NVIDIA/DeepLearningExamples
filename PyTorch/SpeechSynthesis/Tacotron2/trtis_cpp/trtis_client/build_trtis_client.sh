#!/bin/bash

IMAGE_NAME="trt-tacotron2-waveglow:trtis_client"

if [[ $# != 0 ]]; then
  echo "Unexpected number of arguments: $#"
  echo "USAGE:"
  echo "\t${0}"
  exit 1
fi

FULLPATH="$(dirname $(realpath $0))"

docker build -f Dockerfile.trtis_client . -t "${IMAGE_NAME}"
