#!/bin/bash

BASE_IMAGE=${1:-"ngc_pyt"}
TAG=${2:-"21.11-py3"}
URL=${3:-"lddl:latest"}
PUSH=${4:-"none"}  # 'push' or 'none'

set -e

docker build \
  -f docker/${BASE_IMAGE}.Dockerfile \
  --network=host \
  --rm \
  -t ${URL} \
  --build-arg TAG=${TAG} \
  .

if [ "${PUSH}" == "push" ]; then
  docker push ${URL}
elif [ "${PUSH}" == "none" ]; then
  echo "Keep the built image locally."
else
  echo "Invalid \${PUSH} option: ${PUSH} !"
  exit 1
fi
