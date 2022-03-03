#!/bin/bash
URL=${1:-"bert"}
PUSH=${2:-"none"}  # 'push' or 'none'

set -e

docker build \
  --network=host \
  --rm \
  --pull \
  --no-cache \
  -t ${URL} \
  .

if [ "${PUSH}" == "push" ]; then
  docker push ${URL}
elif [ "${PUSH}" == "none" ]; then
  echo "Keep the built image locally."
else
  echo "Invalid \${PUSH} option: ${PUSH} !"
  exit 1
fi
