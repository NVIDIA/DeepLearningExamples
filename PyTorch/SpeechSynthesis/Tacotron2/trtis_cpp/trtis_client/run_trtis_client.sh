#!/bin/bash

IMAGE_NAME="trt-tacotron2-waveglow:trtis_client"

if [[ -z "${1}" ]]; then
  echo "Must supply file containing phrases."
  exit 1
fi
FILENAME="$(realpath ${1})"
INPUT_NAME="$(basename ${FILENAME})"

BATCH_SIZE=1
if [[ ${#} == 2 ]]; then
  BATCH_SIZE="${2}"
fi

if [[ ${#} > 2 ]]; then
  echo "Invalid number of arguments: ${#}"
  echo "Expected: $0 <input filename> <batch size>"
  exit 1
fi

docker run \
       --mount type="bind,source=${FILENAME},target=/mount/input.txt" \
       -v "${PWD}/audio/:/workspace/audio/" \
       --rm \
       --net=host \
       "${IMAGE_NAME}" trtis_client "/mount/input.txt" "${BATCH_SIZE}" || exit 1

