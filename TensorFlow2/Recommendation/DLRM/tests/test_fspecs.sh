#!/bin/bash

NAMES=${1:-'*.yaml'}
COMMON_OPTS="--xla --amp"

bash test_with_opts.sh "${NAMES}" "${COMMON_OPTS}"

#
#        usage:
#        docker build . -t nvidia_dlrm_tf
#        docker run --security-opt seccomp=unconfined --runtime=nvidia -it --rm --ipc=host  -v ${PWD}/data:/data nvidia_dlrm_tf bash
#        cd tests
#        bash test_fspecs.sh