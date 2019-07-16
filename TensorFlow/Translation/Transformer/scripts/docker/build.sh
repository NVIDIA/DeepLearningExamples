#!/bin/bash

docker build .  -t nvcr.io/nvidian/swdl/jbaczek:transformer_tf
if [[ $1 = push ]]
then
docker push nvcr.io/nvidian/swdl/jbaczek:transformer_tf
fi
