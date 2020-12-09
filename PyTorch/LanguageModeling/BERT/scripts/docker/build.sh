#!/bin/bash
docker build --network=host . --rm --pull --no-cache -t torch_bert_20.06-py3
