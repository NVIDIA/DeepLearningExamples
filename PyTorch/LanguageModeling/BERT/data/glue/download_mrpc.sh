#!/usr/bin/env bash

echo "Downloading MRPC data"

wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py

python download_glue_data.py --data_dir . --tasks MRPC
