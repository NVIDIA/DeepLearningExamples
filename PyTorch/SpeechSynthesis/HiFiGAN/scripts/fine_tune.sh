#!/usr/bin/env bash

set -a

: ${FINE_TUNE_DIR:="data/mels-fastpitch-ljs22khz"}
: ${FINE_TUNE_LR_FACTOR:=3}
: ${EPOCHS:=10000}  # 6500 + 3500

if [ ! -d "$FINE_TUNE_DIR" ]; then
    echo "Finetuning spectrograms missing at $FINE_TUNE_DIR ."
    echo "Those need to be generated with scripts/extract_fine_tune_mels.sh"
    echo "Consult the README.md for details."
    exit 1
fi

bash scripts/train_lj22khz.sh "$@"
