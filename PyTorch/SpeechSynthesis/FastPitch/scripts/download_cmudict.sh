#!/usr/bin/env bash

set -e

: ${CMUDICT_DIR:="cmudict"}

if [ ! -f $CMUDICT_DIR/cmudict-0.7b ]; then
  echo "Downloading cmudict-0.7b ..."
  wget https://github.com/Alexir/CMUdict/raw/master/cmudict-0.7b -qO $CMUDICT_DIR/cmudict-0.7b
fi
