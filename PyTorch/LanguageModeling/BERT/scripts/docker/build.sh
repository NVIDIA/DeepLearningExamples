#!/bin/bash

# Check running from repository root
if [ ! -d .git ]; then
  echo "Not running from repository root! Exiting."
  exit 1
fi

docker build . --rm -t bert
