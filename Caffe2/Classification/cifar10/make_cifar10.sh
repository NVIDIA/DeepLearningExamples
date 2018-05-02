#!/bin/bash
set -e

cd "$( cd "$(dirname "$0")" ; pwd -P )"

# Create CIFAR10 train + test databases
make_cifar_db --db lmdb --input_folder "$(pwd)" --output_train_db_name cifar10_train_lmdb --output_test_db_name cifar10_test_lmdb
