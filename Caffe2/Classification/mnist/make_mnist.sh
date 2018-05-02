#!/bin/bash

cd "$( cd "$(dirname "$0")" ; pwd -P )"

# Create MNIST databases from previously downloaded data
make_mnist_db --db lmdb --image_file train-images-idx3-ubyte --label_file train-labels-idx1-ubyte --output_file mnist_train_lmdb
make_mnist_db --db lmdb --image_file t10k-images-idx3-ubyte --label_file t10k-labels-idx1-ubyte --output_file mnist_test_lmdb
