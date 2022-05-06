#!/bin/bash

# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


AMAZON_BOOKS_2014_DESTINATION=/data/amazon_books_2014

mkdir -p $AMAZON_BOOKS_2014_DESTINATION

if [ ! -f $AMAZON_BOOKS_2014_DESTINATION/meta_Books.json ]; then
    echo "Amazon Books 2014 metadata is not found. Proceeds to download it."
    wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz -P $AMAZON_BOOKS_2014_DESTINATION
    gunzip $AMAZON_BOOKS_2014_DESTINATION/meta_Books.json.gz
else
    echo "Amazon Books 2014 metadata is already downloaded."
fi

if [ ! -f $AMAZON_BOOKS_2014_DESTINATION/reviews_Books.json ]; then
    echo "Amazon Books 2014 reviews are not found. Proceeds to download it."
    wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz -P $AMAZON_BOOKS_2014_DESTINATION
    gunzip $AMAZON_BOOKS_2014_DESTINATION/reviews_Books.json.gz
else 
    echo "Amazon Books 2014 reviews are already downloaded."
fi
