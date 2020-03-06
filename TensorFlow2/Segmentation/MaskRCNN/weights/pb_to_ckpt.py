#! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import argparse
import logging

import tensorflow as tf

# Pass the filename as an argument
parser = argparse.ArgumentParser()

parser.add_argument(
    "--frozen_model_filename", default="/path-to-pb-file/Binary_Protobuf.pb", type=str, help="Pb model file to import"
)

parser.add_argument(
    "--output_filename", default="/path-to-ckpt-file/model.ckpt", type=str, help="Pb model file to import"
)

args = parser.parse_args()

if __name__ == "__main__":

    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tf.compat.v1.saved_model.loader.load(sess, [tf.saved_model.SERVING], args.frozen_model_filename)

        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, args.output_filename)
        print("Model saved to ckpt format")
