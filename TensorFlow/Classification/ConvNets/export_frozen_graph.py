#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import tensorflow as tf

import horovod.tensorflow as hvd
from model import resnet

tf.app.flags.DEFINE_string(
    'model_name', 'resnet50', 'The name of the architecture to save. The default name was being ' 
     'used to train the model')

tf.app.flags.DEFINE_integer(
    'image_size', 224,
    'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_integer(
    'num_classes', 1001,
    'The number of classes to predict.')

tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')


tf.app.flags.DEFINE_string('input_format', 'NCHW',
                           'The dataformat used by the layers in the model')

tf.app.flags.DEFINE_string('compute_format', 'NCHW',
                           'The dataformat used by the layers in the model')

tf.app.flags.DEFINE_string('checkpoint', '',
                           'The trained model checkpoint.')

tf.app.flags.DEFINE_string(
    'output_file', '', 'Where to save the resulting file to.')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

tf.app.flags.DEFINE_bool(
    'symmetric', False, 'Using symmetric quantization or not.')


tf.app.flags.DEFINE_bool(
    'use_qdq', False, 'Use quantize and dequantize op instead of fake quant op')

tf.app.flags.DEFINE_bool(
    'use_final_conv', False, 'whether to use quantized graph or not.')

tf.app.flags.DEFINE_bool('write_text_graphdef', False,
                         'Whether to write a text version of graphdef.')

FLAGS = tf.app.flags.FLAGS


def main(_):
  
  # Initialize Horovod (TODO: Remove dependency of horovod for freezing graphs)
  hvd.init()

  if not FLAGS.output_file:
    raise ValueError('You must supply the path to save to with --output_file')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default() as graph:
    if FLAGS.input_format=='NCHW':
        input_shape = [FLAGS.batch_size, 3, FLAGS.image_size, FLAGS.image_size]
    else:
        input_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]
    input_images = tf.placeholder(name='input', dtype=tf.float32, shape=input_shape)
    
    resnet50_config = resnet.model_architectures[FLAGS.model_name]
    network = resnet.ResnetModel(FLAGS.model_name, 
                                 FLAGS.num_classes,
                                 resnet50_config['layers'],
                                 resnet50_config['widths'],
                                 resnet50_config['expansions'],
                                 FLAGS.compute_format,
                                 FLAGS.input_format)
    probs, logits = network.build_model(
                input_images,
                training=False,
                reuse=False,
                use_final_conv=FLAGS.use_final_conv)
    
    if FLAGS.quantize:
        tf.contrib.quantize.experimental_create_eval_graph(symmetric=FLAGS.symmetric, 
                                                           use_qdq=FLAGS.use_qdq)

    # Define the saver and restore the checkpoint
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if FLAGS.checkpoint:
            saver.restore(sess, FLAGS.checkpoint)
        else:
            sess.run(tf.global_variables_initializer())
        graph_def = graph.as_graph_def()
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, [probs.op.name])

    # Write out the frozen graph
    tf.io.write_graph(
        frozen_graph_def,
        os.path.dirname(FLAGS.output_file),
        os.path.basename(FLAGS.output_file),
        as_text=FLAGS.write_text_graphdef)


if __name__ == '__main__':
  tf.app.run()
