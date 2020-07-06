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

import tensorflow as tf
import numpy as np
import argparse
import os
        
def process_checkpoint(input_ckpt, output_ckpt_path, dense_layer):
    """
    This function loads a RN50 checkpoint with Dense layer as the final layer 
    and transforms the final dense layer into a 1x1 convolution layer. The weights
    of the dense layer are reshaped into weights of 1x1 conv layer.
    Args:
        input_ckpt: Path to the input RN50 ckpt which has dense layer as classification layer.
    Returns:
        None. New checkpoint with 1x1 conv layer as classification layer is generated.
    """
    with tf.Session() as sess:
        # Load all the variables
        all_vars = tf.train.list_variables(input_ckpt)
        # Capture the dense layer weights and reshape them to a 4D tensor which would be 
        # the weights of a 1x1 convolution layer. This code replaces the dense (FC) layer
        # to a 1x1 conv layer. 
        dense_layer_value=0.
        new_var_list=[]
        for var in all_vars:
            curr_var = tf.train.load_variable(input_ckpt, var[0])
            if var[0]==dense_layer:
                dense_layer_value = curr_var
            else:
                new_var_list.append(tf.Variable(curr_var, name=var[0]))
 
        dense_layer_shape = [1, 1, 2048, 1001]
        new_var_value = np.reshape(dense_layer_value, dense_layer_shape)
        new_var = tf.Variable(new_var_value, name=dense_layer)
        new_var_list.append(new_var)
        
        sess.run(tf.global_variables_initializer())
        tf.train.Saver(var_list=new_var_list).save(sess, output_ckpt_path, write_meta_graph=False, write_state=False)
        print ("Rewriting checkpoint completed")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to pretrained RN50 checkpoint with dense layer')
    parser.add_argument('--dense_layer', type=str, default='resnet50/output/dense/kernel')
    parser.add_argument('--output', type=str, default='output_dir', help="Output directory to store new checkpoint")
    args = parser.parse_args()
    
    input_ckpt = args.input
    # Create an output directory
    os.mkdir(args.output)
    
    new_ckpt='new.ckpt'
    new_ckpt_path = os.path.join(args.output, new_ckpt)
    with open(os.path.join(args.output, "checkpoint"), 'w') as file:
        file.write("model_checkpoint_path: "+ "\"" + new_ckpt + "\"")
        
    # Process the input checkpoint, apply transforms and generate a new checkpoint.
    process_checkpoint(input_ckpt, new_ckpt_path, args.dense_layer)