# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import pickle 

if len(sys.argv) != 2:
    print("[ERROR] dump_pruned_model.py needs a ckpt file as input. \n e.g. python dump_pruned_model.py model.ckpt")
    sys.exit(0)

# Get the values of all variables in the checkpoint file, and then save the values of all variables in a pickle file by dict
# The key of the dict is the name of variables
# The value of the dict is the values of variables
# For example, all_variables[0]=<tf.Variable 'transformer/decoder/layer_0/masked_multi_head/LayerNorm/beta:0' shape=(512,) dtype=float32_ref>,
# then the key is 'transformer/decoder/layer_0/masked_multi_head/LayerNorm/beta:0'; the value is sess.run(all_variables[0])

# If you need to dump the model which has same structure but different variable name, you can convert the name of your model into opennmt's name one by one.
# For example, the name of beta variable of first layer normalization in first layer of decoder is 'transformer/decoder/layer_0/masked_multi_head/LayerNorm/beta:0',
# and in your model, you use other name like 'body/decoder/layer_0/self_attention/LayerNorm/beta:0'
# then the key is: 'transformer/decoder/layer_0/masked_multi_head/LayerNorm/beta:0' (the model name of opennmt)
# and the value is sess.run(<tf.Variable 'transformer/decoder/layer_0/masked_multi_head/LayerNorm/beta:0', shape=(512,) dtype=float32_ref>) (your variable value)

ckpt_name = sys.argv[1]
    
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(ckpt_name + ".meta")
    saver.restore(sess, (ckpt_name))
    all_variables = tf.trainable_variables()
    ckpt = {}
    
    all_val = sess.run(all_variables)
    for var, val in zip(all_variables, all_val):
        if var.name.find("Adam") == -1:
            ckpt[var.name] = val
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(ckpt, f, pickle.HIGHEST_PROTOCOL)
