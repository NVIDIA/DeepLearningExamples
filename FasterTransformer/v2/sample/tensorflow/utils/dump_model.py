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

ckpt_name = sys.argv[1]

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(ckpt_name + ".meta")
    saver.restore(sess, (ckpt_name))

    def dumpModel_new():
        all_variables = tf.trainable_variables()
        ckpt = {}
        
        for i, var in enumerate(all_variables):
            print("[INFO] %d/%d" %(i, len(all_variables)), end='\r')
            sys.stdout.flush()
            if var in tf.trainable_variables():
                val = sess.run(var)
                name = None
                if var.name.find("Adam") != -1:
                    continue
                elif var.name.find("encoder") != -1:
                    # transformer/encoder/layer_x/multi_head/conv1d/kernel:0 -> transformer/encoder/layer_x/attention/self/query, key, value/kernel:0
                    # transformer/encoder/layer_x/multi_head/conv1d_1/kernel:0 -> transformer/encoder/layer_x/attention/output/kernel:0
                    # transformer/encoder/layer_x/multi_head/LayerNorm/gamma:0 -> transformer/encoder/layer_x/attention/output/LayerNorm/gamma:0
                    if var.name.find("multi_head/conv1d/") != -1:
                        dim = val.shape[-1] / 3
                        Q, K, V = np.split(val, [dim, dim * 2], axis=-1)
                        ckpt[var.name.replace("multi_head/conv1d/", "attention/self/query/")] = Q
                        ckpt[var.name.replace("multi_head/conv1d/", "attention/self/key/")] = K
                        ckpt[var.name.replace("multi_head/conv1d/", "attention/self/value/")] = V

                    elif var.name.find("multi_head/conv1d_1/") != -1:
                        name = var.name.replace("multi_head/conv1d_1/", "attention/output/")
                        ckpt[name] = val

                    elif var.name.find("multi_head/LayerNorm/") != -1:
                        name = var.name.replace("multi_head/LayerNorm/", "attention/output/LayerNorm/")
                        ckpt[name] = val
                    
                    # transformer/encoder/layer_x/ffn/conv1d/kernel:0 -> transformer/encoder/layer_x/intermediate/dense/kernel:0
                    # transformer/encoder/layer_x/ffn/LayerNorm/beta:0 -> transformer/encoder/layer_x/output/LayerNorm/beta:0
                    # transformer/encoder/layer_x/ffn/conv1d_1/kernel:0 -> transformer/encoder/layer_x/output/dense/kernel:0
                    elif var.name.find("ffn/conv1d/") != -1:
                        name = var.name.replace("ffn/conv1d/", "intermediate/dense/")
                        ckpt[name] = val
                    
                    elif var.name.find("ffn/LayerNorm/") != -1:
                        name = var.name.replace("ffn/LayerNorm/", "output/LayerNorm/")
                        ckpt[name] = val
                    
                    elif var.name.find("ffn/conv1d_1/") != -1:
                        name = var.name.replace("ffn/conv1d_1/", "output/dense/")
                        ckpt[name] = val
                        
                    elif var.name.find("transformer/encoder/w_embs") != -1:
                        name = var.name
                        ckpt[name] = val
                        
                elif var.name.find("decoder") != -1:
                    pre_name = var.name.replace("decoder", "decoding/decoder")
                    
                    # transformer/decoder/layer_x/masked_multi_head/conv1d/kernel:0 -> transformer/decoder/layer_x/masked_multi_head/query, key, value/kernel:0
                    # transformer/decoder/layer_x/masked_multi_head/conv1d_1/kernel:0 -> transformer/decoder/layer_x/masked_multi_head/conv1d/kernel:0
                    # transformer/decoder/layer_x/masked_multi_head/LayerNorm/gamma:0 -> transformer/decoder/layer_x/masked_multi_head/LayerNorm/gamma:0
                    if var.name.find("masked_multi_head/conv1d/") != -1:
                        dim = val.shape[-1] / 3
                        Q, K, V = np.split(val, [dim, dim * 2], axis=-1)
                        ckpt[pre_name.replace("masked_multi_head/conv1d/", "masked_multi_head/query/")] = Q
                        ckpt[pre_name.replace("masked_multi_head/conv1d/", "masked_multi_head/key/")] = K
                        ckpt[pre_name.replace("masked_multi_head/conv1d/", "masked_multi_head/value/")] = V
                    elif var.name.find("masked_multi_head/conv1d_1/") != -1:
                        name = pre_name.replace("masked_multi_head/conv1d_1/", "masked_multi_head/conv1d/")
                        ckpt[name] = val
                    elif var.name.find("masked_multi_head/LayerNorm/") != -1:
                        name = pre_name
                        ckpt[name] = val
                        
                    # transformer/decoder/layer_x/multi_head/conv1d/kernel:0 -> transformer/decoder/layer_x/multi_head/query/kernel:0
                    # transformer/decoder/layer_x/multi_head/conv1d_1/kernel:0 -> transformer/decoder/layer_x/multi_head/key, value/kernel:0
                    # transformer/decoder/layer_x/multi_head/conv1d_2/kernel:0 -> transformer/decoder/layer_x/multi_head/conv1d/kernel
                    # transformer/decoder/layer_x/multi_head/LayerNorm/gamma:0 -> transformer/decoder/layer_x/multi_head/LayerNorm/gamma:0
                    elif var.name.find("multi_head/conv1d/") != -1:
                        name = pre_name.replace("multi_head/conv1d/", "multi_head/query/")
                        ckpt[name] = val
                    elif var.name.find("multi_head/conv1d_1/") != -1:
                        dim = val.shape[-1] / 2
                        K, V = np.split(val, [dim], axis=-1)
                        ckpt[pre_name.replace("multi_head/conv1d_1/", "multi_head/key/")] = K
                        ckpt[pre_name.replace("multi_head/conv1d_1/", "multi_head/value/")] = V
                    elif var.name.find("multi_head/conv1d_2/") != -1:
                        name = pre_name.replace("multi_head/conv1d_2/", "multi_head/conv1d/")
                        ckpt[name] = val
                    elif var.name.find("multi_head/LayerNorm/") != -1:
                        name = pre_name
                        ckpt[name] = val
                    
                    # transformer/decoder/layer_x/ffn/conv1d/kernel:0 -> transformer/decoder/layer_x/intermediate/dense/kernel:0
                    # transformer/decoder/layer_x/ffn/LayerNorm/beta:0 -> transformer/decoder/layer_x/output/LayerNorm/beta:0
                    # transformer/decoder/layer_x/ffn/conv1d_1/kernel:0 -> transformer/decoder/layer_x/output/dense/kernel:0                    
                    elif var.name.find("ffn/conv1d/") != -1:
                        # name = var.name.replace("ffn/conv1d/", "intermediate/dense/")
                        name = pre_name
                        ckpt[name] = val
                    elif var.name.find("ffn/LayerNorm/") != -1:
                        # name = var.name.replace("ffn/LayerNorm/", "output/LayerNorm/")
                        name = pre_name
                        ckpt[name] = val
                    elif var.name.find("ffn/conv1d_1/") != -1:
                        # name = var.name.replace("ffn/conv1d_1/", "output/dense/")
                        name = pre_name
                        ckpt[name] = val
                        
                    elif var.name.find("transformer/decoder/w_embs") != -1:
                        name = var.name.replace("decoder", "decoding")
                        ckpt[name] = val
                        
                    elif var.name.find("transformer/decoder/dense/") != -1:
                        name = var.name.replace("decoder", "decoding")
                        ckpt[name] = val
                        
                    elif var.name.find("transformer/decoder/LayerNorm/") != -1:
                        name = var.name.replace("decoder", "decoding")
                        ckpt[name] = val

                if name != None:
                    print("[INFO] {} -> {} ".format(var.name, name))
                
        for key in ckpt:
            print(key)
        with open('model.pkl', 'wb') as f:
            pickle.dump(ckpt, f, 0)

    dumpModel_new()

