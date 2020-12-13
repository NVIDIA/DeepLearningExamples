#!/usr/bin/env python3

# Modified MIT License

# Software Copyright (c) 2019 OpenAI

# We don’t claim ownership of the content you create with GPT-2, so it is yours to do with as you please.
# We only ask that you use GPT-2 responsibly and clearly indicate your content was created using GPT-2.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# The above copyright notice and this permission notice need not be included
# with content created by the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

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

import fire
import json
import os
import numpy as np
import tensorflow as tf

from tensorflow.contrib.training import HParams
from utils import gpt2_token_encoder as encoder
from utils.common import TransformerArgument
from utils.common import DecodingGpt2Argument
from utils.common import time_test

def sample_model(
    model_name='124M',
    nsamples=1,
    batch_size=1,
    length=12,
    temperature=1,
    top_k=4,
    top_p=0,
    models_dir='models',
    data_type='fp32'
):
    """Run the sample_model.

    :model_name=124M : String, which model to use
    :nsamples=0 : Number of samples to return, if 0, continues to
     generate samples indefinately.
    :batch_size=1 : Number of batches (only affects speed/memory).
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=4 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """

    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = HParams(n_vocab=0,
                      n_ctx=1024,
                      n_embd=768,
                      n_head=12,
                      n_layer=12)
    
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    
    # start_ids has shape [batch_size, start_len].flatten()
    # start_ids = [15496, 11, 616, 3290, 468,
    #             15496, 11, 616, 3290, 469,
    #             15496, 11, 616, 3290, 470,
    #             15496, 11, 616, 3290, 471]
    start_ids = [enc.encoder['<|endoftext|>'] for i in range(batch_size)]

    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph("{}/{}/model.ckpt.meta".format(models_dir, model_name))
        print("[INFO] restore the model {}/{}".format(models_dir, model_name))
        saver.restore(sess, ("{}/{}/model.ckpt".format(models_dir, model_name)))

        if data_type == 'fp32':
            tf_data_type = tf.float32
        elif data_type == 'fp16':
            tf_data_type = tf.float16
        else:
            assert(False)
        
        decoder_args = TransformerArgument(beam_width=1,
                                           head_num=hparams.n_head,
                                           size_per_head=hparams.n_embd // hparams.n_head,
                                           num_layer=hparams.n_layer,
                                           dtype=tf_data_type,
                                           kernel_init_range=0.00,
                                           bias_init_range=0.00)

        decoding_args = DecodingGpt2Argument(hparams.n_vocab,
                                             enc.encoder['<|endoftext|>'],
                                             enc.encoder['<|endoftext|>'],
                                             length + 2,
                                             decoder_args,
                                             top_k,
                                             top_p,
                                             temperature)
        
        ckpt_dict = {}
        for var in tf.trainable_variables():
            ckpt_dict[var.name] = var
        decoding_vars = tf.trainable_variables()
        
        op_output = ft_gpt2_op(decoding_vars,
                               decoding_args,
                               batch_size,
                               start_ids)

        generated = 0
        
        while nsamples == 0 or generated < nsamples:
            print("[INFO] FT op time: {}".format(time_test(sess, op_output, iterations=5, warmup=True)))
            op_out = sess.run(op_output)

            for i in range(batch_size):
                generated += 1
                
                text = enc.decode(op_out[i][1:])
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)

def preprocess_decoder_var(decoding_vars,
                            num_layer,
                            using_model_var,
                            checkpoint_filename,
                            data_type,
                            fuse_qkv=True):
    '''
    Args:
        decoding_vars: A list of tf.Tensor. The variables of decoding.  
        num_layer: A int value. The number of transformer layer of decoder in decoding
        using_model_var: A bool value. Using the model variables of TensorFlow or not.
                         If True, then putting the model variables of TensorFlow decoding model into decoding op directly. 
                            The data type is tensor of TensorFlow in this case. 
                        
                         If False, then restoring the values of variables from the checkpoint_filename, and putting
                         the values into decoding op.
                            The data type is numpy is this case. 
        checkpoint_file: A string. The checkpoint file name of storing the values of model. The checkpoint should be stored in 
                         pickle, and the name of checkpoint should be xxx.pkl.
                         The model is saved by dict. 
                         The key of the dict is the name of variables
                         The value of the dict is the values of variables
                         For example, decoding_vars[0]=<tf.Variable 'transformer/decoder/layer_0/masked_multi_head/LayerNorm/beta:0' shape=(512,) dtype=float32_ref>,
                         then the key is 'transformer/decoder/layer_0/masked_multi_head/LayerNorm/beta:0'; the value is sess.run(decoding_vars[0])
        data_type: tf.float32 or tf.float16. 
                   Only used when using_model_var is False. Convert the numpy data to the data type of model.
                   
    Outputs:
        vars_in_diff_layers_dict: A dict to store the variables by their name.
                                
                                For decoder variables, the key is like 'transformer/decoder/layer/masked_multi_head/LayerNorm/beta:0', 
                                which is similar to the name of variables, except we use 'layer' but not 'layer_x'. The value is a list, 
                                which contains 'transformer/decoder/layer_%d/masked_multi_head/LayerNorm/beta:0' % i for i in range(num_layer)
                                
                                For other variables, the key is the name of variable, and the value is the correspoding weight.
                                
                                Note that we return the concated weights. The concat operation would bring other overhead, and this should be optimized in 
                                the real application. The recommended method is pre-processing the weights as numpy format. Because TensorFlow do the operations
                                for each inference if using the TensorFlow to pre-process the weights.
    '''
    
    var_dict = {}
    for var in decoding_vars:
        var_dict[var.name] = var
    
    vars_in_diff_layers_dict = {}
    vars_in_diff_layers_dict["transformer/decoder/LayerNorm/beta:0"] = tf.cast(var_dict["model/ln_f/b:0"], dtype=data_type)
    vars_in_diff_layers_dict["transformer/decoder/LayerNorm/gamma:0"] = tf.cast(var_dict["model/ln_f/g:0"], dtype=data_type)
    vars_in_diff_layers_dict["model/wpe:0"] = tf.cast(var_dict["model/wpe:0"], dtype=data_type)
    vars_in_diff_layers_dict["model/wte:0"] = tf.cast(var_dict["model/wte:0"], dtype=data_type)

    for i in range(num_layer):
        """Handling the names of q, k, v kernel and bias because their names
        are different for fusing the qkv or not."""
        
        layer_prefix_name = "transformer/decoder/layer_%d/" % i
        gpt2_layer_prefix_namx = "model/h%d/" % i

        var_dict[layer_prefix_name + 'masked_multi_head/query/kernel:0'], \
        var_dict[layer_prefix_name + 'masked_multi_head/key/kernel:0'], \
        var_dict[layer_prefix_name + 'masked_multi_head/value/kernel:0'] = tf.split(var_dict[gpt2_layer_prefix_namx + 'attn/c_attn/w:0'], 3, axis=-1)

        var_dict[layer_prefix_name + 'masked_multi_head/query/bias:0'], \
        var_dict[layer_prefix_name + 'masked_multi_head/key/bias:0'], \
        var_dict[layer_prefix_name + 'masked_multi_head/value/bias:0'] = tf.split(var_dict[gpt2_layer_prefix_namx + 'attn/c_attn/b:0'], 3, axis=-1)
            

    layer_prefix_name = 'transformer/decoder/layer'
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/LayerNorm/beta:0'] = \
        tf.cast(tf.concat([ var_dict['model/h%d/ln_1/b:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/LayerNorm/gamma:0'] = \
        tf.cast(tf.concat([ var_dict['model/h%d/ln_1/g:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/query/kernel:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/query/kernel:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/query/bias:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/query/bias:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/key/kernel:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/key/kernel:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/key/bias:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/key/bias:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/value/kernel:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/value/kernel:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/value/bias:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/value/bias:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)

    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/conv1d_1/kernel:0'] = \
        tf.cast(tf.concat([ var_dict['model/h%d/attn/c_proj/w:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/conv1d_1/bias:0'] = \
        tf.cast(tf.concat([ var_dict['model/h%d/attn/c_proj/b:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
        
    vars_in_diff_layers_dict[layer_prefix_name + '/ffn/LayerNorm/beta:0'] = \
        tf.cast(tf.concat([ var_dict['model/h%d/ln_2/b:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/ffn/LayerNorm/gamma:0'] = \
        tf.cast(tf.concat([ var_dict['model/h%d/ln_2/g:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
        
    vars_in_diff_layers_dict[layer_prefix_name + '/ffn/conv1d/kernel:0'] = \
        tf.cast(tf.concat([ var_dict['model/h%d/mlp/c_fc/w:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/ffn/conv1d/bias:0'] = \
        tf.cast(tf.concat([ var_dict['model/h%d/mlp/c_fc/b:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/ffn/conv1d_1/kernel:0'] = \
        tf.cast(tf.concat([ var_dict['model/h%d/mlp/c_proj/w:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/ffn/conv1d_1/bias:0'] = \
        tf.cast(tf.concat([ var_dict['model/h%d/mlp/c_proj/b:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    
    return vars_in_diff_layers_dict

def ft_gpt2_op(decoding_vars,
               decoding_args,
               batch_size,
               start_ids):
    """Run the decoding with sampling by FasterTransformer.

    Args:
        decoder_vars: A list of tf.Tensor. The variables for decoding. A list of model variables of TensorFlow model.
        decoder_args: The arguments for decoding. The details are in the class "DecodingGpt2Argument" of common.py
    Outputs:
        output_ids: A tf.Tensor with shape [batch_size, max(sequence_lengths)], with int type.
                    The results of decoding. It contains the id of token of vocabulary.
        sequence_lengths: A tf.Tensor with shape [batch_size], with int type.
    """
    decoder_args = decoding_args.decoder_args
    decoding_op_module = tf.load_op_library(os.path.join('./lib/libtf_gpt2.so'))
    data_type = decoder_args.dtype

    vars_dict_in_differ_layers = preprocess_decoder_var(decoding_vars,
                                                        decoder_args.num_layer,
                                                        True,
                                                        None,
                                                        data_type,
                                                        False)
    
    output_ids = decoding_op_module.decoding_gpt2(
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/LayerNorm/beta:0'], # 0
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/LayerNorm/gamma:0'], # 1
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/query/kernel:0'], # 2
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/query/bias:0'], # 3
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/key/kernel:0'], # 4
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/key/bias:0'], # 5
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/value/kernel:0'], # 6
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/value/bias:0'], # 7
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/conv1d_1/kernel:0'], # 8
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/conv1d_1/bias:0'],  # 9
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/LayerNorm/beta:0'], # 10
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/LayerNorm/gamma:0'], # 11
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/conv1d/kernel:0'], # 12
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/conv1d/bias:0'], # 13
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/conv1d_1/kernel:0'], # 14
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/conv1d_1/bias:0'], # 15
        vars_dict_in_differ_layers['transformer/decoder/LayerNorm/beta:0'], # 16
        vars_dict_in_differ_layers['transformer/decoder/LayerNorm/gamma:0'], # 17
        vars_dict_in_differ_layers['model/wte:0'], # 18
        vars_dict_in_differ_layers['model/wte:0'], # 19
        vars_dict_in_differ_layers['model/wpe:0'], # 20
        batch_size=batch_size,
        candidate_num=decoding_args.top_k,
        probability_threshold=decoding_args.top_p,
        max_seq_len=decoding_args.max_seq_len,
        head_num=decoder_args.head_num, 
        size_per_head=decoder_args.size_per_head,
        num_layer=decoder_args.num_layer,
        start_id=decoding_args.start_id, 
        start_ids=start_ids,
        end_id=decoding_args.end_id,
        temperature=decoding_args.temperature
    )
    
    output_ids = tf.transpose(output_ids, [1, 0])
    return output_ids

if __name__ == '__main__':
    fire.Fire(sample_model)

