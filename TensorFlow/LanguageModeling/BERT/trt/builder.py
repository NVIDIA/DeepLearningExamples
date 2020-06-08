#!/usr/bin/env python3
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

import tensorrt as trt
import ctypes
import argparse
import numpy as np
import json
import time
import sys
import re
import os
import os.path
from helpers.calibrator import BertCalibrator as BertCalibrator

try:
    from tensorflow.python import pywrap_tensorflow as pyTF
except ImportError as err:
    sys.stderr.write("""Error: Failed to import tensorflow module ({})\n""".format(err))
    sys.exit()

"""
TensorRT Initialization
"""
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

handle = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Could not load plugin library. Is `libnvinfer_plugin.so` on your LD_LIBRARY_PATH?")

trt.init_libnvinfer_plugins(TRT_LOGGER, "")
plg_registry = trt.get_plugin_registry()
qkv2_plg_creator = plg_registry.get_plugin_creator("CustomQKVToContextPluginDynamic", "1", "")
skln_plg_creator = plg_registry.get_plugin_creator("CustomSkipLayerNormPluginDynamic", "1", "")
gelu_plg_creator = plg_registry.get_plugin_creator("CustomGeluPluginDynamic", "1", "")
emln_plg_creator = plg_registry.get_plugin_creator("CustomEmbLayerNormPluginDynamic", "1", "")
fc_plg_creator = plg_registry.get_plugin_creator("CustomFCPluginDynamic", "1", "")


"""
Attentions Keys
"""
WQ = "query_kernel"
BQ = "query_bias"
WK = "key_kernel"
BK = "key_bias"
WV = "value_kernel"
BV = "value_bias"
WQKV = "qkv_kernel"
BQKV = "qkv_bias"


"""
Transformer Keys
"""
W_AOUT = "attention_output_dense_kernel"
B_AOUT = "attention_output_dense_bias"
AOUT_LN_BETA = "attention_output_layernorm_beta"
AOUT_LN_GAMMA = "attention_output_layernorm_gamma"
W_MID = "intermediate_dense_kernel"
B_MID = "intermediate_dense_bias"
W_LOUT = "output_dense_kernel"
B_LOUT = "output_dense_bias"
LOUT_LN_BETA = "output_layernorm_beta"
LOUT_LN_GAMMA = "output_layernorm_gamma"


"""
Squad Output Keys
"""
SQD_W = "squad_output_weights"
SQD_B = "squad_output_bias"

class BertConfig:
    def __init__(self, bert_config_path, use_fp16, use_int8, use_strict, use_fc2_gemm):
        with open(bert_config_path, 'r') as f:
            data = json.load(f)
            self.num_attention_heads = data['num_attention_heads']
            self.hidden_size = data['hidden_size']
            self.intermediate_size = data['intermediate_size']
            self.num_hidden_layers = data['num_hidden_layers']
            self.use_fp16 = use_fp16
            self.use_int8 = use_int8
            self.use_fc2_gemm = use_fc2_gemm
            self.use_strict = use_strict
            self.head_size = self.hidden_size // self.num_attention_heads



def set_tensor_name(tensor, prefix, name):
    tensor.name = prefix + name

def set_output_name(layer, prefix, name, out_idx = 0):
    layer.name = prefix + name
    set_tensor_name(layer.get_output(out_idx), prefix, name)

def make_gelu_layer(prefix, config, network, input_tensor):
    POW = network.add_constant((1, 1, 1, 1, 1), trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
    MULTIPLY = network.add_constant((1, 1, 1, 1, 1), trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
    SQRT = network.add_constant((1, 1, 1, 1, 1), trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
    ONE = network.add_constant((1, 1, 1, 1, 1), trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
    HALF = network.add_constant((1, 1, 1, 1, 1), trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
    X_pow = network.add_elementwise(input_tensor, POW.get_output(0), trt.ElementWiseOperation.POW)
    X_pow_t = X_pow.get_output(0)
    X_mul = network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
    X_add = network.add_elementwise(mid_dense_out, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
    X_sqrt = network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
    X_sqrt_tensor = X_sqrt.get_output(0)
    X_tanh = network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
    X_tanh_tensor = X_tanh.get_output(0)
    X_one = network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
    CDF = network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
    gelu_layer = network.add_elementwise(CDF.get_output(0), mid_dense_out, trt.ElementWiseOperation.PROD)

    # enable elementwise fusing for int8 && fp16
    POW.precision = trt.DataType.FLOAT
    MULTIPLY.precision = trt.DataType.FLOAT
    SQRT.precision = trt.DataType.FLOAT
    ONE.precision = trt.DataType.FLOAT
    HALF.precision = trt.DataType.FLOAT
    X_pow.precision = trt.DataType.FLOAT
    X_mul.precision = trt.DataType.FLOAT
    X_add.precision = trt.DataType.FLOAT
    X_sqrt.precision = trt.DataType.FLOAT
    X_tanh.precision = trt.DataType.FLOAT
    X_one.precision = trt.DataType.FLOAT
    CDF.precision = trt.DataType.FLOAT
    gelu_layer.precision = trt.DataType.FLOAT
    return gelu_layer



def attention_layer_opt(prefix, config, init_dict, network, input_tensor, imask):
    """
    Add the attention layer
    """
    assert(len(input_tensor.shape) == 5)
    B, S, hidden_size, _, _ = input_tensor.shape
    num_heads = config.num_attention_heads
    head_size = int(hidden_size / num_heads)

    Wall = init_dict[prefix + WQKV]
    Ball = init_dict[prefix + BQKV]

    # FC_attention
    if config.use_int8:
        mult_all = network.add_convolution(input_tensor, 3 * hidden_size, (1, 1), Wall, Ball)
    else:
        mult_all = network.add_fully_connected(input_tensor, 3 * hidden_size, Wall, Ball)

    set_output_name(mult_all, prefix, "qkv_mult")

    has_mask = imask is not None

    pf_type = trt.PluginField("type_id", np.array([1 if config.use_fp16 else 0], np.int32), trt.PluginFieldType.INT32)
    pf_hidden_size = trt.PluginField("hidden_size", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32)
    pf_num_heads = trt.PluginField("num_heads", np.array([num_heads], np.int32), trt.PluginFieldType.INT32)
    pf_has_mask = trt.PluginField("has_mask", np.array([has_mask], np.int32), trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([pf_hidden_size, pf_num_heads, pf_has_mask, pf_type])
    qkv2ctx_plug = qkv2_plg_creator.create_plugin("qkv2ctx", pfc)

    qkv_in = [mult_all.get_output(0)]
    if has_mask:
        qkv_in.append(imask)
    qkv2ctx = network.add_plugin_v2(qkv_in, qkv2ctx_plug)
    set_output_name(qkv2ctx, prefix, "context_layer")
    return qkv2ctx


def skipln(prefix, config, init_dict, network, input_tensor, skip, bias=None):
    """
    Add the skip layer
    """
    idims = input_tensor.shape
    assert len(idims) == 5
    hidden_size = idims[2]

    pf_ld = trt.PluginField("ld", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32)
    wbeta = init_dict[prefix + "beta"]
    pf_beta = trt.PluginField("beta", wbeta.numpy(), trt.PluginFieldType.FLOAT32)
    wgamma = init_dict[prefix + "gamma"]
    pf_gamma = trt.PluginField("gamma", wgamma.numpy(), trt.PluginFieldType.FLOAT32)
    pf_type = trt.PluginField("type_id", np.array([1 if config.use_fp16 else 0], np.int32), trt.PluginFieldType.INT32)

    fields = [pf_ld, pf_beta, pf_gamma, pf_type ]

    if bias:
        pf_bias = trt.PluginField("bias", bias.numpy(), trt.PluginFieldType.FLOAT32)
        fields.append(pf_bias)

    pfc = trt.PluginFieldCollection(fields)
    skipln_plug = skln_plg_creator.create_plugin("skipln", pfc)

    skipln_inputs = [input_tensor, skip]
    layer = network.add_plugin_v2(skipln_inputs, skipln_plug)
    return layer

def my_fc(config, network, input_tensor,out_dims, W):
    pf_out_dims = trt.PluginField('out_dims', np.array([out_dims], dtype=np.int32), trt.PluginFieldType.INT32)
    pf_W = trt.PluginField('W', W.numpy(), trt.PluginFieldType.FLOAT32)
    pf_type = trt.PluginField("type_id", np.array([1 if config.use_fp16 else 0], np.int32), trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([pf_out_dims, pf_W, pf_type])
    fc_plugin = fc_plg_creator.create_plugin('fcplugin', pfc)
    plug_inputs = [input_tensor]
    out_dense = network.add_plugin_v2(plug_inputs, fc_plugin)
    return out_dense


def transformer_layer_opt(prefix, config, init_dict, network, input_tensor, imask):
    """
    Add the transformer layer
    """
    idims = input_tensor.shape
    assert len(idims) == 5
    hidden_size = idims[2]

    B_noop = trt.Weights()

    context_transposed = attention_layer_opt(prefix + "attention_self_", config, init_dict, network, input_tensor, imask)
    attention_heads = context_transposed.get_output(0)

    # FC0
    B_aout = init_dict[prefix + B_AOUT]
    if config.use_int8:
        W_aout = init_dict[prefix + W_AOUT]
        attention_out_fc = network.add_convolution(attention_heads, hidden_size, (1, 1), W_aout, B_aout)
        B_aout = None

        if config.use_fp16:
            attention_out_fc.precision = trt.DataType.INT8
            attention_out_fc.set_output_type(0, trt.DataType.HALF)
    else:
        W_aoutT = init_dict[prefix + W_AOUT + '_notrans']
        attention_out_fc = my_fc(config, network, attention_heads, hidden_size, W_aoutT)

    skiplayer = skipln(prefix + "attention_output_layernorm_",config, init_dict, network, attention_out_fc.get_output(0), input_tensor, B_aout)
    attention_ln = skiplayer.get_output(0)

    # FC1 + GELU
    B_mid = init_dict[prefix + B_MID]
    W_midT = init_dict[prefix + W_MID + '_notrans']
    mid_dense = my_fc(config, network, attention_ln, config.intermediate_size, W_midT)
    mid_dense_out = mid_dense.get_output(0)

    pf_type = trt.PluginField("type_id", np.array([1 if config.use_fp16 else 0], np.int32), trt.PluginFieldType.INT32)
    pf_bias = trt.PluginField("bias", B_mid.numpy(), trt.PluginFieldType.FLOAT32)
    pfc = trt.PluginFieldCollection([pf_type, pf_bias])

    plug = gelu_plg_creator.create_plugin("gelu", pfc)

    gelu_layer = network.add_plugin_v2([mid_dense_out], plug)

    intermediate_act = gelu_layer.get_output(0)
    set_tensor_name(intermediate_act, prefix, "gelu")
    
    if config.use_int8 and config.use_strict:
        intermediate_act.set_dynamic_range(-10, 10)

    # FC2
    # Dense to hidden size
    B_lout = init_dict[prefix + B_LOUT]
    if config.use_int8 and config.use_strict and not config.use_fc2_gemm:
        W_lout = init_dict[prefix + W_LOUT]
        out_dense = network.add_convolution(intermediate_act, hidden_size, (1, 1), W_lout, B_lout)
        B_lout = None
    else:
        W_loutT = init_dict[prefix + W_LOUT + '_notrans']
        out_dense = my_fc(config, network, intermediate_act, hidden_size, W_loutT)

    set_output_name(out_dense, prefix + "output_", "dense")
    out_layer = skipln(prefix + "output_layernorm_", config, init_dict, network, out_dense.get_output(0), attention_ln, B_lout)
    out_ln = out_layer.get_output(0)

    set_tensor_name(out_ln, prefix + "output_", "reshape")

    return out_ln


def bert_model(config, init_dict, network, input_tensor, input_mask):
    """
    Create the bert model
    """
    prev_input = input_tensor
    for layer in range(0, config.num_hidden_layers):
        ss = "l{}_".format(layer)
        prev_input = transformer_layer_opt(ss, config,  init_dict, network, prev_input, input_mask)
    return prev_input


def squad_output(prefix, config, init_dict, network, input_tensor):
    """
    Create the squad output
    """

    idims = input_tensor.shape
    assert len(idims) == 5
    B, S, hidden_size, _, _ = idims

    W_out = init_dict[prefix + SQD_W]
    B_out = init_dict[prefix + SQD_B]

    W = network.add_constant((1, hidden_size, 2), W_out)
    dense = network.add_fully_connected(input_tensor, 2, W_out, B_out)

    OUT = network.add_shuffle(dense.get_output(0))
    OUT.second_transpose = (1, 0, 2, 3, 4)
    set_output_name(OUT, prefix, "squad_logits")
    return OUT


def load_weights(inputbase, config):
    """
    Load the weights from the tensorflow checkpoint
    """
    weights_dict = dict()

    try:
        reader = pyTF.NewCheckpointReader(inputbase)
        tensor_dict = reader.get_variable_to_shape_map()

        # There might be training-related variables in the checkpoint that can be discarded
        param_names = [key for key in sorted(tensor_dict) if 'adam' not in key and 'global_step' not in key and 'pooler' not in key]
        count = len(param_names)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Found {:} entries in weight map".format(count))

        for pn in param_names:
            toks = pn.lower().split('/')
            if 'encoder' in pn:
                assert ('layer' in pn)
                l = (re.findall('\d+', pn))[0]
                outname = 'l{}_'.format(l) + '_'.join(toks[3:])
            else:
                outname = '_'.join(toks)

            tensor = reader.get_tensor(pn)
            shape = tensor.shape
            if pn.find('kernel') != -1:
                weights_dict[outname +'_notrans'] = trt.Weights(np.ascontiguousarray(tensor).flatten())

                TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Transposing {}\n".format(np))
                tensor = np.transpose(tensor)


            shape = tensor.shape
            flat_tensor = tensor.flatten()
            shape_str = '{} '.format(len(shape)) + ' '.join([str(d) for d in shape])
            weights_dict[outname] = trt.Weights(flat_tensor)

            TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Orig.name: {:}, TRT name: {:}, shape: {:}".format(pn, outname, shape_str))

        N = config.num_attention_heads
        H = config.head_size

        additional_dict = dict()
        for key, value in weights_dict.items():
            pos = key.find(BQ)
            if pos != -1:
                hidden_size = value.size
                prefix = key[:pos]

                Bq_ = value
                Bk_ = weights_dict[prefix + BK]
                Bv_ = weights_dict[prefix + BV]
                Wq_ = weights_dict[prefix + WQ]
                Wk_ = weights_dict[prefix + WK]
                Wv_ = weights_dict[prefix + WV]

                mat_size = hidden_size * hidden_size
                wcount = 3 * mat_size
                Wall = np.zeros(wcount, np.float32)
                bcount = 3 * hidden_size
                Ball = np.zeros(bcount, np.float32)
                Wall[0:mat_size] = Wq_.numpy()[0:mat_size]
                Wall[mat_size:2*mat_size] = Wk_.numpy()[0:mat_size]
                Wall[2*mat_size:3*mat_size] = Wv_.numpy()[0:mat_size]
                Ball[0:hidden_size] = Bq_.numpy()[0:hidden_size]
                Ball[hidden_size:2*hidden_size] = Bk_.numpy()[0:hidden_size]
                Ball[2*hidden_size:3*hidden_size] = Bv_.numpy()[0:hidden_size]

                Wall = np.ascontiguousarray(Wall.reshape((3,N,H,N,H)).transpose((1,0, 2,3,4)), dtype=np.float32)
                Ball = np.ascontiguousarray(Ball.reshape((3,N,H)).transpose((1,0, 2)), dtype=np.float32)

                additional_dict[prefix + WQKV] = trt.Weights(Wall)
                additional_dict[prefix + BQKV] = trt.Weights(Ball)

                additional_dict[prefix + WQKV + "_notrans"] = trt.Weights(Wall.T)

    except Exception as error:
        TRT_LOGGER.log(TRT_LOGGER.ERROR, str(error))

    weights_dict.update(additional_dict)
    return weights_dict


def emb_layernorm(builder, network, config, weights_dict, builder_config, sequence_length, batch_sizes):
    if len(batch_sizes) > 1:
        input_ids = network.add_input(name="input_ids", dtype=trt.int32, shape=(sequence_length, -1))
        segment_ids = network.add_input(name="segment_ids", dtype=trt.int32, shape=(sequence_length, -1))
        input_mask = network.add_input(name="input_mask", dtype=trt.int32, shape=(sequence_length, -1))

        # Specify profiles for the batch sizes we're interested in.
        # Make sure the profile also works for all sizes not covered by the previous profile.
        prev_size = 0
        for batch_size in sorted(batch_sizes):
            profile = builder.create_optimization_profile()
            min_shape = (sequence_length, prev_size + 1)
            shape = (sequence_length, batch_size)
            profile.set_shape("input_ids", min=min_shape, opt=shape, max=shape)
            profile.set_shape("segment_ids", min=min_shape, opt=shape, max=shape)
            profile.set_shape("input_mask", min=min_shape, opt=shape, max=shape)
            builder_config.add_optimization_profile(profile)
            prev_size = batch_size
    else:
        input_ids = network.add_input(name="input_ids", dtype=trt.int32, shape=(sequence_length, batch_sizes[0]))
        segment_ids = network.add_input(name="segment_ids", dtype=trt.int32, shape=(sequence_length, batch_sizes[0]))
        input_mask = network.add_input(name="input_mask", dtype=trt.int32, shape=(sequence_length, batch_sizes[0]))

    wbeta = trt.PluginField("bert_embeddings_layernorm_beta", weights_dict["bert_embeddings_layernorm_beta"].numpy(), trt.PluginFieldType.FLOAT32)
    wgamma = trt.PluginField("bert_embeddings_layernorm_gamma", weights_dict["bert_embeddings_layernorm_gamma"].numpy(), trt.PluginFieldType.FLOAT32)
    wwordemb = trt.PluginField("bert_embeddings_word_embeddings", weights_dict["bert_embeddings_word_embeddings"].numpy(), trt.PluginFieldType.FLOAT32)
    wtokemb = trt.PluginField("bert_embeddings_token_type_embeddings", weights_dict["bert_embeddings_token_type_embeddings"].numpy(), trt.PluginFieldType.FLOAT32)
    wposemb = trt.PluginField("bert_embeddings_position_embeddings", weights_dict["bert_embeddings_position_embeddings"].numpy(), trt.PluginFieldType.FLOAT32)

    output_fp16 = trt.PluginField("output_fp16", np.array([1 if config.use_fp16 else 0]).astype(np.int32), trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([wbeta, wgamma, wwordemb, wtokemb, wposemb, output_fp16])
    fn = emln_plg_creator.create_plugin("embeddings", pfc)

    inputs = [input_ids, segment_ids, input_mask]
    emb_layer = network.add_plugin_v2(inputs, fn)
    set_output_name(emb_layer, "embeddings_", "output")
    return emb_layer


def build_engine(batch_sizes, sequence_length, config, weights_dict, squad_json, vocab_file, calibrationCacheFile, calib_num):
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
        builder_config.max_workspace_size = 5000 * (1024 * 1024) # 5000 MiB
        if config.use_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
        if config.use_int8:
            calibrator = BertCalibrator(squad_json, vocab_file, calibrationCacheFile, 1, sequence_length, calib_num)
            builder_config.set_flag(trt.BuilderFlag.INT8)
            builder_config.int8_calibrator = calibrator
        if config.use_strict:
            builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # Create the network
        emb_layer = emb_layernorm(builder, network, config, weights_dict, builder_config, sequence_length, batch_sizes)
        embeddings = emb_layer.get_output(0)
        mask_idx = emb_layer.get_output(1)

        bert_out = bert_model(config, weights_dict, network, embeddings, mask_idx)

        squad_logits = squad_output("cls_", config, weights_dict, network, bert_out)
        squad_logits_out = squad_logits.get_output(0)

        network.mark_output(squad_logits_out)

        build_start_time = time.time()
        engine = builder.build_engine(network, builder_config)
        build_time_elapsed = (time.time() - build_start_time)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "build engine in {:.3f} Sec".format(build_time_elapsed))
        if config.use_int8:
            calibrator.free()
        return engine

def generate_calibration_cache(sequence_length, config, weights_dict, squad_json, vocab_file, calib_num):
    # dynamic shape not working with calibration, so we need generate a calibration cache first using fulldims network
    calibrationCacheFile = "bertSquadCalibCache"
    if not config.use_int8 or os.path.exists(calibrationCacheFile):
        return calibrationCacheFile

    # generate calibration cache
    saved_use_fp16 = config.use_fp16
    config.use_fp16 = False

    with build_engine([1], sequence_length, config, weights_dict, squad_json, vocab_file, calibrationCacheFile, calib_num) as engine:
        TRT_LOGGER.log(TRT_LOGGER.INFO, "calibration cache generated in {:}".format(calibrationCacheFile))

    config.use_fp16 = saved_use_fp16
    return calibrationCacheFile

def main():
    parser = argparse.ArgumentParser(description='TensorRT BERT Sample', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--ckpt', required=True,
                        help='The checkpoint file basename, e.g.: basename(model.ckpt-766908.data-00000-of-00001) is model.ckpt-766908')
    parser.add_argument('-o', '--output', required=True, default="bert_base_384.engine", help='The bert engine file, ex bert.engine')
    parser.add_argument('-b', '--batch-size', default=[], action="append", help='Batch size(s) to optimize for. The engine will be usable with any batch size below this, but may not be optimal for smaller sizes. Can be specified multiple times to optimize for more than one batch size.', type=int)
    parser.add_argument('-s', '--sequence-length', default=128, help='Sequence length of the BERT model', type=int)
    parser.add_argument('-c', '--config-dir', required=True,
                        help='The folder containing the bert_config.json, which can be downloaded e.g. from https://github.com/google-research/bert#pre-trained-models or by running download_models.py in dle/TensorFlow/LanguageModeling/BERT/data/pretrained_models_google')
    parser.add_argument('-f', '--fp16', action='store_true', help='Indicates that inference should be run in FP16 precision', required=False)
    parser.add_argument('-i', '--int8', action='store_true', help='Indicates that inference should be run in INT8 precision', required=False)
    parser.add_argument('-t', '--strict', action='store_true', help='Indicates that inference should be run in strict precision mode', required=False)
    parser.add_argument('-j', '--squad-json', default='squad/dev-v1.1.json', help='squad json dataset used for int8 calibration', required=False)
    parser.add_argument('-v', '--vocab-file', default='./pre-trained_model/uncased_L-24_H-1024_A-16/vocab.txt', help='Path to file containing entire understandable vocab', required=False)
    parser.add_argument('-n', '--calib-num', default=100, help='calibration batch numbers', type=int)
    parser.add_argument('-g', '--force-fc2-gemm', action='store_true', help='Force use gemm to implement FC2 layer', required=False)

    args, _ = parser.parse_known_args()
    args.batch_size = args.batch_size or [1]

    bert_config_path = os.path.join(args.config_dir, 'bert_config.json')
    TRT_LOGGER.log(TRT_LOGGER.INFO, "Using configuration file: {:}".format(bert_config_path))
    config = BertConfig(bert_config_path, args.fp16, args.int8, args.strict, args.force_fc2_gemm)

    weights_dict = load_weights(args.ckpt, config)
    calib_cache = generate_calibration_cache(args.sequence_length, config, weights_dict, args.squad_json, args.vocab_file, args.calib_num)

    with build_engine(args.batch_size, args.sequence_length, config, weights_dict, args.squad_json, args.vocab_file, calib_cache, args.calib_num) as engine:
        TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Serializing Engine...")
        serialized_engine = engine.serialize()
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(args.output))
        with open(args.output, 'wb') as fout:
            fout.write(serialized_engine)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")


if __name__ == '__main__':
    main()
