# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import ctypes
import glob
import os
import pathlib
import sys
from collections import OrderedDict

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorrt import Dims, ElementWiseOperation, MatrixOperation, Weights

import fastspeech.trt.common as common
from fastspeech.trt import TRT_BASE_PATH, TRT_LOGGER
from fastspeech.trt.trt_inferencer import TRTInferencer
from fastspeech.utils.logging import tprint
from fastspeech.utils.nvtx import Nvtx
from fastspeech.utils.pytorch import (remove_module_in_state_dict,
                                      to_cpu_numpy, to_gpu_async)


class FastSpeechTRTInferencer(TRTInferencer):

    def __init__(self, model_name, model, data_loader, ckpt_path=None, ckpt_file=None,
                 trt_max_ws_size=1, trt_file_path=None, trt_force_build=False, use_fp16=False,
                 trt_max_input_seq_len=256, trt_max_output_seq_len=1024, validate_accuracy=False):
        self.trt_max_input_seq_len = trt_max_input_seq_len
        self.trt_max_output_seq_len = trt_max_output_seq_len
        self.validate_accuracy = validate_accuracy

        self.load_plugin(os.path.join(TRT_BASE_PATH, 'plugins/repeat/RepeatPlugin.so'))
        self.load_plugin(os.path.join(TRT_BASE_PATH, 'plugins/add_pos_enc/AddPosEncPlugin.so'))

        super(FastSpeechTRTInferencer, self).__init__(model_name, model, data_loader, ckpt_path, ckpt_file, trt_max_ws_size, trt_file_path, trt_force_build, use_fp16)

    def build_engine(self):
        engine = None
        if self.trt_file_path and os.path.isfile(self.trt_file_path) and not self.trt_force_build:
            with open(self.trt_file_path, 'rb') as f:
                engine_str = f.read()
            with trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(engine_str)

        if engine:
            tprint('TRT Engine Loaded from {} successfully.'.format(self.trt_file_path))
            return engine
        else:
            tprint('Loading TRT Engine from {} failed.'.format(self.trt_file_path))

        tprint('Building a TRT Engine..')

        engine = self.do_build_engine()
        tprint('TRT Engine Built.')
        if self.trt_file_path:
            with open(self.trt_file_path, 'wb') as f:
                f.write(engine.serialize())
            tprint('TRT Engine Saved in {}.'.format(self.trt_file_path))

        return engine

    def create_plugins(self):
        # create "adding positional encoding" plugin
        self.plugins['AddPosEncPlugin'] = self.get_plugin_creator(
            'AddPosEncPlugin').create_plugin('AddPosEncPlugin', trt.PluginFieldCollection())

        # create "repeat" plugin
        self.plugins['RepeatPlugin'] = self.get_plugin_creator('RepeatPlugin').create_plugin('RepeatPlugin', trt.PluginFieldCollection([
            trt.PluginField('maxOutputLength', np.array(
                [self.trt_max_output_seq_len], dtype=np.int32), trt.PluginFieldType.INT32)
        ]))

    def do_build_engine(self):
        weights = self.model.state_dict()
        weights = self.preprocess_weights(weights)

        self.create_plugins()

        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags) as network:
            builder.max_workspace_size = common.GiB(self.trt_max_ws_size)
            builder.fp16_mode = self.use_fp16
            # builder.strict_type_constraints = True
            network = self.populate_network(network, weights, self.batch_size, self.trt_max_input_seq_len, self.trt_max_output_seq_len)
            return builder.build_cuda_engine(network)

    def infer(self, acts=None):
        inputs = next(self.data_loader_iter)

        text_encoded = inputs["text_encoded"]  # (b, t)
        text_pos = inputs["text_pos"]  # (b, t)

        text_encoded = F.pad(text_encoded, pad=(0, self.trt_max_input_seq_len - text_encoded.size(1)))  # (b, t)
        text_pos = F.pad(text_pos, pad=(0, self.trt_max_input_seq_len - text_pos.size(1)))  # (b, t)

        text_mask = text_pos.ne(0)  # padded is False

        # TODO: process word emb in TRT if the API allows.
        with torch.no_grad():
            text_encoded = self.model.word_emb(text_encoded)
        
        if self.use_fp16:
            text_encoded = text_encoded.half()
            
        # create input/output buffers
        input_buffers = common.create_inputs_from_torch(self.engine, [text_encoded, text_mask])
        output_buffers = common.create_outputs_from_torch(self.engine)

        # execute
        # self.context.profiler = trt.Profiler()
        stream = cuda.Stream()
        bindings = [int(data.data_ptr()) for data in (input_buffers + output_buffers)]
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # self.context.execute(batch_size=self.batch_size, bindings=bindings)
        stream.synchronize()

        outputs = dict()
        outputs['mel'] = output_buffers[-2]
        outputs['mel_mask'] = output_buffers[-1]
        outputs['text'] = inputs["text_norm"]

        # activations for verifying accuracy.
        if acts is not None:
            act_names = common.trt_output_names(self.engine)
            n_acts = len(output_buffers) - 2  # exclude outputs(mel and mel_mask)
            for i in range(n_acts):
                acts[act_names[i]] = output_buffers[i]

        return outputs

    def add_activation_as_output(self, network, tensor, tensor_name):
        tensor.name = tensor_name
        network.mark_output(tensor=tensor)

    def populate_network(self, network, weights, batch_size, trt_max_input_seq_len, trt_max_output_seq_len):
        d_model = self.model.d_model

        ##
        # Inputs
        ##
        out_seq = network.add_input(
            name="input_seq", dtype=trt.float32, shape=(batch_size, trt_max_input_seq_len, d_model))  # (b, t, d_model)
        #
        zeros = network.add_constant(weights=Weights(
            np.zeros(shape=(batch_size, trt_max_input_seq_len, 1), dtype=np.float32)),
            shape=(batch_size, trt_max_input_seq_len, 1))  # (b, t, 1)
        out_zeros = zeros.get_output(0)  # (b, t, 1)
        seq = network.add_elementwise(input1=out_seq, input2=out_zeros, op=trt.ElementWiseOperation.SUM)
        out_seq = seq.get_output(0)  # (b, t, d_model)

        if self.validate_accuracy:
            self.add_activation_as_output(network, out_seq, "act.emb")
        #

        out_seq_mask = network.add_input(  # paddings are False
            name="input_mask", dtype=trt.bool, shape=(batch_size, trt_max_input_seq_len, 1))  # (b, t, 1)

        ##
        # Phoneme-side FFT Blocks
        ##

        # Positional Encoding
        # The plugin adds positional encoding to the padding values also (for better performance), whereas Pytorch impl does not.
        # It's fine because the padding values will be eventually masked out in coming layers, giving accurate output.
        seq = network.add_plugin_v2([out_seq], self.get_plugin('AddPosEncPlugin'))
        seq.name = "phoneme_side.add_pos_enc"
        out_seq = seq.get_output(0)  # (b, t, d_model)

        if self.validate_accuracy:
            self.add_activation_as_output(network, out_seq, "act.phoneme_side.add_pos_enc")
        
        for layer_idx in range(self.model.phoneme_side_n_layer):
            out_seq = self.populate_fft(name='phoneme_side.layer_stack.{}'.format(layer_idx),
                                        network=network,
                                        weights=weights,
                                        seq_tensor=out_seq,
                                        seq_mask_tensor=out_seq_mask,
                                        batch_size=self.batch_size,
                                        max_seq_len=trt_max_input_seq_len,
                                        d_model=d_model,
                                        n_heads=self.model.phoneme_side_head,
                                        d_k=self.model.phoneme_side.d_k,
                                        d_v=self.model.phoneme_side.d_v,
                                        self_attn_temp=self.model.phoneme_side.d_k**0.5,
                                        conv_filter_size=self.model.phoneme_side_conv1d_filter_size,
                                        conv_kernel_size=self.model.fft_conv1d_kernel,
                                        conv_padding=self.model.fft_conv1d_padding)

        if self.validate_accuracy:
            self.add_activation_as_output(network, out_seq, "act.phoneme_side.seq")

        out_seq, out_seq_mask, out_dur = self.populate_length_regulator(name="length_regulator",
                                                                        network=network,
                                                                        weights=weights,
                                                                        seq_tensor=out_seq,
                                                                        seq_mask_tensor=out_seq_mask,
                                                                        batch_size=batch_size,
                                                                        trt_max_input_seq_len=trt_max_input_seq_len,
                                                                        trt_max_output_seq_len=trt_max_output_seq_len,
                                                                        d_model=d_model)

        if self.validate_accuracy:
            self.add_activation_as_output(network, out_seq, "act.length_regulator.seq")
            self.add_activation_as_output(network, out_dur, "act.length_regulator.dur")

        ##
        # Mel-side FFT Blocks
        ##

        # Type int to bool: out_seq_mask. TODO: remove if bool output is allowed in the plugin.
        ones = network.add_constant(weights=Weights(
            np.ones(shape=(batch_size, trt_max_output_seq_len, 1), dtype=np.int32)),
            shape=(batch_size, trt_max_output_seq_len, 1))  # (b, t, 1)
        out_ones = ones.get_output(0)  # (b, t, 1)
        seq_mask = network.add_elementwise(input1=out_seq_mask,
                                           input2=out_ones,
                                           op=ElementWiseOperation.EQUAL)  # (b, t, 1)
        seq_mask.name = "mel_side.seq_mask"
        out_seq_mask = seq_mask.get_output(0)

        # Positional Encoding
        seq = network.add_plugin_v2([out_seq], self.get_plugin('AddPosEncPlugin'))
        seq.name = "mel_side.add_pos_enc"
        out_seq = seq.get_output(0)

        if self.validate_accuracy:
            self.add_activation_as_output(network, out_seq, "act.mel_side.add_pos_enc")

        for layer_idx in range(self.model.mel_side_n_layer):
            out_seq = self.populate_fft(name="mel_side.layer_stack.{}".format(layer_idx),
                                        network=network,
                                        weights=weights,
                                        seq_tensor=out_seq,
                                        seq_mask_tensor=out_seq_mask,
                                        batch_size=self.batch_size,
                                        max_seq_len=trt_max_output_seq_len,
                                        d_model=d_model,
                                        n_heads=self.model.mel_side_head,
                                        d_k=self.model.mel_side.d_k,
                                        d_v=self.model.mel_side.d_v,
                                        self_attn_temp=self.model.mel_side.d_k**0.5,
                                        conv_filter_size=self.model.mel_side_conv1d_filter_size,
                                        conv_kernel_size=self.model.fft_conv1d_kernel,
                                        conv_padding=self.model.fft_conv1d_padding)

        if self.validate_accuracy:
            self.add_activation_as_output(network, out_seq, "act.mel_side.seq")

        ##
        # Linear
        ##

        # Pytorch: self.mel_linear = nn.Linear(mel_side_output_size, n_mels, bias=True)
        w = weights["mel_linear.weight"]  # (n_mels, d_model)
        out_w = network.add_constant(shape=(1, self.model.n_mels, d_model), weights=trt.Weights(w)).get_output(0)  # (1, n_mels, d_model)
        linear_w = network.add_matrix_multiply(out_seq, MatrixOperation.NONE, out_w, MatrixOperation.TRANSPOSE)  # (b, t, d_model) * (1->b, d_model, n_mels) => (b, t, n_mels)
        linear_w.name = "linear.w"
        out_seq = linear_w.get_output(0)  # (b, t, n_mels)

        b = weights["mel_linear.bias"]  # (n_mels,)
        out_b = network.add_constant(shape=(1, 1, self.model.n_mels), weights=trt.Weights(b)).get_output(0)  # (1, 1, n_mels)
        linear_b = network.add_elementwise(input1=out_seq, input2=out_b, op=trt.ElementWiseOperation.SUM)
        linear_b.name = "linear.b"
        out_seq = linear_b.get_output(0)  # (b, t, n_mels)

        ##
        # Outputs
        ##

        if self.validate_accuracy:
            self.add_activation_as_output(network, out_seq_mask, "out.seq_mask")
            self.add_activation_as_output(network, out_seq, "out.seq")

        seq = network.add_shuffle(input=out_seq)  # (b, t, n_mels) to (b, n_mels, t)
        seq.reshape_dims = Dims((batch_size, trt_max_output_seq_len, self.model.n_mels))
        seq.second_transpose = trt.Permutation([0, 2, 1])
        seq.name = "trans_seq"
        out_seq = seq.get_output(0)

        seq_mask = network.add_shuffle(input=out_seq_mask)  # (b, t, 1) to (b, t)
        seq_mask.reshape_dims = Dims((batch_size, trt_max_output_seq_len))
        out_seq_mask = seq_mask.get_output(0)  # (b, t)

        network.mark_output(tensor=out_seq)  # (b, n_mels, t)
        network.mark_output(tensor=out_seq_mask)  # (b, t)

        return network

    def populate_fft(self, name, network, weights, seq_tensor, seq_mask_tensor, batch_size,
                     max_seq_len, d_model, n_heads, d_k, d_v, self_attn_temp,
                     conv_filter_size, conv_kernel_size, conv_padding):
        # Self attn
        out = self.populate_slf_attn("{}.slf_attn".format(name), network, weights, seq_tensor, seq_mask_tensor, batch_size,
                                     max_seq_len, d_model, n_heads, d_k, d_v)  # (b, t, d_model)

        # Masking
        zeros = network.add_constant(weights=Weights(
            np.zeros(shape=(batch_size, max_seq_len, 1), dtype=np.float32)),
            shape=(batch_size, max_seq_len, 1))  # (b, t, 1)
        out_zeros = zeros.get_output(0)  # (b, t, 1)
        seq = network.add_select(condition=seq_mask_tensor, then_input=out, else_input=out_zeros)
        seq.name = "{}.mask1".format(name)
        out = seq.get_output(0)  # (b, t, d_model)

        # Position-wise
        out = self.populate_pos_wise("{}.pos_ffn".format(name), network, weights, out,
                          batch_size, max_seq_len, d_model,
                          conv_filter_size, conv_kernel_size, conv_padding)  # (b, t, d_model)

        # Masking
        seq = network.add_select(condition=seq_mask_tensor, then_input=out, else_input=out_zeros)
        seq.name = "{}.mask2".format(name)
        out = seq.get_output(0)  # (b, t, d_model)

        if self.validate_accuracy:
            self.add_activation_as_output(network, out, "act.{}".format(name))

        return out

    def populate_slf_attn(self, name, network, weights, seq_tensor, seq_mask_tensor, batch_size,
                     max_seq_len, d_model, n_heads, d_k, d_v):
        d_qkv = d_k + d_k + d_v

        # Pytorch: x = self.linear(x)
        w = weights["{}.linear.weight".format(name)]  # (n_heads * d_qkv, d_model)
        out_w = network.add_constant(shape=(1, d_model, n_heads * d_qkv), weights=trt.Weights(w)).get_output(0)  # (1, n_heads * d_qkv, d_model)
        linear_w = network.add_matrix_multiply(seq_tensor, MatrixOperation.NONE, out_w, MatrixOperation.TRANSPOSE)  # (b, t, d_model) * (1->b, d_model, n_heads * d_qkv) => (b, t, n_heads * d_qkv)
        linear_w.name = "{}.linear.w".format(name)
        out = linear_w.get_output(0)  # (b, t, n_heads * d_qkv)

        b = weights["{}.linear.bias".format(name)]  # (n_heads * d_qkv,)
        out_b = network.add_constant(shape=(1, 1, n_heads * d_qkv), weights=trt.Weights(b)).get_output(0)  # (1, 1, n_heads * d_qkv)
        linear_b = network.add_elementwise(input1=out, input2=out_b, op=trt.ElementWiseOperation.SUM)
        linear_b.name = "{}.linear.b".format(name)
        out = linear_b.get_output(0)  # (b, t, n_heads * d_qkv)

        if self.validate_accuracy:
            self.add_activation_as_output(network, out, "act.{}.linear".format(name))

        trans1 = network.add_shuffle(input=out)  # (b, t, n_heads * d_qkv) to (b, n_heads, t, d_qkv)
        trans1.reshape_dims = Dims(
            (batch_size, max_seq_len, n_heads, d_qkv))
        trans1.second_transpose = trt.Permutation([0, 2, 1, 3])
        trans1.name = "{}.trans1".format(name)
        out = trans1.get_output(0)  # (b, n_heads, t, d_qkv)

        # if self.validate_accuracy:
        #     self.add_activation_as_output(network, out, "act.{}.reshape".format(name))

        q = network.add_slice(input=out,
                              start=Dims((0, 0, 0, 0)),
                              shape=Dims(
                                  (batch_size, n_heads, max_seq_len, d_k)),
                              stride=Dims((1, 1, 1, 1)))
        q.name = "{}.slide_q".format(name)

        k = network.add_slice(input=out,
                              start=Dims((0, 0, 0, d_k)),
                              shape=Dims(
                                  (batch_size, n_heads, max_seq_len, d_k)),
                              stride=Dims((1, 1, 1, 1)))
        k.name = "{}.slide_k".format(name)

        v = network.add_slice(input=out,
                              start=Dims((0, 0, 0, 2 * d_k)),
                              shape=Dims(
                                  (batch_size, n_heads, max_seq_len, d_k)),
                              stride=Dims((1, 1, 1, 1)))
        v.name = "{}.slide_v".format(name)

        out_q = q.get_output(0)  # (b, n_heads, t, d_q)
        out_k = k.get_output(0)  # (b, n_heads, t, d_k)
        out_v = v.get_output(0)  # (b, n_heads, t, d_v)

        # Pytorch: output, attn = self.attention(q, k, v, mask=mask)       
        out = self.populate_scaled_dot(
            name="{}.scaled_dot".format(name),  # (b, n_heads, t, d_k)
            network=network,
            q_tensor=out_q, 
            k_tensor=out_k, 
            v_tensor=out_v, 
            mask_tensor=seq_mask_tensor, 
            batch_size=batch_size, 
            max_seq_len=max_seq_len, 
            n_heads=n_heads, 
            temperature=d_k**0.5)

        # Pytorch:
        # output = output.view(self.n_head, bs, seq_len, self.d_v)
        # output = output.permute(1, 2, 0, 3).contiguous().view(bs, seq_len, self.n_head * self.d_v)
        trans2 = network.add_shuffle(input=out)  # b, n_heads, t, d_k) to (b, t, n_heads * d_k)
        trans2.first_transpose = trt.Permutation([0, 2, 1, 3])
        trans2.reshape_dims = Dims((batch_size, max_seq_len, n_heads * d_v))
        trans2.name = "{}.trans2".format(name)
        out = trans2.get_output(0)  # (b, t, n_heads * d_k)

        if self.validate_accuracy:
            self.add_activation_as_output(network, out, "act.{}.scaled_dot".format(name))

        # Pytorch: output = self.fc(output)
        w = weights["{}.fc.weight".format(name)]  # (d_model, n_heads * d_v)
        out_w = network.add_constant(shape=(1, d_model, n_heads * d_v), weights=trt.Weights(w)).get_output(0)  # (1, d_model, n_heads * d_v)
        fc_w = network.add_matrix_multiply(out, MatrixOperation.NONE, out_w, MatrixOperation.TRANSPOSE)  # (b, t, n_heads * d_k) * (1->b, n_heads * d_k, d_model) => (b, t, d_model)
        fc_w.name = "{}.fc.w".format(name)
        out = fc_w.get_output(0)  # (b, t, d_model)

        b = weights["{}.fc.bias".format(name)]  # (d_model,)
        out_b = network.add_constant(shape=(1, 1, n_heads * d_qkv), weights=trt.Weights(b)).get_output(0)  # (1, 1, d_model)
        fc_b = network.add_elementwise(input1=out, input2=out_b, op=trt.ElementWiseOperation.SUM)
        fc_b.name = "{}.fc.b".format(name)
        out = fc_b.get_output(0)  # (b, t, d_model)

        # if self.validate_accuracy:
        #     self.add_activation_as_output(network, out, "act.{}.fc".format(name))

        # Pytorch: output += residual
        residual = network.add_elementwise(input1=seq_tensor, input2=out, op=ElementWiseOperation.SUM)
        residual.name = "{}.residual".format(name)
        out = residual.get_output(0)  # (b, t, d_model)

        if self.validate_accuracy:
            self.add_activation_as_output(network, out, "act.{}.residual".format(name))

        # Pytorch: output = self.layer_norm(output)
        out = self.populate_layernorm(name="{}.layer_norm".format(name),
                                      network=network,
                                      weights=weights,
                                      seq_tensor=out,
                                      batch_size=self.batch_size,
                                      max_seq_len=max_seq_len,
                                      d_layer=d_model,
                                      )  # (b, t, d_model)

        if self.validate_accuracy:
            self.add_activation_as_output(network, out, "act.{}.ln".format(name))

        return out       
        

    def populate_scaled_dot(self, name, network, q_tensor, k_tensor, v_tensor, mask_tensor, batch_size, max_seq_len, n_heads, temperature):
        # if self.validate_accuracy:
        #     self.add_activation_as_output(network, q_tensor, "act.{}.q".format(name))
        #     self.add_activation_as_output(network, k_tensor, "act.{}.k".format(name))
        #     self.add_activation_as_output(network, v_tensor, "act.{}.v".format(name))

        # Pytorch: attn = self.bmm1(q, k.transpose(1, 2))
        attn = network.add_matrix_multiply(q_tensor, MatrixOperation.NONE, k_tensor, MatrixOperation.TRANSPOSE)  # (b, n, t, d_k) * (b, n, d_k, t) = (b, n, t, t)
        attn.name = "{}.bmm1".format(name)
        out = attn.get_output(0)

        # if self.validate_accuracy:
        #     self.add_activation_as_output(network, out, "act.{}.bmm1".format(name))

        # Pytorch: attn = attn / self.temperature
        temperature = network.add_constant(weights=Weights(np.full((batch_size, n_heads, max_seq_len, max_seq_len), temperature, dtype=np.float32)),
                                           shape=Dims((batch_size, n_heads, max_seq_len, max_seq_len)))  # (b, n, t, t)
        output_temperature = temperature.get_output(0)

        attn = network.add_elementwise(input1=out, input2=output_temperature, op=ElementWiseOperation.DIV)  # (b, n, t, t)
        attn.name = "{}.div".format(name)
        out = attn.get_output(0)

        # Pytorch: attn = attn.masked_fill(mask, -65504)
        minus_inf = network.add_constant(weights=Weights(np.full((batch_size, n_heads, max_seq_len, max_seq_len), -65504, dtype=np.float32)),
                                       shape=Dims((batch_size, n_heads, max_seq_len, max_seq_len)))  # (b, n, t, t)
        output_minus_inf = minus_inf.get_output(0)
        mask = network.add_shuffle(input=mask_tensor)
        mask.reshape_dims = Dims((batch_size, 1, 1, max_seq_len))  # (b, t, 1) -> (b, 1, 1, t)
        mask.name = "{}.mask_reshape".format(name)
        mask_tensor = mask.get_output(0)
        attn = network.add_select(condition=mask_tensor, # (b, 1->n, 1, t)
                                  then_input=out, # (b, n, t, t)
                                  else_input=output_minus_inf)  # (b, n, t, t)
        attn.name = "{}.mask".format(name)
        out = attn.get_output(0)

        # if self.validate_accuracy:
        #     self.add_activation_as_output(network, out, "act.{}.masked_fill".format(name))

        # Pytorch: attn = self.softmax(attn)
        softmax = network.add_softmax(input=out)
        softmax.axes = (1 << 3)  # dim=3
        softmax.name = "{}.softmax".format(name)
        out = softmax.get_output(0)

        # if self.validate_accuracy:
        #     self.add_activation_as_output(network, out, "act.{}.softmax".format(name))

        # Pytorch: output = self.bmm2(attn, v)
        attn = network.add_matrix_multiply(out, MatrixOperation.NONE, v_tensor, MatrixOperation.NONE)  # (b, n, t, t) * (b, n, t, d_k) => (b, n, t, d_k)
        attn.name = "{}.bmm2".format(name)
        out = attn.get_output(0)

        # if self.validate_accuracy:
        #     self.add_activation_as_output(network, out, "act.{}.bmm2".format(name))

        return out


    def populate_pos_wise(self, name, network, weights, seq_tensor,
                          batch_size, max_seq_len, d_model,
                          conv_filter_size, conv_kernel_size, conv_padding):
        # Pytorch: output = x.transpose(1, 2)
        trans1 = network.add_shuffle(input=seq_tensor)  # (b, t, d_model) to (b, d_model, t, 1)
        trans1.first_transpose = trt.Permutation([0, 2, 1])
        trans1.reshape_dims = Dims((batch_size, d_model, max_seq_len, 1))
        trans1.name = "{}.trans1".format(name)
        out = trans1.get_output(0)  # (b, d_model, t, 1)

        # Pytorch: output = self.w_1(output)
        conv1_w = weights["{}.w_1.weight".format(name)]  # (1, conv_filter_size, d_model, conv_kernel_size, 1)
        conv1_b = weights["{}.w_1.bias".format(name)]  # (cov_filter_size,)
        conv1 = network.add_convolution(input=out, num_output_maps=conv_filter_size, kernel_shape=trt.DimsHW(conv_kernel_size, 1),
                                        kernel=Weights(conv1_w), bias=Weights(conv1_b))
        conv1.padding = trt.DimsHW(1, 0)
        conv1.name = "{}.conv1".format(name)
        out = conv1.get_output(0)  # (b, conv_filter_size, t, 1)

        if self.validate_accuracy:
            self.add_activation_as_output(network, out, "act.{}.conv1".format(name))

        # Pytorch: output = F.relu(output)
        relu = network.add_activation(input=out, type=trt.ActivationType.RELU)
        relu.name = "{}.relu".format(name)
        out = relu.get_output(0)  # (b, conv_filter_size, t, 1)

        # Pytorch: output = self.w_2(output)
        conv2_w = weights["{}.w_2.weight".format(name)]  # (1, d_model, conv_filter_size, conv_kernel_size, 1)
        conv2_b = weights["{}.w_2.bias".format(name)]  # (d_model, )
        conv2 = network.add_convolution(input=out, num_output_maps=d_model, kernel_shape=trt.DimsHW(conv_kernel_size, 1),
                                        kernel=Weights(conv2_w), bias=Weights(conv2_b))
        conv2.padding = trt.DimsHW(1, 0)
        conv2.name = "{}.conv2".format(name)
        out = conv2.get_output(0)  # (b, d_model, t, 1)

        if self.validate_accuracy:
            self.add_activation_as_output(network, out, "act.{}.conv2".format(name))

        # Pytorch: output = output.transpose(1, 2)
        trans2 = network.add_shuffle(input=out)  # (b, d_model, t, 1) to (b, t, d_model)
        trans2.first_transpose = trt.Permutation([0, 2, 1, 3])
        trans2.reshape_dims = Dims((batch_size, max_seq_len, d_model))
        trans2.name = "{}.trans2".format(name)
        out = trans2.get_output(0)  # (b, t, d_model)

        # Pytorch: output += residual
        residual = network.add_elementwise(input1=seq_tensor, input2=out, op=trt.ElementWiseOperation.SUM)
        residual.name = "{}.residual".format(name)
        out = residual.get_output(0)  # (b, t, d_model)

        if self.validate_accuracy:
            self.add_activation_as_output(network, out, "act.{}.residual".format(name))

        # Pytorch: output = self.layer_norm(output)
        out = self.populate_layernorm(name="{}.layer_norm".format(name),
                                      network=network,
                                      weights=weights,
                                      seq_tensor=out,
                                      batch_size=self.batch_size,
                                      max_seq_len=max_seq_len,
                                      d_layer=d_model,
                                      )  # (b, t, d_model)

        if self.validate_accuracy:
            self.add_activation_as_output(network, out, "act.{}.ln".format(name))

        return out

    def populate_length_regulator(self, name, network, weights, seq_tensor, seq_mask_tensor, batch_size, trt_max_input_seq_len, trt_max_output_seq_len, d_model):
        out_dur = self.populate_duration_predictor(name="{}.duration_predictor".format(name),
                                                   network=network,
                                                   weights=weights,
                                                   seq_tensor=seq_tensor,
                                                   seq_mask_tensor=seq_mask_tensor,
                                                   batch_size=batch_size,
                                                   max_seq_len=trt_max_input_seq_len,
                                                   d_model=d_model)  # (b, t)

        # Pytorch: output.append(torch.repeat_interleave(input[i], repeats, dim=0))
        seq = network.add_plugin_v2([seq_tensor, out_dur], self.get_plugin('RepeatPlugin'))
        seq.name = "{}.repeat_seq".format(name)
        out_seq = seq.get_output(0)  # (b, t, d), (b, t) => (b, t', d), dtype: float32

        # Type bool to int: seq_mask_tensor. TODO: remove if bool input is allowed in the plugin.
        zeros = network.add_constant(weights=Weights(
            np.zeros(shape=(batch_size, trt_max_input_seq_len, 1), dtype=np.int32)),
            shape=(batch_size, trt_max_input_seq_len, 1))
        out_zeros = zeros.get_output(0)  # (b, t, 1)
        ones = network.add_constant(weights=Weights(
            np.ones(shape=(batch_size, trt_max_input_seq_len, 1), dtype=np.int32)),
            shape=(batch_size, trt_max_input_seq_len, 1))
        out_ones = ones.get_output(0)  # (b, t, 1)
        seq_mask = network.add_select(condition=seq_mask_tensor, then_input=out_ones, else_input=out_zeros)
        seq_mask.name = "{}.seq_mask".format(name)
        out_seq_mask = seq_mask.get_output(0)  # (b, t, 1)

        seq_mask = network.add_plugin_v2([out_seq_mask, out_dur], self.get_plugin('RepeatPlugin'))
        seq_mask.name = "{}.repeat_seq_mask".format(name)
        out_seq_mask = seq_mask.get_output(0)  # (b, t, 1), (b, t) => (b, t', 1), dtype: int32

        return out_seq, out_seq_mask, out_dur

    def populate_duration_predictor(self, name, network, weights, seq_tensor, seq_mask_tensor, batch_size, max_seq_len, d_model):
        duration_predictor_filter_size=self.model.duration_predictor_filter_size
        duration_predictor_kernel_size=self.model.duration_predictor_kernel_size

        # Pytorch: input *= input_mask.to(input.dtype)
        # can be skipped.
        
        # Pytorch: out = self.conv1d_1(input.transpose(1,2)).transpose(1,2)
        trans1 = network.add_shuffle(input=seq_tensor)  # (b, t, d_model) to  (b, d_model, t, 1)
        trans1.first_transpose = trt.Permutation([0, 2, 1])
        trans1.reshape_dims = Dims((batch_size, d_model, max_seq_len, 1))
        trans1.name = "{}.trans1".format(name)
        out = trans1.get_output(0)  # (b, d_model, t, 1)

        conv1_w = weights["{}.conv1d_1.weight".format(name)]  # (1, d_model, duration_predictor_filter_size, duration_predictor_kernel_size, 1)
        conv1_b = weights["{}.conv1d_1.bias".format(name)]  # (duration_predictor_filter_size, )       
        conv1 = network.add_convolution(input=out, num_output_maps=duration_predictor_filter_size, kernel_shape=trt.DimsHW(duration_predictor_kernel_size, 1),
                                        kernel=Weights(conv1_w), bias=Weights(conv1_b))
        conv1.padding = trt.DimsHW(1, 0)
        conv1.name = "{}.conv1".format(name)
        out = conv1.get_output(0)  # (b, duration_predictor_filter_size, t, 1)

        trans2 = network.add_shuffle(input=out)  # (b, duration_predictor_filter_size, t, 1) to (b, t, duration_predictor_filter_size)
        trans2.first_transpose = trt.Permutation([0, 2, 1, 3])
        trans2.reshape_dims = Dims((batch_size, max_seq_len, duration_predictor_filter_size))
        trans2.name = "{}.trans2".format(name)
        out = trans2.get_output(0)  # (b, t, duration_predictor_filter_size)

        # Pytorch: out = self.relu_1(out)
        relu = network.add_activation(input=out, type=trt.ActivationType.RELU)
        relu.name = "{}.relu1".format(name)
        out_relu = relu.get_output(0)  # (b, t, duration_predictor_filter_size)        

        # Pytorch: out = self.layer_norm_1(out)
        out = self.populate_layernorm(name="{}.layer_norm_1".format(name),
                                      network=network,
                                      weights=weights,
                                      seq_tensor=out_relu,
                                      d_layer=duration_predictor_filter_size,
                                      batch_size=batch_size,
                                      max_seq_len=max_seq_len)

        # Pytorch: out = self.conv1d_2(out.transpose(1,2)).transpose(1,2)
        trans3 = network.add_shuffle(input=out)  # (b, t, duration_predictor_filter_size) to (b, duration_predictor_filter_size, t, 1)
        trans3.first_transpose = trt.Permutation([0, 2, 1])
        trans3.reshape_dims = Dims((batch_size, duration_predictor_filter_size, max_seq_len, 1))
        trans3.name = "{}.trans3".format(name)
        out = trans3.get_output(0)  # (b, duration_predictor_filter_size, t, 1)

        conv2_w = weights["{}.conv1d_2.weight".format(name)]  # (1, duration_predictor_filter_size, duration_predictor_filter_size, duration_predictor_kernel_size, 1)
        conv2_b = weights["{}.conv1d_2.bias".format(name)]  # (duration_predictor_filter_size, )       
        conv2 = network.add_convolution(input=out, num_output_maps=duration_predictor_filter_size, kernel_shape=trt.DimsHW(duration_predictor_kernel_size, 1),
                                        kernel=Weights(conv2_w), bias=Weights(conv2_b))
        conv2.padding = trt.DimsHW(1, 0)
        conv2.name = "{}.conv2".format(name)
        out = conv2.get_output(0)

        trans4 = network.add_shuffle(input=out)  # (b, duration_predictor_filter_size, t, 1) to (b, t, duration_predictor_filter_size)
        trans4.first_transpose = trt.Permutation([0, 2, 1, 3])
        trans4.reshape_dims = Dims((batch_size, max_seq_len, duration_predictor_filter_size))
        trans4.name = "{}.trans4".format(name)
        out = trans4.get_output(0)  # (b, t, duration_predictor_filter_size)

        # Pytorch: out = self.relu_2(out)
        relu = network.add_activation(input=out, type=trt.ActivationType.RELU)
        relu.name = "{}.relu2".format(name)
        out_relu = relu.get_output(0)  # (b, t, duration_predictor_filter_size)
        
        # Pytorch: out = self.layer_norm_2(out)
        out = self.populate_layernorm(name="{}.layer_norm_2".format(name),
                                      network=network,
                                      weights=weights,
                                      seq_tensor=out_relu,
                                      d_layer=duration_predictor_filter_size,
                                      batch_size=batch_size,
                                      max_seq_len=max_seq_len,
                                      )  # (b, t, duration_predictor_filter_size)

        # Pytorch: out = self.linear_layer(out)
        w = weights["{}.linear_layer.weight".format(name)]  # (1, duration_predictor_filter_size)
        out_w = network.add_constant(shape=(1, 1, duration_predictor_filter_size), weights=trt.Weights(w)).get_output(0)  # (1, 1, duration_predictor_filter_size)
        linear_w = network.add_matrix_multiply(out, MatrixOperation.NONE, out_w, MatrixOperation.TRANSPOSE)  # (b, t, duration_predictor_filter_size) * (1->b, duration_predictor_filter_size, 1) => (b, t, 1)
        linear_w.name = "{}.linear.w".format(name)
        out = linear_w.get_output(0)  # (b, t, 1)

        b = weights["{}.linear_layer.bias".format(name)]  # (1,)
        out_b = network.add_constant(shape=(1, 1, 1), weights=trt.Weights(b)).get_output(0)  # (1, 1, 1)
        linear_b = network.add_elementwise(input1=out, input2=out_b, op=trt.ElementWiseOperation.SUM)
        linear_b.name = "{}.linear.b".format(name)
        out = linear_b.get_output(0)  # (b, t, 1)

        # Pytorch: out *= input_mask.to(out.dtype)
        zeros = network.add_constant(weights=Weights(
            np.zeros(shape=(batch_size, max_seq_len, 1), dtype=np.float32)),
            shape=(batch_size, max_seq_len, 1))
        out_zeros = zeros.get_output(0)  # (b, t, 1)
        dur = network.add_select(condition=seq_mask_tensor, then_input=out, else_input=out_zeros)
        dur.name = "{}.mask".format(name)
        out_dur = dur.get_output(0)

        # Pytorch: duration = torch.clamp_min(torch.exp(duration) - 1, 0)
        exp = network.add_unary(input=out_dur, op=trt.UnaryOperation.EXP)
        exp.name = "{}.exp".format(name)
        out_exp = exp.get_output(0)
        ones = network.add_constant(weights=Weights(
            np.ones(shape=(batch_size, max_seq_len, 1), dtype=np.float32)),
            shape=(batch_size, max_seq_len, 1))
        out_ones = ones.get_output(0)  # (b, t, 1)
        sub = network.add_elementwise(input1=out_exp, input2=out_ones, op=trt.ElementWiseOperation.SUB)
        sub.name = "{}.sub_one".format(name)
        out_sub = sub.get_output(0)
        dur = network.add_elementwise(input1=out_sub, input2=out_zeros, op=trt.ElementWiseOperation.MAX)
        dur.name = "{}.max".format(name)
        out_dur = dur.get_output(0)

        # Pytorch: repeats = torch.round(repeats).long()
        half_ones = network.add_constant(weights=Weights(
            np.full((batch_size, max_seq_len, 1), 0.5, dtype=np.float32)),
            shape=(batch_size, max_seq_len, 1))
        out_half_ones = half_ones.get_output(0)  # (b, t, 1)
        add = network.add_elementwise(input1=out_dur, input2=out_half_ones, op=trt.ElementWiseOperation.SUM)
        add.name = "{}.round_add".format(name)
        out_add = add.get_output(0) # (b, t, 1)
        dur = network.add_elementwise(input1=out_add, input2=out_ones, op=trt.ElementWiseOperation.FLOOR_DIV)
        dur.name = "{}.round_floor_div".format(name)
        out_dur = dur.get_output(0) # (b, t, 1)

        dur = network.add_shuffle(input=out_dur)   # (b, t, 1) to (b, t)
        dur.reshape_dims = Dims(shape=(batch_size, max_seq_len))
        out_dur = dur.get_output(0) # (b, t)

        return out_dur

    def populate_layernorm(self, name, network, weights, seq_tensor, batch_size, max_seq_len, d_layer):
        # m
        mean = network.add_reduce(input=seq_tensor, op=trt.ReduceOperation.AVG, axes=(1 << 2), keep_dims=True)
        mean.name = "{}.mean".format(name)
        out_mean = mean.get_output(0)  # (b, t, 1)

        # m^2
        square_mean = network.add_elementwise(input1=out_mean, input2=out_mean, op=ElementWiseOperation.PROD)
        square_mean.name = "{}.square_mean".format(name)
        out_square_mean = square_mean.get_output(0)  # (b, t, 1)

        # x^2
        square = network.add_elementwise(input1=seq_tensor, input2=seq_tensor, op=ElementWiseOperation.PROD)
        square.name = "{}.square".format(name)
        out_square = square.get_output(0)  # (b, t, h)

        # e[x^2]
        mean_square = network.add_reduce(input=out_square, op=trt.ReduceOperation.AVG, axes=(1 << 2), keep_dims=True)
        mean_square.name = "{}.mean_square".format(name)
        out_mean_square = mean_square.get_output(0)  # (b, t, 1)

        # e[x^2] - m^2
        sub_square = network.add_elementwise(input1=out_mean_square, input2=out_square_mean, op=ElementWiseOperation.SUB)
        sub_square.name = "{}.sub_square".format(name)
        out_sub_square = sub_square.get_output(0)  # (b, t, 1)

        # + eps
        eps = network.add_constant(weights=Weights(np.full((batch_size, max_seq_len, 1), 1e-5, dtype=np.float32)),
                                shape=Dims((batch_size, max_seq_len, 1)))  # (b, t, 1)      
        out_eps = eps.get_output(0)
        eps.name = "{}.eps".format(name)
        std = network.add_elementwise(input1=out_sub_square, input2=out_eps, op=ElementWiseOperation.SUM)
        std.name = "{}.std".format(name)
        out_std = std.get_output(0)  # (b, t, 1)

        # std
        sqrt = network.add_unary(input=out_std, op=trt.UnaryOperation.SQRT)
        sqrt.name = "{}.sqrt".format(name)
        out_sqrt = sqrt.get_output(0)  # (b, t, 1)
                
        # y = (x - mean) / std
        sub = network.add_elementwise(input1=seq_tensor, input2=out_mean, op=ElementWiseOperation.SUB)
        sub.name = "{}.sub".format(name)
        out_sub_square = sub.get_output(0)  # (b, t, h)

        div = network.add_elementwise(input1=out_sub_square, input2=out_sqrt, op=ElementWiseOperation.DIV)
        div.name = "{}.div".format(name)
        out = div.get_output(0)  # (b, t, h)        

        # Pytorch: y = self.weight * y + self.bias
        w = weights["{}.weight".format(name)]  # (h, )
        out_w = network.add_constant(shape=(1, 1, d_layer), weights=trt.Weights(w)).get_output(0)  # (1, 1, h)
        scale_w = network.add_elementwise(input1=out, input2=out_w, op=ElementWiseOperation.PROD)  # (b, t, h) * (1->b, 1->t, h) => (b, t, h)
        scale_w.name = "{}.scale.w".format(name)
        out = scale_w.get_output(0)  # (b, t, h)

        b = weights["{}.bias".format(name)]  # (h, )
        out_b = network.add_constant(shape=(1, 1, d_layer), weights=trt.Weights(b)).get_output(0)  # (1, 1, h)
        scale_b = network.add_elementwise(input1=out, input2=out_b, op=ElementWiseOperation.SUM)  # (b, t, h) * (1->b, 1->t, h) => (b, t, h)
        scale_b.name = "{}.scale.b".format(name)
        out = scale_b.get_output(0)  # (b, t, h)

        return out


    def preprocess_weights(self, weights):
        # torch.Tensor to numpy
        weights = OrderedDict({k:v.numpy() for k,v in weights.items()})

        return weights
