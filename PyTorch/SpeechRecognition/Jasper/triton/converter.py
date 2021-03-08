# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************


import os
import json
import torch
import argparse
import importlib
from pytorch.utils import extract_io_props, load_io_props

import logging

def get_parser():

    parser = argparse.ArgumentParser()
    # required args
    parser.add_argument("--model-module", type=str, default="", required=True,
                        help="Module with model initializer and data loader")
    parser.add_argument('--convert', choices=['ts-script', 'ts-trace',
                                              'onnx', 'tensorrt'],
                        required=True, help='convert to '
                        'ts-script: TorchScript using torch.jit.script, '
                        'ts-trace: TorchScript using torch.jit.trace, '
                        'onnx: ONNX using torch.onnx.export, '
                        'tensorrt: TensorRT using OnnxParser, ')
    parser.add_argument("--max_workspace_size", type=int,
                        default=512*1024*1024,
                        help="set the size of the workspace for TensorRT \
                        conversion")
    parser.add_argument("--precision", choices=['fp16', 'fp32'],
                        default='fp32', help="convert TensorRT or \
                        TorchScript model in a given precision")
    parser.add_argument('--convert-filename', type=str, default='',
                        help='Saved model name')
    parser.add_argument('--save-dir', type=str, default='',
                        help='Saved model directory')
    parser.add_argument("--max-batch-size", type=int, default=1,
                        help="Specifies the 'max_batch_size' in the Triton \
                           model config and in TensorRT builder. See the \
                           Triton and TensorRT documentation for more info.")
    parser.add_argument('--device', type=str, default='cuda',
                        help='Select device for conversion.')

    parser.add_argument('model_arguments', nargs=argparse.REMAINDER,
                        help='arguments that will be ignored by \
                           converter lib and will be forwarded to your convert \
                           script')

    return parser


class Converter:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

        self.convert_props = {
            'ts-script': {
                'convert_func':  self.to_torchscript,
                'convert_filename': 'model.pt'
            },
            'ts-trace': {
                'convert_func'    :  self.to_torchscript,
                'convert_filename': 'model.pt'
            },
            'onnx': {
                'convert_func'    :  self.to_onnx,
                'convert_filename': 'model.onnx'
            },
            'tensorrt': {
                'convert_func'    :  self.to_tensorrt,
                'convert_filename': 'model.plan'
            }
        }

    def convert(self, convert_type, save_dir, convert_filename,
                device, precision='fp32',
                max_batch_size=1,
                # args for TensorRT:
                max_workspace_size=None):
        ''' convert the model '''
        self.convert_type = convert_type
        self.max_workspace_size = max_workspace_size
        self.max_batch_size = max_batch_size
        self.precision = precision

        # override default name if user provided name
        if convert_filename != '':
            self.convert_props[convert_type]['convert_filename'] = convert_filename

        # setup device
        torch_device = torch.device(device)

        # prepare model
        self.model.to(torch_device)
        self.model.eval()
        assert (not self.model.training), \
            "[Converter error]: could not set the model to eval() mode!"

        io_props = None
        if self.dataloader is not None:
            io_props = extract_io_props(self.model, self.dataloader,
                                        torch_device, precision, max_batch_size)

        assert self.convert_type == "ts-script" or io_props is not None, \
            "Input and output properties are empty. For conversion types \
other than \'ts-script\' input shapes are required to generate dummy input. \
Make sure that dataloader works correctly or that IO props file is provided."

        # prepare save path
        model_name = self.convert_props[convert_type]['convert_filename']
        convert_model_path = os.path.join(save_dir, model_name)

        # get convert method depending on the convert type
        convert_func = self.convert_props[convert_type]['convert_func']

        # convert the model - will be saved to disk
        if self.convert_type == "tensorrt":
            io_filepath = "triton/tensorrt_io_props_" + str(precision) + ".json"
            io_props = load_io_props(io_filepath)

        convert_func(model, torch_device, io_props, convert_model_path)

        assert (os.path.isfile(convert_model_path)), \
            f"[Converter error]: saving model to {convert_model_path} failed!"


    def generate_dummy_input(self, io_props, device):

        from pytorch.utils import triton_type_to_torch_type

        dummy_input = []
        for s,t in zip(io_props['opt_shapes'], io_props['input_types']):
            t = triton_type_to_torch_type[t]
            tensor = torch.empty(size=s, dtype=t, device=device).random_()
            dummy_input.append(tensor)
        dummy_input = tuple(dummy_input)

        return dummy_input

    def to_onnx(self, model, device, io_props, convert_model_path):
        ''' convert the model to onnx '''

        dummy_input = self.generate_dummy_input(io_props, device)

        opset_version = 11
        # convert the model to onnx
        with torch.no_grad():
            torch.onnx.export(model, dummy_input,
                              convert_model_path,
                              do_constant_folding=True,
                              input_names=io_props['input_names'],
                              output_names=io_props['output_names'],
                              dynamic_axes=io_props['dynamic_axes'],
                              opset_version=opset_version,
                              enable_onnx_checker=True)


    def to_tensorrt(self, model, device, io_props, convert_model_path):
        ''' convert the model to tensorrt '''

        assert (self.max_workspace_size), "[Converter error]: for TensorRT conversion you must provide \'max_workspace_size\'."

        import tensorrt as trt
        from pytorch.utils import build_tensorrt_engine

        # convert the model to onnx first
        self.to_onnx(model, device, io_props, convert_model_path)
        del model
        torch.cuda.empty_cache()

        zipped = zip(io_props['input_names'], io_props['min_shapes'],
                     io_props['opt_shapes'], io_props['max_shapes'])
        shapes = []
        for name,min_shape,opt_shape,max_shape in zipped:
            d = {"name":name, "min": min_shape,
                 "opt": opt_shape, "max": max_shape}
            shapes.append(d)

        tensorrt_fp16 = True if self.precision == 'fp16' else False
        # build tensorrt engine
        engine = build_tensorrt_engine(convert_model_path, shapes,
                                       self.max_workspace_size,
                                       self.max_batch_size,
                                       tensorrt_fp16)

        assert engine is not None, "[Converter error]: TensorRT build failure"

        # write tensorrt engine
        with open(convert_model_path, 'wb') as f:
            f.write(engine.serialize())


    def to_torchscript(self, model, device, io_props, convert_model_path):
        ''' convert the model to torchscript '''

        if self.convert_type == 'ts-script':

            model_ts = torch.jit.script(model)

        else: # self.convert_type == 'ts-trace'

            dummy_input = self.generate_dummy_input(io_props, device)
            with torch.no_grad():
                model_ts = torch.jit.trace(model, dummy_input)

        # save the model
        torch.jit.save(model_ts, convert_model_path)


if __name__=='__main__':

    parser = get_parser()
    args = parser.parse_args()
    model_args_list = args.model_arguments[1:]

    logging.basicConfig(level=logging.INFO)

    mm = importlib.import_module(args.model_module)

    model = mm.init_model(model_args_list, args.precision, args.device)

    dataloader = mm.get_dataloader(model_args_list)
    converter = Converter(model, dataloader)

    converter.convert(args.convert, args.save_dir, args.convert_filename,
                      args.device, args.precision,
                      args.max_batch_size,
                      args.max_workspace_size)
