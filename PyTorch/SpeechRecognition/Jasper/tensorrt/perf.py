#!/usr/bin/env python3
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Constructs TensorRT engine for JASPER and evaluates inference latency'''
import argparse
import sys, os
# Get local modules in parent directory and current directory (assuming this was called from root of repository)
sys.path.append("./")
sys.path.append("./trt")
import perfutils
import trtutils
import perfprocedures
from model import GreedyCTCDecoder
from helpers import __ctc_decoder_predictions_tensor
import caffe2.python.onnx.backend as c2backend
import onnxruntime as ort

import torch
from torch import nn
from torch.nn import functional as F


def main(args):        
    print ("Getting component")
    # Get shared utility across PyTorch and TRT
    pyt_components, saved_onnx = perfutils.get_pytorch_components_and_onnx(args)

    print ("Getting engine")
    # Get a TRT engine. See function for argument parsing logic
    engine = trtutils.get_engine(args)
    print ("Got engine.")

    if args.wav:
        with torch.no_grad():
            audio_processor = pyt_components['audio_preprocessor']
            audio_processor.eval()
            greedy_decoder = GreedyCTCDecoder()
            input_wav, num_audio_samples = pyt_components['input_wav']
            features = audio_processor(input_wav, num_audio_samples)
            features = perfutils.adjust_shape(features, args)
            if not args.engine_path:
                outputs = engine.run(None, {'FEATURES': features[0].data.cpu().numpy()})
                inference = 1.0
                t_log_probs_e = outputs[0]
                t_log_probs_e=perfutils.torchify_trt_out(t_log_probs_e, t_log_probs_e.shape)
            else:
                with engine.create_execution_context() as context:
                    t_log_probs_e, copyto, inference, copyfrom= perfprocedures.do_inference(context, [features[0]])
            t_predictions_e = greedy_decoder(t_log_probs_e)
            hypotheses = __ctc_decoder_predictions_tensor(t_predictions_e, labels=perfutils.get_vocab())
            print("INTERENCE TIME: {} ms".format(inference*1000.0))
            print("TRANSCRIPT: ", hypotheses)
            return

    wer, preds, times = perfprocedures.compare_times_trt_pyt_exhaustive(engine,
                                                                        pyt_components,
                                                                        args)

    string_header, string_data = perfutils.do_csv_export(wer, times, args.batch_size, args.seq_len)

    if args.csv_path is not None:
        print ("Exporting to " + args.csv_path)
        with open(args.csv_path, 'a+') as f:
            # See if header is there, if so, check that it matches
            f.seek(0) # Read from start of file
            existing_header = f.readline()
            if existing_header == "":
                f.write(string_header)
                f.write("\n")
            elif existing_header[:-1] != string_header:
                raise Exception(f"Writing to existing CSV with incorrect format\nProduced:\n{string_header}\nFound:\n{existing_header}\nIf you intended to write to a new results csv, please change the csv_path argument")
            f.seek(0,2) # Write to end of file
            f.write(string_data)
            f.write("\n")
    else:
        print(string_header)
        print(string_data)

    if args.trt_prediction_path is not None:
        with open(args.trt_prediction_path, 'w') as fp:
            fp.write('\n'.join(preds['trt']))
     
    if args.pyt_prediction_path is not None:
        with open(args.pyt_prediction_path, 'w') as fp:
            fp.write('\n'.join(preds['pyt']))   


def parse_args():
    parser = argparse.ArgumentParser(description="Performance test of TRT")
    parser.add_argument("--engine_path", default=None, type=str, help="Path to serialized TRT engine")
    parser.add_argument("--use_existing_engine", action="store_true", default=False, help="If set, will deserialize engine at --engine_path" )
    parser.add_argument("--engine_batch_size", default=16, type=int, help="Maximum batch size for constructed engine; needed when building")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for data when running inference.")
    parser.add_argument("--dataset_dir", type=str, help="Root directory of dataset")
    parser.add_argument("--model_toml", type=str, required=True, help="Config toml to use. A selection can be found in configs/")
    parser.add_argument("--val_manifest", type=str, help="JSON manifest of dataset.")
    parser.add_argument("--onnx_path", default=None, type=str, help="Path to onnx model for engine creation")
    parser.add_argument("--seq_len", default=None, type=int, help="Generate an ONNX export with this fixed sequence length, and save to --onnx_path. Requires also using --onnx_path and --ckpt_path.")
    parser.add_argument("--max_seq_len", default=3600, type=int, help="Max sequence length for TRT engine build. Default works with TRTIS benchmark. Set it larger than seq_len")
    parser.add_argument("--ckpt_path", default=None, type=str, help="If provided, will also construct pytorch acoustic model")
    parser.add_argument("--max_duration", default=None, type=float, help="Maximum possible length of audio data in seconds")
    parser.add_argument("--num_steps", default=-1, type=int, help="Number of inference steps to run")
    parser.add_argument("--trt_fp16", action="store_true", default=False, help="If set, will allow TRT engine builder to select fp16 kernels as well as fp32")
    parser.add_argument("--pyt_fp16", action="store_true", default=False, help="If set, will construct pytorch model with fp16 weights")
    parser.add_argument("--make_onnx", action="store_true", default=False, help="If set, will create an ONNX model and store it at the path specified by --onnx_path")
    parser.add_argument("--csv_path", type=str, default=None, help="File to append csv info about inference time")
    parser.add_argument("--trt_prediction_path", type=str, default=None, help="File to write predictions inferred with trt")
    parser.add_argument("--pyt_prediction_path", type=str, default=None, help="File to write predictions inferred with pytorch")
    parser.add_argument("--verbose", action="store_true", default=False, help="If set, will verbosely describe TRT engine building and deserialization as well as TRT inference")
    parser.add_argument("--wav", type=str, help='absolute path to .wav file (16KHz)')
    parser.add_argument("--max_workspace_size", default=0, type=int, help="Maximum GPU memory workspace size for constructed engine; needed when building")
    parser.add_argument("--transpose", action="store_true", default=False, help="If set, will transpose input")
    parser.add_argument("--static_shape", action="store_true", default=False, help="If set, use static shape otherwise dynamic shape. Dynamic shape is always preferred.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main(args)
