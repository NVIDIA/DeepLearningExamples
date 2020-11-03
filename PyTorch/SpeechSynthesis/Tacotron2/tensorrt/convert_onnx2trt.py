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

import pycuda.driver as cuda
import pycuda.autoinit
import onnx
import argparse
import tensorrt as trt
import os

import sys
sys.path.append('./')

from trt_utils import build_engine

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--encoder', type=str, default="",
                        help='full path to the Encoder ONNX')
    parser.add_argument('--decoder', type=str, default="",
                        help='full path to the DecoderIter ONNX')
    parser.add_argument('--postnet', type=str, default="",
                        help='full path to the Postnet ONNX')
    parser.add_argument('--waveglow', type=str, default="",
                        help='full path to the WaveGlow ONNX')
    parser.add_argument('--fp16', action='store_true',
                        help='inference with FP16')

    return parser


def main():

    parser = argparse.ArgumentParser(
        description='Export from ONNX to TensorRT for Tacotron 2 and WaveGlow')
    parser = parse_args(parser)
    args = parser.parse_args()

    engine_prec = "_fp16" if args.fp16 else "_fp32"

    # Encoder
    shapes=[{"name": "sequences",        "min": (1,4), "opt": (1,128), "max": (1,256)},
            {"name": "sequence_lengths", "min": (1,),  "opt": (1,),    "max": (1,)}]
    if args.encoder != "":
        print("Building Encoder ...")
        encoder_engine = build_engine(args.encoder, shapes=shapes, fp16=args.fp16)
        if encoder_engine is not None:
            with open(args.output+"/"+"encoder"+engine_prec+".engine", 'wb') as f:
                f.write(encoder_engine.serialize())
        else:
            print("Failed to build engine from", args.encoder)
            sys.exit()

    # DecoderIter
    shapes=[{"name": "decoder_input",         "min": (1,80),    "opt": (1,80),      "max": (1,80)},
            {"name": "attention_hidden",      "min": (1,1024),  "opt": (1,1024),    "max": (1,1024)},
            {"name": "attention_cell",        "min": (1,1024),  "opt": (1,1024),    "max": (1,1024)},
            {"name": "decoder_hidden",        "min": (1,1024),  "opt": (1,1024),    "max": (1,1024)},
            {"name": "decoder_cell",          "min": (1,1024),  "opt": (1,1024),    "max": (1,1024)},
            {"name": "attention_weights",     "min": (1,4),     "opt": (1,128),     "max": (1,256)},
            {"name": "attention_weights_cum", "min": (1,4),     "opt": (1,128),     "max": (1,256)},
            {"name": "attention_context",     "min": (1,512),   "opt": (1,512),     "max": (1,512)},
            {"name": "memory",                "min": (1,4,512), "opt": (1,128,512), "max": (1,256,512)},
            {"name": "processed_memory",      "min": (1,4,128), "opt": (1,128,128), "max": (1,256,128)},
            {"name": "mask",                  "min": (1,4),     "opt": (1,128),     "max": (1,256)}]
    if args.decoder != "":
        print("Building Decoder ...")
        decoder_iter_engine = build_engine(args.decoder, shapes=shapes, fp16=args.fp16)
        if decoder_iter_engine is not None:
            with open(args.output+"/"+"decoder_iter"+engine_prec+".engine", 'wb') as f:
                f.write(decoder_iter_engine.serialize())
        else:
            print("Failed to build engine from", args.decoder)
            sys.exit()

    # Postnet
    shapes=[{"name": "mel_outputs", "min": (1,80,32), "opt": (1,80,768), "max": (1,80,1664)}]
    if args.postnet != "":
        print("Building Postnet ...")
        postnet_engine = build_engine(args.postnet, shapes=shapes, fp16=args.fp16)
        if postnet_engine is not None:
            with open(args.output+"/"+"postnet"+engine_prec+".engine", 'wb') as f:
                f.write(postnet_engine.serialize())
        else:
            print("Failed to build engine from", args.postnet)
            sys.exit()

    # WaveGlow
    shapes=[{"name": "mel", "min": (1,80,32),  "opt": (1,80,768),  "max": (1,80,1664)},
            {"name": "z",   "min": (1,8,1024), "opt": (1,8,24576), "max": (1,8,53248)}]
    if args.waveglow != "":
        print("Building WaveGlow ...")
        waveglow_engine = build_engine(args.waveglow, shapes=shapes, fp16=args.fp16)
        if waveglow_engine is not None:
            engine_path = os.path.join(args.output, "waveglow"+engine_prec+".engine")
            with open(engine_path, 'wb') as f:
                f.write(waveglow_engine.serialize())
        else:
            print("Failed to build engine from", args.waveglow)
            sys.exit()


if __name__ == '__main__':
    main()
