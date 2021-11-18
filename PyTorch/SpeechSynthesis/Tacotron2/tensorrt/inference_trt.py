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

import tensorrt as trt
import numpy as np
from scipy.io.wavfile import write
import time
import torch
import argparse

import sys
sys.path.append('./')

from tacotron2_common.utils import to_gpu, get_mask_from_lengths
from tacotron2.text import text_to_sequence
from inference import MeasureTime, prepare_input_sequence, load_and_setup_model
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
from trt_utils import load_engine, run_trt_engine

from waveglow.denoiser import Denoiser

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='full path to the input text (phareses separated by new line)')
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--encoder', type=str, required=True,
                        help='full path to the Encoder engine')
    parser.add_argument('--decoder', type=str, required=True,
                        help='full path to the DecoderIter engine')
    parser.add_argument('--postnet', type=str, required=True,
                        help='full path to the Postnet engine')
    parser.add_argument('--waveglow', type=str, required=True,
                        help='full path to the WaveGlow engine')
    parser.add_argument('--waveglow-ckpt', type=str, default="",
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--fp16', action='store_true',
                        help='inference with FP16')

    return parser


def init_decoder_inputs(memory, processed_memory, memory_lengths):

    device = memory.device
    dtype = memory.dtype
    bs = memory.size(0)
    seq_len = memory.size(1)
    attention_rnn_dim = 1024
    decoder_rnn_dim = 1024
    encoder_embedding_dim = 512
    n_mel_channels = 80

    attention_hidden = torch.zeros(bs, attention_rnn_dim, device=device, dtype=dtype)
    attention_cell = torch.zeros(bs, attention_rnn_dim, device=device, dtype=dtype)
    decoder_hidden = torch.zeros(bs, decoder_rnn_dim, device=device, dtype=dtype)
    decoder_cell = torch.zeros(bs, decoder_rnn_dim, device=device, dtype=dtype)
    attention_weights = torch.zeros(bs, seq_len, device=device, dtype=dtype)
    attention_weights_cum = torch.zeros(bs, seq_len, device=device, dtype=dtype)
    attention_context = torch.zeros(bs, encoder_embedding_dim, device=device, dtype=dtype)
    mask = get_mask_from_lengths(memory_lengths).to(device)
    decoder_input = torch.zeros(bs, n_mel_channels, device=device, dtype=dtype)

    return (decoder_input, attention_hidden, attention_cell, decoder_hidden,
            decoder_cell, attention_weights, attention_weights_cum,
            attention_context, memory, processed_memory, mask)

def init_decoder_outputs(memory, memory_lengths):

    device = memory.device
    dtype = memory.dtype
    bs = memory.size(0)
    seq_len = memory.size(1)
    attention_rnn_dim = 1024
    decoder_rnn_dim = 1024
    encoder_embedding_dim = 512
    n_mel_channels = 80

    attention_hidden = torch.zeros(bs, attention_rnn_dim, device=device, dtype=dtype)
    attention_cell = torch.zeros(bs, attention_rnn_dim, device=device, dtype=dtype)
    decoder_hidden = torch.zeros(bs, decoder_rnn_dim, device=device, dtype=dtype)
    decoder_cell = torch.zeros(bs, decoder_rnn_dim, device=device, dtype=dtype)
    attention_weights = torch.zeros(bs, seq_len, device=device, dtype=dtype)
    attention_weights_cum = torch.zeros(bs, seq_len, device=device, dtype=dtype)
    attention_context = torch.zeros(bs, encoder_embedding_dim, device=device, dtype=dtype)
    decoder_output = torch.zeros(bs, n_mel_channels, device=device, dtype=dtype)
    gate_prediction = torch.zeros(bs, 1, device=device, dtype=dtype)

    return (attention_hidden, attention_cell, decoder_hidden,
            decoder_cell, attention_weights, attention_weights_cum,
            attention_context, decoder_output, gate_prediction)

def init_decoder_tensors(decoder_inputs, decoder_outputs):

    decoder_tensors = {
        "inputs" : {
            'decoder_input': decoder_inputs[0],
            'attention_hidden': decoder_inputs[1],
            'attention_cell': decoder_inputs[2],
            'decoder_hidden': decoder_inputs[3],
            'decoder_cell': decoder_inputs[4],
            'attention_weights': decoder_inputs[5],
            'attention_weights_cum': decoder_inputs[6],
            'attention_context': decoder_inputs[7],
            'memory': decoder_inputs[8],
            'processed_memory': decoder_inputs[9],
            'mask': decoder_inputs[10]
        },
        "outputs" : {
            'out_attention_hidden': decoder_outputs[0],
            'out_attention_cell': decoder_outputs[1],
            'out_decoder_hidden': decoder_outputs[2],
            'out_decoder_cell': decoder_outputs[3],
            'out_attention_weights': decoder_outputs[4],
            'out_attention_weights_cum': decoder_outputs[5],
            'out_attention_context': decoder_outputs[6],
            'decoder_output': decoder_outputs[7],
            'gate_prediction': decoder_outputs[8]
        }
    }
    return decoder_tensors

def swap_inputs_outputs(decoder_inputs, decoder_outputs):

    new_decoder_inputs = (decoder_outputs[7], # decoder_output
                          decoder_outputs[0], # attention_hidden
                          decoder_outputs[1], # attention_cell
                          decoder_outputs[2], # decoder_hidden
                          decoder_outputs[3], # decoder_cell
                          decoder_outputs[4], # attention_weights
                          decoder_outputs[5], # attention_weights_cum
                          decoder_outputs[6], # attention_context
                          decoder_inputs[8],  # memory
                          decoder_inputs[9],  # processed_memory
                          decoder_inputs[10]) # mask

    new_decoder_outputs = (decoder_inputs[1], # attention_hidden
                           decoder_inputs[2], # attention_cell
                           decoder_inputs[3], # decoder_hidden
                           decoder_inputs[4], # decoder_cell
                           decoder_inputs[5], # attention_weights
                           decoder_inputs[6], # attention_weights_cum
                           decoder_inputs[7], # attention_context
                           decoder_inputs[0], # decoder_input
                           decoder_outputs[8])# gate_output

    return new_decoder_inputs, new_decoder_outputs


def infer_tacotron2_trt(encoder, decoder_iter, postnet,
                        encoder_context, decoder_context, postnet_context,
                        sequences, sequence_lengths, measurements, fp16):

    memory = torch.zeros((len(sequence_lengths), sequence_lengths[0], 512)).cuda()
    if fp16:
        memory = memory.half()
    device = memory.device
    dtype = memory.dtype

    processed_memory = torch.zeros((len(sequence_lengths),sequence_lengths[0],128), device=device, dtype=dtype)
    lens = torch.zeros_like(sequence_lengths)

    encoder_tensors = {
        "inputs" :
        {'sequences': sequences, 'sequence_lengths': sequence_lengths},
        "outputs" :
        {'memory': memory, 'lens': lens, 'processed_memory': processed_memory}
    }

    print("Running Tacotron2 Encoder")
    with MeasureTime(measurements, "tacotron2_encoder_time"):
        run_trt_engine(encoder_context, encoder, encoder_tensors)

    device = memory.device
    mel_lengths = torch.zeros([memory.size(0)], dtype=torch.int32, device = device)
    not_finished = torch.ones([memory.size(0)], dtype=torch.int32, device = device)
    mel_outputs, gate_outputs, alignments = (torch.zeros(1, device = device), torch.zeros(1, device = device), torch.zeros(1, device = device))
    gate_threshold = 0.5
    max_decoder_steps = 1664
    first_iter = True

    decoder_inputs = init_decoder_inputs(memory, processed_memory, sequence_lengths)
    decoder_outputs = init_decoder_outputs(memory, sequence_lengths)

    print("Running Tacotron2 Decoder")
    measurements_decoder = {}
    while True:
        decoder_tensors = init_decoder_tensors(decoder_inputs, decoder_outputs)
        with MeasureTime(measurements_decoder, "step"):
            run_trt_engine(decoder_context, decoder_iter, decoder_tensors)

        if first_iter:
            mel_outputs = torch.unsqueeze(decoder_outputs[7], 2)
            gate_outputs = torch.unsqueeze(decoder_outputs[8], 2)
            alignments = torch.unsqueeze(decoder_outputs[4], 2)
            measurements['tacotron2_decoder_time'] = measurements_decoder['step']
            first_iter = False
        else:
            mel_outputs = torch.cat((mel_outputs, torch.unsqueeze(decoder_outputs[7], 2)), 2)
            gate_outputs = torch.cat((gate_outputs, torch.unsqueeze(decoder_outputs[8], 2)), 2)
            alignments = torch.cat((alignments, torch.unsqueeze(decoder_outputs[4], 2)), 2)
            measurements['tacotron2_decoder_time'] += measurements_decoder['step']

        dec = torch.le(torch.sigmoid(decoder_outputs[8]), gate_threshold).to(torch.int32).squeeze(1)
        not_finished = not_finished*dec
        mel_lengths += not_finished

        if torch.sum(not_finished) == 0:
            print("Stopping after",mel_outputs.size(2),"decoder steps")
            break
        if mel_outputs.size(2) == max_decoder_steps:
            print("Warning! Reached max decoder steps")
            break

        decoder_inputs, decoder_outputs = swap_inputs_outputs(decoder_inputs, decoder_outputs)

    mel_outputs_postnet = torch.zeros_like(mel_outputs, device=device, dtype=dtype)

    postnet_tensors = {
        "inputs" :
        {'mel_outputs': mel_outputs},
        "outputs" :
        {'mel_outputs_postnet': mel_outputs_postnet}
    }
    print("Running Tacotron2 Postnet")
    with MeasureTime(measurements, "tacotron2_postnet_time"):
        run_trt_engine(postnet_context, postnet, postnet_tensors)

    print("Tacotron2 Postnet done")

    return mel_outputs_postnet, mel_lengths


def infer_waveglow_trt(waveglow, waveglow_context, mel, measurements, fp16):

    mel_size = mel.size(2)
    batch_size = mel.size(0)
    stride = 256
    n_group = 8
    z_size = mel_size*stride
    z_size = z_size//n_group
    z = torch.randn(batch_size, n_group, z_size).cuda()
    audios = torch.zeros(batch_size, mel_size*stride).cuda()

    if fp16:
        z = z.half()
        mel = mel.half()
        audios = audios.half()

    waveglow_tensors = {
        "inputs" :
        {'input__0': mel, 'input__1': z},
        "outputs" :
        {'output__0': audios}
    }
    print("Running WaveGlow")
    with MeasureTime(measurements, "waveglow_time"):
        run_trt_engine(waveglow_context, waveglow, waveglow_tensors)

    return audios

def main():

    parser = argparse.ArgumentParser(
        description='TensorRT Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    # initialize CUDA state
    torch.cuda.init()

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    encoder = load_engine(args.encoder, TRT_LOGGER)
    decoder_iter = load_engine(args.decoder, TRT_LOGGER)
    postnet = load_engine(args.postnet, TRT_LOGGER)
    waveglow = load_engine(args.waveglow, TRT_LOGGER)

    if args.waveglow_ckpt != "":
        # setup denoiser using WaveGlow PyTorch checkpoint
        waveglow_ckpt = load_and_setup_model('WaveGlow', parser, args.waveglow_ckpt,
                                             True, forward_is_infer=True)
        denoiser = Denoiser(waveglow_ckpt).cuda()
        # after initialization, we don't need WaveGlow PyTorch checkpoint
        # anymore - deleting
        del waveglow_ckpt
        torch.cuda.empty_cache()

    # create TRT contexts for each engine
    encoder_context = encoder.create_execution_context()
    decoder_context = decoder_iter.create_execution_context()
    postnet_context = postnet.create_execution_context()
    waveglow_context = waveglow.create_execution_context()

    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT,
                                              args.output+'/'+args.log_file),
                            StdOutBackend(Verbosity.VERBOSE)])

    texts = []
    try:
        f = open(args.input, 'r')
        texts = f.readlines()
    except:
        print("Could not read file")
        sys.exit(1)

    measurements = {}

    sequences, sequence_lengths = prepare_input_sequence(texts)
    sequences = sequences.to(torch.int32)
    sequence_lengths = sequence_lengths.to(torch.int32)

    with MeasureTime(measurements, "latency"):
        mel, mel_lengths = infer_tacotron2_trt(encoder, decoder_iter, postnet,
                                               encoder_context, decoder_context, postnet_context,
                                               sequences, sequence_lengths, measurements, args.fp16)
        audios = infer_waveglow_trt(waveglow, waveglow_context, mel, measurements, args.fp16)

    with encoder_context, decoder_context,  postnet_context, waveglow_context:
        pass

    audios = audios.float()
    if args.waveglow_ckpt != "":
        with MeasureTime(measurements, "denoiser"):
            audios = denoiser(audios, strength=args.denoising_strength).squeeze(1)

    for i, audio in enumerate(audios):
        audio = audio[:mel_lengths[i]*args.stft_hop_length]
        audio = audio/torch.max(torch.abs(audio))
        audio_path = args.output + "audio_"+str(i)+"_trt.wav"
        write(audio_path, args.sampling_rate, audio.cpu().numpy())


    DLLogger.log(step=0, data={"tacotron2_encoder_latency": measurements['tacotron2_encoder_time']})
    DLLogger.log(step=0, data={"tacotron2_decoder_latency": measurements['tacotron2_decoder_time']})
    DLLogger.log(step=0, data={"tacotron2_postnet_latency": measurements['tacotron2_postnet_time']})
    DLLogger.log(step=0, data={"waveglow_latency": measurements['waveglow_time']})
    DLLogger.log(step=0, data={"latency": measurements['latency']})

    if args.waveglow_ckpt != "":
        DLLogger.log(step=0, data={"denoiser": measurements['denoiser']})
    DLLogger.flush()

    prec = "fp16" if args.fp16 else "fp32"
    latency = measurements['latency']
    throughput = audios.size(1)/latency
    log_data = "1,"+str(sequence_lengths[0].item())+","+prec+","+str(latency)+","+str(throughput)+","+str(mel_lengths[0].item())+"\n"
    with open("log_bs1_"+prec+".log", 'a') as f:
        f.write(log_data)

if __name__ == "__main__":
    main()
