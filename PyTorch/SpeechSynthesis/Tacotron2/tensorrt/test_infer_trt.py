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

import sys
sys.path.append('./')
from tacotron2.text import text_to_sequence
import models
import torch
import argparse
import numpy as np
from scipy.io.wavfile import write

from inference import checkpoint_from_distributed, unwrap_distributed, MeasureTime, prepare_input_sequence, load_and_setup_model
from inference_trt import infer_tacotron2_trt, infer_waveglow_trt

from trt_utils import load_engine
import tensorrt as trt

import time
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from apex import amp

def parse_args(parser):
    """
    Parse commandline arguments.
    """
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
    parser.add_argument('-s', '--sigma-infer', default=0.6, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--fp16', action='store_true',
                        help='inference with FP16')
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--num-iters', type=int, default=10,
                        help='Number of iterations')
    parser.add_argument('-il', '--input-length', type=int, default=64,
                        help='Input length')
    parser.add_argument('-bs', '--batch-size', type=int, default=1,
                        help='Batch size')

    return parser


def print_stats(measurements_all):

    print(np.mean(measurements_all['latency'][1:]),
          np.mean(measurements_all['throughput'][1:]),
          np.mean(measurements_all['pre_processing'][1:]),
          np.mean(measurements_all['type_conversion'][1:])+
          np.mean(measurements_all['storage'][1:])+
          np.mean(measurements_all['data_transfer'][1:]),
          np.mean(measurements_all['num_mels_per_audio'][1:]))

    throughput = measurements_all['throughput']
    preprocessing = measurements_all['pre_processing']
    type_conversion = measurements_all['type_conversion']
    storage = measurements_all['storage']
    data_transfer = measurements_all['data_transfer']
    postprocessing = [sum(p) for p in zip(type_conversion,storage,data_transfer)]
    latency = measurements_all['latency']
    num_mels_per_audio = measurements_all['num_mels_per_audio']

    latency.sort()

    cf_50 = max(latency[:int(len(latency)*0.50)])
    cf_90 = max(latency[:int(len(latency)*0.90)])
    cf_95 = max(latency[:int(len(latency)*0.95)])
    cf_99 = max(latency[:int(len(latency)*0.99)])
    cf_100 = max(latency[:int(len(latency)*1.0)])

    print("Throughput average (samples/sec) = {:.4f}".format(np.mean(throughput)))
    print("Preprocessing average (seconds) = {:.4f}".format(np.mean(preprocessing)))
    print("Postprocessing average (seconds) = {:.4f}".format(np.mean(postprocessing)))
    print("Number of mels per audio average = {}".format(np.mean(num_mels_per_audio))) #
    print("Latency average (seconds) = {:.4f}".format(np.mean(latency)))
    print("Latency std (seconds) = {:.4f}".format(np.std(latency)))
    print("Latency cl 50 (seconds) = {:.4f}".format(cf_50))
    print("Latency cl 90 (seconds) = {:.4f}".format(cf_90))
    print("Latency cl 95 (seconds) = {:.4f}".format(cf_95))
    print("Latency cl 99 (seconds) = {:.4f}".format(cf_99))
    print("Latency cl 100 (seconds) = {:.4f}".format(cf_100))


def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, unknown_args = parser.parse_known_args()

    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, args.log_file),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})
    DLLogger.log(step="PARAMETER", data={'model_name':'Tacotron2_PyT'})

    measurements_all = {"pre_processing": [],
                        "tacotron2_encoder_time": [],
                        "tacotron2_decoder_time": [],
                        "tacotron2_postnet_time": [],
                        "tacotron2_latency": [],
                        "waveglow_latency": [],
                        "latency": [],
                        "type_conversion": [],
                        "data_transfer": [],
                        "storage": [],
                        "tacotron2_items_per_sec": [],
                        "waveglow_items_per_sec": [],
                        "num_mels_per_audio": [],
                        "throughput": []}

    print("args:", args, unknown_args)

    torch.cuda.init()

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    encoder = load_engine(args.encoder, TRT_LOGGER)
    decoder_iter = load_engine(args.decoder, TRT_LOGGER)
    postnet = load_engine(args.postnet, TRT_LOGGER)
    waveglow = load_engine(args.waveglow, TRT_LOGGER)

    if args.waveglow_ckpt != "":
        # setup denoiser using WaveGlow PyTorch checkpoint
        waveglow_ckpt = load_and_setup_model('WaveGlow', parser,
                                             args.waveglow_ckpt,
                                             fp16_run=args.fp16,
                                             cpu_run=False,
                                             forward_is_infer=True)
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


    texts = ["The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves. The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves."]
    texts = [texts[0][:args.input_length]]
    texts = texts*args.batch_size

    warmup_iters = 3

    for iter in range(args.num_iters):

        measurements = {}

        with MeasureTime(measurements, "pre_processing"):
            sequences_padded, input_lengths = prepare_input_sequence(texts)
            sequences_padded = sequences_padded.to(torch.int32)
            input_lengths = input_lengths.to(torch.int32)

        with torch.no_grad():
            with MeasureTime(measurements, "latency"):
                with MeasureTime(measurements, "tacotron2_latency"):
                    mel, mel_lengths = infer_tacotron2_trt(encoder, decoder_iter, postnet,
                                                           encoder_context, decoder_context, postnet_context,
                                                           sequences_padded, input_lengths, measurements, args.fp16)

                with MeasureTime(measurements, "waveglow_latency"):
                    audios = infer_waveglow_trt(waveglow, waveglow_context, mel, measurements, args.fp16)

        num_mels = mel.size(0)*mel.size(2)
        num_samples = audios.size(0)*audios.size(1)

        with MeasureTime(measurements, "type_conversion"):
            audios = audios.float()

        with MeasureTime(measurements, "data_transfer"):
            audios = audios.cpu()

        with MeasureTime(measurements, "storage"):
            audios = audios.numpy()
            for i, audio in enumerate(audios):
                audio_path = "audio_"+str(i)+".wav"
                write(audio_path, args.sampling_rate,
                      audio[:mel_lengths[i]*args.stft_hop_length])

        measurements['tacotron2_items_per_sec'] = num_mels/measurements['tacotron2_latency']
        measurements['waveglow_items_per_sec'] = num_samples/measurements['waveglow_latency']
        measurements['num_mels_per_audio'] = mel.size(2)
        measurements['throughput'] = num_samples/measurements['latency']

        if iter >= warmup_iters:
            for k,v in measurements.items():
                if k in measurements_all.keys():
                    measurements_all[k].append(v)
                    DLLogger.log(step=(iter-warmup_iters), data={k: v})

    DLLogger.flush()

    print_stats(measurements_all)

if __name__ == '__main__':
    main()
