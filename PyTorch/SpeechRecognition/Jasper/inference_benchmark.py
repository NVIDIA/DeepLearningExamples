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

import argparse
import itertools
import os
import sys
import time
import random
import numpy as np
from heapq import nlargest
import math
from tqdm import tqdm
import toml
import torch
from apex import amp
from dataset import AudioToTextDataLayer
from helpers import process_evaluation_batch, process_evaluation_epoch, add_ctc_labels, AmpOptimizations, print_dict
from model import AudioPreprocessing, GreedyCTCDecoder, JasperEncoderDecoder

def parse_args():
    parser = argparse.ArgumentParser(description='Jasper')
    parser.add_argument("--steps", default=None, help='if not specified do evaluation on full dataset. otherwise only evaluates the specified number of iterations for each worker', type=int)
    parser.add_argument("--batch_size", default=16, type=int, help='data batch size')
    parser.add_argument("--max_duration", default=None, type=float, help='maximum duration of sequences. if None uses attribute from model configuration file')
    parser.add_argument("--pad_to", default=None, type=int, help="default is pad to value as specified in model configurations. if -1 pad to maximum duration. If > 0 pad batch to next multiple of value")
    parser.add_argument("--model_toml", type=str, help='relative model configuration path given dataset folder')
    parser.add_argument("--dataset_dir", type=str, help='absolute path to dataset folder')
    parser.add_argument("--val_manifest", type=str, help='relative path to evaluation dataset manifest file')
    parser.add_argument("--cudnn_benchmark", action='store_true', help="enable cudnn benchmark")
    parser.add_argument("--ckpt", default=None, type=str, required=True, help='path to model checkpoint')
    parser.add_argument("--fp16", action='store_true', help='use half precision')
    parser.add_argument("--seed", default=42, type=int, help='seed')
    return parser.parse_args()

def eval(
        data_layer,
        audio_processor,
        encoderdecoder,
        greedy_decoder,
        labels,
        args):
    """performs evaluation and prints performance statistics
    Args:
        data_layer: data layer object that holds data loader
        audio_processor: data processing module
        encoderdecoder: acoustic model
        greedy_decoder: greedy decoder
        labels: list of labels as output vocabulary
        args: script input arguments
    """
    batch_size=args.batch_size
    steps=args.steps
    audio_processor.eval()
    encoderdecoder.eval()
    with torch.no_grad():
        _global_var_dict = {
            'predictions': [],
            'transcripts': [],
        }

        it = 0
        ep = 0

        if steps is None:
            steps = math.ceil(len(data_layer) / batch_size)
        durations_dnn = []
        durations_dnn_and_prep = []
        seq_lens = []
        while True:
            ep += 1
            for data in tqdm(data_layer.data_iterator):
                it += 1
                if it > steps:
                    break
                tensors = []
                dl_device = torch.device("cuda")
                for d in data:
                    tensors.append(d.to(dl_device))
     
                t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = tensors
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                t_processed_signal = audio_processor(t_audio_signal_e, t_a_sig_length_e)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                t_log_probs_e, _  = encoderdecoder.infer(t_processed_signal)

                torch.cuda.synchronize()
                stop_time = time.perf_counter()

                time_prep_and_dnn = stop_time - t0
                time_dnn = stop_time - t1
                t_predictions_e = greedy_decoder(log_probs=t_log_probs_e)

                values_dict = dict(
                    predictions=[t_predictions_e],
                    transcript=[t_transcript_e],
                    transcript_length=[t_transcript_len_e],
                )
                process_evaluation_batch(values_dict, _global_var_dict, labels=labels)
                durations_dnn.append(time_dnn)
                durations_dnn_and_prep.append(time_prep_and_dnn)
                seq_lens.append(t_processed_signal[0].shape[-1])

            if it >= steps:

                wer, _ = process_evaluation_epoch(_global_var_dict)
                print("==========>>>>>>Evaluation of all iterations WER: {0}\n".format(wer))
                break

        ratios = [0.9,  0.95,0.99, 1.]
        latencies_dnn = take_durations_and_output_percentile(durations_dnn, ratios)
        latencies_dnn_and_prep = take_durations_and_output_percentile(durations_dnn_and_prep, ratios)
        print("\n using batch size {} and {} frames ".format(batch_size, seq_lens[-1]))
        print("\n".join(["dnn latency {} : {} ".format(k, v) for k, v in latencies_dnn.items()]))
        print("\n".join(["prep + dnn latency {} : {} ".format(k, v) for k, v in latencies_dnn_and_prep.items()]))

def take_durations_and_output_percentile(durations, ratios):
    durations = np.asarray(durations) * 1000 # in ms
    latency = durations 

    latency = latency[5:]
    mean_latency = np.mean(latency)

    latency_worst = nlargest(math.ceil( (1 - min(ratios))* len(latency)), latency)
    latency_ranges=get_percentile(ratios, latency_worst, len(latency))
    latency_ranges["0.5"] = mean_latency
    return latency_ranges

def get_percentile(ratios, arr, nsamples):
    res = {}
    for a in ratios:
        idx = max(int(nsamples * (1 - a)), 0)
        res[a] = arr[idx]
    return res

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    assert(args.steps is None or args.steps > 5)
    print("CUDNN BENCHMARK ", args.cudnn_benchmark)
    assert(torch.cuda.is_available())

    if args.fp16:
        optim_level = 3
    else:
        optim_level = 0
    batch_size = args.batch_size

    jasper_model_definition = toml.load(args.model_toml)
    dataset_vocab = jasper_model_definition['labels']['labels']
    ctc_vocab = add_ctc_labels(dataset_vocab)

    val_manifest = args.val_manifest
    featurizer_config = jasper_model_definition['input_eval']
    featurizer_config["optimization_level"] = optim_level

    if args.max_duration is not None:
        featurizer_config['max_duration'] = args.max_duration
    
    # TORCHSCRIPT: Cant use mixed types. Using -1 for "max"
    if args.pad_to is not None:
        featurizer_config['pad_to'] = args.pad_to if args.pad_to >= 0 else -1
    
    if featurizer_config['pad_to'] == "max":
        featurizer_config['pad_to'] = -1

    print('model_config')
    print_dict(jasper_model_definition)
    print('feature_config')
    print_dict(featurizer_config)

    data_layer = AudioToTextDataLayer(
                            dataset_dir=args.dataset_dir,
                            featurizer_config=featurizer_config,
                            manifest_filepath=val_manifest,
                            labels=dataset_vocab,
                            batch_size=batch_size,
                            pad_to_max=featurizer_config['pad_to'] == -1,
                            shuffle=False,
                            multi_gpu=False)

    audio_preprocessor = AudioPreprocessing(**featurizer_config)

    encoderdecoder = JasperEncoderDecoder(jasper_model_definition=jasper_model_definition, feat_in=1024, num_classes=len(ctc_vocab))

    if args.ckpt is not None:
        print("loading model from ", args.ckpt)
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        for k in audio_preprocessor.state_dict().keys():
            checkpoint['state_dict'][k] = checkpoint['state_dict'].pop("audio_preprocessor." + k)
        audio_preprocessor.load_state_dict(checkpoint['state_dict'], strict=False)
        encoderdecoder.load_state_dict(checkpoint['state_dict'], strict=False)

    greedy_decoder = GreedyCTCDecoder()

    # print("Number of parameters in encoder: {0}".format(model.jasper_encoder.num_weights()))

    N = len(data_layer)
    step_per_epoch = math.ceil(N / args.batch_size)

    print('-----------------')
    if args.steps is None:
        print('Have {0} examples to eval on.'.format(N))
        print('Have {0} steps / (gpu * epoch).'.format(step_per_epoch))
    else:
        print('Have {0} examples to eval on.'.format(args.steps * args.batch_size))
        print('Have {0} steps / (gpu * epoch).'.format(args.steps))
    print('-----------------')

    audio_preprocessor.cuda()
    encoderdecoder.cuda()
    if args.fp16:
        encoderdecoder = amp.initialize(
            models=encoderdecoder,
            opt_level=AmpOptimizations[optim_level])

    eval(
        data_layer=data_layer,
        audio_processor=audio_preprocessor,
        encoderdecoder=encoderdecoder,
        greedy_decoder=greedy_decoder,
        labels=ctc_vocab,
        args=args)

if __name__=="__main__":
    args = parse_args()

    print_dict(vars(args))

    main(args)
