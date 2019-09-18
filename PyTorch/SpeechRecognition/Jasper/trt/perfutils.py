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
'''Contains helper functions for non-TRT components of JASPER inference
'''

from model import GreedyCTCDecoder, AudioPreprocessing, Jasper
from dataset import AudioToTextDataLayer
from helpers import Optimization, AmpOptimizations, process_evaluation_batch, process_evaluation_epoch, add_ctc_labels, norm
from apex import amp
import torch
import torch.nn as nn
import toml
from parts.features import audio_from_file

_global_ctc_labels = None
def get_vocab():
    ''' Gets the CTC vocab

    Requires calling get_pytorch_components_and_onnx() to setup global labels.
    '''
    if _global_ctc_labels is None:
        raise Exception("Feature labels have not been found. Execute `get_pytorch_components_and_onnx()` first")

    return _global_ctc_labels

def get_results(log_probs, original_tensors, batch_size):
    ''' Returns WER and predictions for the outputs of the acoustic model

    Used for one-off batches. Epoch-wide evaluation should use
    global_process_batch and global_process_epoch
    '''
    # Used to get WER and predictions for one-off batches
    greedy_decoder = GreedyCTCDecoder()
    predicts = norm(greedy_decoder(log_probs=log_probs))
    values_dict = dict(
        predictions=[predicts],
        transcript=[original_tensors[2][0:batch_size,...]],
        transcript_length=[original_tensors[3][0:batch_size,...]],
    )
    temp_dict = {
        'predictions': [],
        'transcripts': [],
    }
    process_evaluation_batch(values_dict, temp_dict, labels=get_vocab())
    predictions = temp_dict['predictions']
    wer, _ = process_evaluation_epoch(temp_dict)
    return wer, predictions


_global_trt_dict = {
        'predictions': [],
        'transcripts': [],
}
_global_pyt_dict = {
        'predictions': [],
        'transcripts': [],
}

def global_process_batch(log_probs, original_tensors, batch_size, is_trt=True):
    '''Accumulates prediction evaluations for batches across an epoch

    is_trt determines which global dictionary will be used.
    To get WER at any point, use global_process_epoch.
    For one-off WER evaluations, use get_results()
    '''
    # State-based approach for full WER comparison across a dataset.
    greedy_decoder = GreedyCTCDecoder()
    predicts = norm(greedy_decoder(log_probs=log_probs))
    values_dict = dict(
        predictions=[predicts],
        transcript=[original_tensors[2][0:batch_size,...]],
        transcript_length=[original_tensors[3][0:batch_size,...]],
    )
    dict_to_process = _global_trt_dict if is_trt else _global_pyt_dict
    process_evaluation_batch(values_dict, dict_to_process, labels=get_vocab())


def global_process_epoch(is_trt=True):
    '''Returns WER in accumulated global dictionary
    '''
    dict_to_process = _global_trt_dict if is_trt else _global_pyt_dict
    wer, _ = process_evaluation_epoch(dict_to_process)
    return wer


def get_onnx(path, acoustic_model, signal_shape, dtype=torch.float):
    ''' Get an ONNX model with float weights

    Requires an --onnx_save_path and --ckpt_path (so that an acoustic model could be constructed).
    Fixed-length --seq_len must be provided as well.
    '''
    with torch.no_grad():
        phony_signal = torch.zeros(signal_shape, dtype=dtype, device=torch.device("cuda"))
        torch.onnx.export(acoustic_model, (phony_signal,), path, input_names=["FEATURES"], output_names=["LOGITS"])
        fn=path+".readable"
        with open(fn, 'w') as f:
            #Write human-readable graph representation to file as well.
            import onnx
            tempModel = onnx.load(path)
            pgraph = onnx.helper.printable_graph(tempModel.graph)
            f.write(pgraph)

    return path


def get_pytorch_components_and_onnx(args):
    '''Returns PyTorch components used for inference
    '''
    model_definition = toml.load(args.model_toml)
    dataset_vocab = model_definition['labels']['labels']
    # Set up global labels for future vocab calls
    global _global_ctc_labels
    _global_ctc_labels= add_ctc_labels(dataset_vocab)
    featurizer_config = model_definition['input_eval']

    optim_level = Optimization.mxprO3 if args.pyt_fp16 else Optimization.mxprO0

    featurizer_config["optimization_level"] = optim_level
    acoustic_model = None
    audio_preprocessor = None
    onnx_path = None
    data_layer = None
    wav = None
    seq_len = None
    dtype=torch.float
    
    if args.max_duration is not None:
        featurizer_config['max_duration'] = args.max_duration
    if args.dataset_dir is not None:    
        data_layer =  AudioToTextDataLayer(dataset_dir=args.dataset_dir,
                                           featurizer_config=featurizer_config,
                                           manifest_filepath=args.val_manifest,
                                           labels=dataset_vocab,
                                           batch_size=args.batch_size,
                                           shuffle=False)
    if args.wav is not None:
        args.batch_size=1
        args.engine_batch_size=1
        wav, seq_len = audio_from_file(args.wav)
        if args.seq_len is None or args.seq_len == 0:
            args.seq_len = seq_len/(featurizer_config['sample_rate']/100)
        

    model = Jasper(feature_config=featurizer_config,
                   jasper_model_definition=model_definition,
                   feat_in=1024,
                   num_classes=len(get_vocab()))

    model.cuda()
    model.eval()
    acoustic_model = model.acoustic_model
    audio_preprocessor = model.audio_preprocessor

    if args.ckpt_path is not None:
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
    if args.make_onnx:
        if args.onnx_path is None or acoustic_model is None:
            raise Exception("--ckpt_path, --onnx_path must be provided when using --make_onnx")
        onnx_path = get_onnx(args.onnx_path, acoustic_model,
                             signal_shape=(args.engine_batch_size, 64, args.seq_len), dtype=torch.float)

    if args.pyt_fp16:
        amp.initialize(models=acoustic_model, opt_level=AmpOptimizations[optim_level])
        
    return {'data_layer': data_layer,
            'audio_preprocessor': audio_preprocessor,
            'acoustic_model': acoustic_model,
            'input_wav' : (wav, seq_len) }, onnx_path

def adjust_shape(am_input, baked_length):
    '''Pads or cuts acoustic model input tensor to some fixed_length

    '''
    in_seq_len = am_input[0].shape[2]
    newSeq=am_input[0]
    if in_seq_len > baked_length:
        # Cut extra bits off, no inference done
        newSeq = am_input[0][...,0:baked_length].contiguous()
    elif in_seq_len < baked_length:
        # Zero-pad to satisfy length
        pad_length = baked_length - in_seq_len
        newSeq = nn.functional.pad(am_input[0], (0, pad_length), 'constant', 0)
    return (newSeq,)

def torchify_trt_out(trt_out, batch_size):
    '''Reshapes flat data to format for greedy+CTC decoding

    Used to convert numpy array on host to PyT Tensor
    '''
    desired_shape = (batch_size,-1,len(get_vocab()))

    # Predictions must be reshaped.
    return torch.Tensor(trt_out).reshape(desired_shape)

def do_csv_export(wers, times, batch_size, num_frames):
    '''Produces CSV header and data for input data

    wers: dictionary of WER with keys={'trt', 'pyt'}
    times: dictionary of execution times
    '''
    def take_durations_and_output_percentile(durations, ratios):
        from heapq import nlargest, nsmallest
        import numpy as np
        import math
        durations = np.asarray(durations) * 1000 # in ms
        latency = durations
        # The first few entries may not be representative due to warm-up effects
        # The last entry might not be representative if dataset_size % batch_size != 0
        latency = latency[5:-1]
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

    ratios = [0.9, 0.95, 0.99, 1.]
    header=[]
    data=[]
    header.append("BatchSize")
    header.append("NumFrames")
    data.append(f"{batch_size}")
    data.append(f"{num_frames}")
    for title, wer in wers.items():
        header.append(title)
        data.append(f"{wer}")
    for title, durations in times.items():
        ratio_latencies_dict = take_durations_and_output_percentile(durations, ratios)
        for ratio, latency in ratio_latencies_dict.items():
            header.append(f"{title}_{ratio}")
            data.append(f"{latency}")
    string_header = ", ".join(header)
    string_data = ", ".join(data)
    return string_header, string_data
