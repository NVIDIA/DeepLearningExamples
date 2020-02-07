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
from typing import List
from tqdm import tqdm
import math
import toml
from dataset import AudioToTextDataLayer
from helpers import process_evaluation_batch, process_evaluation_epoch, add_ctc_labels, AmpOptimizations, print_dict, model_multi_gpu, __ctc_decoder_predictions_tensor
from model import AudioPreprocessing, GreedyCTCDecoder, JasperEncoderDecoder
from parts.features import audio_from_file
import torch
import torch.nn as nn
import apex
from apex import amp
import random
import numpy as np
import pickle
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Jasper')

    parser.register("type", "bool", lambda x: x.lower() in ("yes", "true", "t", "1"))

    parser.add_argument("--local_rank", default=None, type=int)
    parser.add_argument("--batch_size", default=16, type=int, help='data batch size')
    parser.add_argument("--steps", default=None, help='if not specified do evaluation on full dataset. otherwise only evaluates the specified number of iterations for each worker', type=int)
    parser.add_argument("--model_toml", type=str, help='relative model configuration path given dataset folder')
    parser.add_argument("--dataset_dir", type=str, help='absolute path to dataset folder')
    parser.add_argument("--val_manifest", type=str, help='relative path to evaluation dataset manifest file')
    parser.add_argument("--ckpt", default=None, type=str, required=True, help='path to model checkpoint')
    parser.add_argument("--max_duration", default=None, type=float, help='maximum duration of sequences. if None uses attribute from model configuration file')
    parser.add_argument("--pad_to", default=None, type=int, help="default is pad to value as specified in model configurations. if -1 pad to maximum duration. If > 0 pad batch to next multiple of value")
    parser.add_argument("--fp16", action='store_true', help='use half precision')
    parser.add_argument("--pyt_fp16", action='store_true', help='use half precision')
    parser.add_argument("--cudnn_benchmark", action='store_true', help="enable cudnn benchmark")
    parser.add_argument("--save_prediction", type=str, default=None, help="if specified saves predictions in text form at this location")
    parser.add_argument("--logits_save_to", default=None, type=str, help="if specified will save logits to path")
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--masked_fill", type="bool", help="Overrides the masked_fill option for the Encoder")
    parser.add_argument("--output_dir", default="results/", type=str, help="Output directory to store exported models. Only used if --export_model is used")
    parser.add_argument("--export_model", action='store_true', help="Exports the audio_featurizer, encoder and decoder using torch.jit to the output_dir")
    parser.add_argument("--wav", type=str, help='absolute path to .wav file (16KHz)')
    return parser.parse_args()

def calc_wer(data_layer, audio_processor, 
             encoderdecoder, greedy_decoder, 
             labels, args):

    encoderdecoder = encoderdecoder.module if hasattr(encoderdecoder, 'module') else encoderdecoder
    with torch.no_grad():
        # reset global_var_dict - results of evaluation will be stored there
        _global_var_dict = {
            'predictions': [],
            'transcripts': [],
            'logits' : [],
        }

        # Evaluation mini-batch for loop
        for it, data in enumerate(tqdm(data_layer.data_iterator)):

            tensors = []
            for d in data:
                tensors.append(d.cuda())
    
            t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = tensors
    
            t_processed_signal = audio_processor(t_audio_signal_e, t_a_sig_length_e) 
            t_log_probs_e, _ = encoderdecoder.infer(t_processed_signal)
            t_predictions_e = greedy_decoder(t_log_probs_e)
    
            values_dict = dict(
                predictions=[t_predictions_e],
                transcript=[t_transcript_e],
                transcript_length=[t_transcript_len_e],
                output=[t_log_probs_e]
            )
            # values_dict will contain results from all workers
            process_evaluation_batch(values_dict, _global_var_dict, labels=labels)
    
            if args.steps is not None and it + 1 >= args.steps:
                break

        # final aggregation (over minibatches) and logging of results
        wer, _ = process_evaluation_epoch(_global_var_dict)

        return wer, _global_var_dict


def jit_export(
         audio, audio_len,
         audio_processor,
         encoderdecoder,
         greedy_decoder,
         args):

                print("##############")

                module_name = "{}_{}".format(os.path.basename(args.model_toml), "fp16" if args.fp16 else "fp32")

                if args.masked_fill is not None and args.masked_fill == False:
                    module_name = module_name + "_noMaskConv"

                # Export just the featurizer
                print("exporting featurizer ...")
                traced_module_feat = torch.jit.script(audio_processor)
                traced_module_feat.save(os.path.join(args.output_dir, module_name + "_feat.pt"))

                # Export just the acoustic model
                print("exporting acoustic model ...")
                inp_postFeat, _ = audio_processor(audio, audio_len)
                traced_module_acoustic = torch.jit.trace(encoderdecoder, inp_postFeat)
                traced_module_acoustic.save(os.path.join(args.output_dir, module_name + "_acoustic.pt"))

                # Export just the decoder
                print("exporting decoder ...")

                inp_postAcoustic = encoderdecoder(inp_postFeat)
                traced_module_decode = torch.jit.script(greedy_decoder, inp_postAcoustic)
                traced_module_decode.save(os.path.join(args.output_dir, module_name + "_decoder.pt"))
                print("JIT export complete")

                return traced_module_feat, traced_module_acoustic, traced_module_decode

def run_once(audio_processor, encoderdecoder, greedy_decoder, audio, audio_len, labels):
            features = audio_processor(audio, audio_len)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            t_log_probs_e = encoderdecoder(features[0])
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            t_predictions_e = greedy_decoder(log_probs=t_log_probs_e)
            hypotheses = __ctc_decoder_predictions_tensor(t_predictions_e, labels=labels)
            print("INFERENCE TIME\t\t: {} ms".format((t1-t0)*1000.0))
            print("TRANSCRIPT\t\t:", hypotheses[0])


def eval(
         data_layer,
         audio_processor,
         encoderdecoder,
         greedy_decoder,
         labels,
         multi_gpu,
         args):
    """performs inference / evaluation
    Args:
        data_layer: data layer object that holds data loader
        audio_processor: data processing module
        encoderdecoder: acoustic model
        greedy_decoder: greedy decoder
        labels: list of labels as output vocabulary
        multi_gpu: true if using multiple gpus
        args: script input arguments
    """
    logits_save_to=args.logits_save_to
    
    with torch.no_grad():
        if args.wav:
            audio, audio_len = audio_from_file(args.wav)
            run_once(audio_processor, encoderdecoder, greedy_decoder, audio, audio_len, labels)
            if args.export_model:
                jit_audio_processor, jit_encoderdecoder, jit_greedy_decoder = jit_export(audio, audio_len, audio_processor,
                                                                                         encoderdecoder,
                                                                                         greedy_decoder,args)
            run_once(jit_audio_processor, jit_encoderdecoder, jit_greedy_decoder, audio, audio_len, labels)
            return
        wer, _global_var_dict = calc_wer(data_layer, audio_processor, encoderdecoder, greedy_decoder, labels, args)
        if (not multi_gpu or (multi_gpu and torch.distributed.get_rank() == 0)):
            print("==========>>>>>>Evaluation WER: {0}\n".format(wer))
      
            if args.save_prediction is not None:
                with open(args.save_prediction, 'w') as fp:
                    fp.write('\n'.join(_global_var_dict['predictions']))
            if logits_save_to is not None:
                logits = []
                for batch in _global_var_dict["logits"]:
                    for i in range(batch.shape[0]):
                        logits.append(batch[i].cpu().numpy())
                with open(logits_save_to, 'wb') as f:
                    pickle.dump(logits, f, protocol=pickle.HIGHEST_PROTOCOL)

            # if args.export_model:            
            #     feat, acoustic, decoder = jit_export(inp, audio_processor, encoderdecoder, greedy_decoder,args)
            #     wer_after = calc_wer(data_layer, feat, acoustic, decoder, labels, args)
            #     print("===>>>Before WER: {0}".format(wer))
            #     print("===>>>Traced WER: {0}".format(wer_after))
            #     print("===>>>Diff      : {0} %".format((wer_after - wer_before) * 100.0 / wer_before))
            #     print("")

                
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    print("CUDNN BENCHMARK ", args.cudnn_benchmark)
    assert(torch.cuda.is_available())

    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    multi_gpu = args.local_rank is not None
    if multi_gpu:
        print("DISTRIBUTED with ", torch.distributed.get_world_size())

    if args.fp16:
        optim_level = 3
    else:
        optim_level = 0

    jasper_model_definition = toml.load(args.model_toml)
    dataset_vocab = jasper_model_definition['labels']['labels']
    ctc_vocab = add_ctc_labels(dataset_vocab)

    val_manifest = args.val_manifest
    featurizer_config = jasper_model_definition['input_eval']
    featurizer_config["optimization_level"] = optim_level
    featurizer_config["fp16"] = args.fp16
    args.use_conv_mask = jasper_model_definition['encoder'].get('convmask', True)

    if args.masked_fill is not None:
        print("{} masked_fill".format("Enabling" if args.masked_fill else "Disabling"))
        jasper_model_definition["encoder"]["conv_mask"] = args.masked_fill

    if args.max_duration is not None:
        featurizer_config['max_duration'] = args.max_duration
    if args.pad_to is not None:
        featurizer_config['pad_to'] = args.pad_to 

    if featurizer_config['pad_to'] == "max":
        featurizer_config['pad_to'] = -1
        
    print('=== model_config ===')
    print_dict(jasper_model_definition)
    print()
    print('=== feature_config ===')
    print_dict(featurizer_config)
    print()
    data_layer = None
    
    if args.wav is None:
        data_layer = AudioToTextDataLayer(
            dataset_dir=args.dataset_dir, 
            featurizer_config=featurizer_config,
            manifest_filepath=val_manifest,
            labels=dataset_vocab,
            batch_size=args.batch_size,
            pad_to_max=featurizer_config['pad_to'] == -1,
            shuffle=False,
            multi_gpu=multi_gpu)
    audio_preprocessor = AudioPreprocessing(**featurizer_config)
    encoderdecoder = JasperEncoderDecoder(jasper_model_definition=jasper_model_definition, feat_in=1024, num_classes=len(ctc_vocab))

    if args.ckpt is not None:
        print("loading model from ", args.ckpt)

        if os.path.isdir(args.ckpt):
            exit(0)
        else:
            checkpoint = torch.load(args.ckpt, map_location="cpu")
            for k in audio_preprocessor.state_dict().keys():
                checkpoint['state_dict'][k] = checkpoint['state_dict'].pop("audio_preprocessor." + k)
            audio_preprocessor.load_state_dict(checkpoint['state_dict'], strict=False)
            encoderdecoder.load_state_dict(checkpoint['state_dict'], strict=False)

    greedy_decoder = GreedyCTCDecoder()

    # print("Number of parameters in encoder: {0}".format(model.jasper_encoder.num_weights()))
    if args.wav is None:
        N = len(data_layer)
        step_per_epoch = math.ceil(N / (args.batch_size * (1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())))

        if args.steps is not None:
            print('-----------------')
            print('Have {0} examples to eval on.'.format(args.steps * args.batch_size * (1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())))
            print('Have {0} steps / (gpu * epoch).'.format(args.steps))
            print('-----------------')
        else:
            print('-----------------')
            print('Have {0} examples to eval on.'.format(N))
            print('Have {0} steps / (gpu * epoch).'.format(step_per_epoch))
            print('-----------------')

    print ("audio_preprocessor.normalize: ", audio_preprocessor.featurizer.normalize)
    audio_preprocessor.cuda()
    encoderdecoder.cuda()
    if args.fp16:
        encoderdecoder = amp.initialize( models=encoderdecoder,
                                         opt_level=AmpOptimizations[optim_level])

    encoderdecoder = model_multi_gpu(encoderdecoder, multi_gpu)
    audio_preprocessor.eval()
    encoderdecoder.eval()
    greedy_decoder.eval()
    
    eval(
        data_layer=data_layer,
        audio_processor=audio_preprocessor,
        encoderdecoder=encoderdecoder,
        greedy_decoder=greedy_decoder,
        labels=ctc_vocab,
        args=args,
        multi_gpu=multi_gpu)

if __name__=="__main__":
    args = parse_args()

    print_dict(vars(args))

    main(args)
