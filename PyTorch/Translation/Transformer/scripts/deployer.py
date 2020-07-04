#!/usr/bin/python

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved. 
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


import sys
import torch
import argparse
import deployer_lib
# 
import torch
from fairseq import data
from fairseq.data import load_dataset_splits, data_utils
from fairseq.models.transformer import TransformerModel
from copy import deepcopy

def get_model_and_args(model_args):
    ''' the arguments initialize_model will receive '''
    parser = argparse.ArgumentParser()
    ## Required parameters by the model. 
    parser.add_argument("--checkpoint", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="The checkpoint of the model. ")
    parser.add_argument('--batch-size', 
                        default=10240, 
                        type=int, 
                        help='Batch size for inference')
    parser.add_argument('--num-batches',
                        default=2,
                        type=int,
                        help='Number of batches to check accuracy on')
    parser.add_argument("--data",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to the dataset")
    parser.add_argument('--part',
                        choices=['encoder', 'decoder', 'model'],
                        default='model',
                        type=str,
                        help='Choose the part of the model to export')

    args = parser.parse_args(model_args)

    state_dict = torch.load(args.checkpoint, map_location='cpu')

    model_args = state_dict['args']
    model_args.data = args.data
    model_args.num_batches = args.num_batches
    model_args.max_tokens = args.batch_size
    model_args.fuse_layer_norm = False
    model_args.part = args.part

    model = TransformerModel.build_model(model_args)
    model.load_state_dict(state_dict['model'], strict=True)
    model.make_generation_fast_(need_attn=False)

    return model, model_args

def get_dataloader(args, encoder=None):
    ''' return dataloader for inference '''
    assert not(args.part == 'decoder' and encoder is None), "Cannot export decoder without providing encoder"
    src_dict, tgt_dict = data_utils.load_dictionaries(args)
    datasets = load_dataset_splits(args, ['valid'], src_dict, tgt_dict)
    itr = data.EpochBatchIterator(
        dataset=datasets['valid'],
        max_tokens=args.max_tokens,
        max_positions=args.max_positions,
    ).next_epoch_itr(shuffle=False)

    def input_itr():
        for batch in itr:
            if itr.count > args.num_batches:
                break
            ni = batch['net_input']
            if args.part == 'decoder': #this part works only on GPU
                with torch.no_grad():
                    encoder_out = encoder(ni['src_tokens'].cuda(), ni['src_lengths'].cuda()) 
                yield ni['prev_output_tokens'], encoder_out[0], encoder_out[1]
            elif args.part == 'encoder':
                yield ni['src_tokens'], ni['src_lengths']
            else:
                yield ni['src_tokens'], ni['src_lengths'], ni['prev_output_tokens']

    return input_itr()


if __name__=='__main__':
    # don't touch this! 
    deployer, model_argv = deployer_lib.create_deployer(sys.argv[1:]) # deployer and returns removed deployer arguments
    
    model, model_args = get_model_and_args(model_argv)

    if model_args.part == 'decoder':
        encoder = model.encoder
        encoder.embed_tokens = deepcopy(encoder.embed_tokens)
        encoder.cuda()
    else:
        encoder = None
    
    dataloader = get_dataloader(model_args, encoder=encoder)

    if model_args.part == 'encoder':
        model = model.encoder
    elif model_args.part == 'decoder':
        model = model.decoder
    
    deployer.deploy(dataloader, model)

