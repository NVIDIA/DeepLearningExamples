# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team and Huawei Noah's Ark Lab.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import json
import math
import time

import numpy as np
import torch
from collections import namedtuple
from tempfile import TemporaryDirectory
from pathlib import Path
from torch.utils.data import (DataLoader, RandomSampler,Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import MSELoss

sys.path.append('/workspace/bert/')
from modeling import BertForPreTraining, BertModel, Project, WEIGHTS_NAME, CONFIG_NAME
from schedulers import LinearWarmUpScheduler, ConstantLR
from tokenization_utils import BertTokenizer
from apex.optimizers import FusedAdam

from hooks import *
from losses import *
from utils import data_utils
from utils.utils import is_main_process, get_rank, get_world_size, unwrap_ddp, set_seed
from itertools import chain
csv.field_size_limit(sys.maxsize)

import lddl.torch

# This is used for running on Huawei Cloud.
oncloud = True
try:
    import moxing as mox
except:
    oncloud = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_dir",
                        type=str,
                        required=True)
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument('--vocab_file',
                        type=str,
                        default=None,
                        required=True,
                        help="Vocabulary mapping/file BERT was pretrainined on")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--reduce_memory",
                        action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay',
                        '--wd',
                        default=1e-4,
                        type=float, metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--steps_per_epoch',
                        type=int,
                        default=-1,
                        help="Number of updates steps to in one epoch.")
    parser.add_argument('--max_steps',
                        type=int,
                        default=-1,
                        help="Number of training steps.")
    parser.add_argument('--amp',
                        action='store_true',
                        default=False,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--continue_train',
                        action='store_true',
                        default=False,
                        help='Whether to train from checkpoints')
    parser.add_argument('--disable_progress_bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=1.,
                        help="Gradient Clipping threshold")

    # Additional arguments
    parser.add_argument('--eval_step',
                        type=int,
                        default=1000)

    # This is used for running on Huawei Cloud.
    parser.add_argument('--data_url',
                        type=str,
                        default="")

    #Distillation specific
    parser.add_argument('--value_state_loss',
                        action='store_true',
                        default=False)
    parser.add_argument('--hidden_state_loss',
                        action='store_true',
                        default=False)
    parser.add_argument('--use_last_layer',
                        action='store_true',
                        default=False)
    parser.add_argument('--use_kld',
                        action='store_true',
                        default=False)
    parser.add_argument('--use_cosine',
                        action='store_true',
                        default=False)
    parser.add_argument('--distill_config',
                        default="distillation_config.json",
                        type=str,
                        help="path the distillation config")
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='number of DataLoader worker processes per rank')

    args = parser.parse_args()
    logger.info('args:{}'.format(args))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        stream=sys.stdout)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.amp))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    # Reference params
    author_gbs = 256
    author_steps_per_epoch = 22872
    author_epochs = 3
    author_max_steps = author_steps_per_epoch * author_epochs
    # Compute present run params
    if args.max_steps == -1 or args.steps_per_epoch == -1:
        args.steps_per_epoch = author_steps_per_epoch * author_gbs // (args.train_batch_size * get_world_size() * args.gradient_accumulation_steps)
        args.max_steps = author_max_steps * author_gbs // (args.train_batch_size * get_world_size() * args.gradient_accumulation_steps)

    #Set seed
    set_seed(args.seed, n_gpu)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and is_main_process():
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=args.do_lower_case)

    teacher_model, teacher_config = BertModel.from_pretrained(args.teacher_model,
                                              distill_config=args.distill_config)

    # Required to make sure model's fwd doesn't return anything. required for DDP.
    # fwd output not being used in loss computation crashes DDP
    teacher_model.make_teacher()

    if args.continue_train:
        student_model, student_config = BertForPreTraining.from_pretrained(args.student_model,
                                                           distill_config=args.distill_config)
    else:
        student_model, student_config = BertForPreTraining.from_scratch(args.student_model, 
                                                        distill_config=args.distill_config)

    # We need a projection layer since teacher.hidden_size != student.hidden_size
    use_projection = student_config.hidden_size != teacher_config.hidden_size
    if use_projection:
        project = Project(student_config, teacher_config)
        if args.continue_train:
            project_model_file = os.path.join(args.student_model, "project.bin")
            project_ckpt = torch.load(project_model_file, map_location="cpu")
            project.load_state_dict(project_ckpt)

    distill_config = {"nn_module_names": []} #Empty list since we don't want to use nn module hooks here
    distill_hooks_student, distill_hooks_teacher = DistillHooks(distill_config), DistillHooks(distill_config)

    student_model.register_forward_hook(distill_hooks_student.child_to_main_hook)
    teacher_model.register_forward_hook(distill_hooks_teacher.child_to_main_hook)

    ## Register hooks on nn.Modules
    # student_fwd_pre_hook = student_model.register_forward_pre_hook(distill_hooks_student.register_nn_module_hook)
    # teacher_fwd_pre_hook = teacher_model.register_forward_pre_hook(distill_hooks_teacher.register_nn_module_hook)

    student_model.to(device)
    teacher_model.to(device)
    if use_projection:
        project.to(device)
    if args.local_rank != -1:
        teacher_model = torch.nn.parallel.DistributedDataParallel(
               teacher_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False
           )
        student_model = torch.nn.parallel.DistributedDataParallel(
               student_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False
           )
        if use_projection:
            project = torch.nn.parallel.DistributedDataParallel(
                   project, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False
               )
    size = 0
    for n, p in student_model.named_parameters():
        logger.info('n: {}'.format(n))
        logger.info('p: {}'.format(p.nelement()))
        size += p.nelement()

    logger.info('Total parameters: {}'.format(size))

    # Prepare optimizer
    param_optimizer = list(student_model.named_parameters())
    if use_projection:
        param_optimizer += list(project.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)
    scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion, total_steps=args.max_steps)    

    global_step = 0
    logging.info("***** Running training *****")
    logging.info("  Num examples = {}".format(args.train_batch_size * args.max_steps))
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", args.max_steps)

    # Prepare the data loader.
    if is_main_process():
        tic = time.perf_counter()
    train_dataloader = lddl.torch.get_bert_pretrain_data_loader(
        args.input_dir,
        local_rank=args.local_rank,
        vocab_file=args.vocab_file,
        data_loader_kwargs={
            'batch_size': args.train_batch_size * n_gpu,
            'num_workers': args.num_workers,
            'pin_memory': True,
        },
        base_seed=args.seed,
        log_dir=None if args.output_dir is None else os.path.join(args.output_dir, 'lddl_log'),
        log_level=logging.WARNING,
        start_epoch=0,
    )
    if is_main_process():
        print('get_bert_pretrain_data_loader took {} s!'.format(time.perf_counter() - tic))
    train_dataloader = tqdm(train_dataloader, desc="Iteration", disable=args.disable_progress_bar) if is_main_process() else train_dataloader

    tr_loss, tr_att_loss, tr_rep_loss, tr_value_loss = 0., 0., 0., 0.
    nb_tr_examples, local_step = 0, 0

    student_model.train()
    scaler = torch.cuda.amp.GradScaler()

    transformer_losses = TransformerLosses(student_config, teacher_config, device, args)
    iter_start = time.time()
    while global_step < args.max_steps:
        for batch in train_dataloader:
            if global_step >= args.max_steps:
                break

            #remove forward_pre_hook after one forward pass
            #the purpose of forward_pre_hook is to register
            #forward_hooks on nn_module_names provided in config
            # if idx == 1:
            #     student_fwd_pre_hook.remove()
            #     teacher_fwd_pre_hook.remove()
            #     # return

            # Initialize loss metrics
            if global_step % args.steps_per_epoch == 0:
                tr_loss, tr_att_loss, tr_rep_loss, tr_value_loss = 0., 0., 0., 0.
                mean_loss, mean_att_loss, mean_rep_loss, mean_value_loss = 0., 0., 0., 0.

            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids, segment_ids, input_mask, lm_label_ids, is_next = batch['input_ids'], batch['token_type_ids'], batch['attention_mask'], batch['labels'], batch['next_sentence_labels']

            att_loss = 0.
            rep_loss = 0.
            value_loss = 0.
            with torch.cuda.amp.autocast(enabled=args.amp):
                student_model(input_ids, segment_ids, input_mask, None)

                # Gather student states extracted by hooks
                temp_model = unwrap_ddp(student_model)
                student_atts = flatten_states(temp_model.distill_states_dict, "attention_scores")
                student_reps = flatten_states(temp_model.distill_states_dict, "hidden_states")
                student_values = flatten_states(temp_model.distill_states_dict, "value_states")
                student_embeddings = flatten_states(temp_model.distill_states_dict, "embedding_states")
                bsz, attn_heads, seq_len, _  = student_atts[0].shape

                #No gradient for teacher training
                with torch.no_grad():
                    teacher_model(input_ids, segment_ids, input_mask)

                # Gather teacher states extracted by hooks
                temp_model = unwrap_ddp(teacher_model)
                teacher_atts = [i.detach() for i in flatten_states(temp_model.distill_states_dict, "attention_scores")]
                teacher_reps = [i.detach() for i in flatten_states(temp_model.distill_states_dict, "hidden_states")]
                teacher_values = [i.detach() for i in flatten_states(temp_model.distill_states_dict, "value_states")]
                teacher_embeddings = [i.detach() for i in flatten_states(temp_model.distill_states_dict, "embedding_states")]

                teacher_layer_num = len(teacher_atts)
                student_layer_num = len(student_atts)

                #MiniLM
                if student_config.distillation_config["student_teacher_layer_mapping"] == "last_layer":
                    if student_config.distillation_config["use_attention_scores"]:
                        student_atts = [student_atts[-1]]
                        new_teacher_atts = [teacher_atts[-1]]

                    if student_config.distillation_config["use_value_states"]:
                        student_values = [student_values[-1]]
                        new_teacher_values = [teacher_values[-1]]

                    if student_config.distillation_config["use_hidden_states"]:
                        new_teacher_reps = [teacher_reps[-1]]
                        new_student_reps = [student_reps[-1]]
                else:
                    assert teacher_layer_num % student_layer_num == 0

                    layers_per_block = int(teacher_layer_num / student_layer_num)
                    if student_config.distillation_config["use_attention_scores"]:
                        new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                            for i in range(student_layer_num)]

                    if student_config.distillation_config["use_value_states"]:
                        new_teacher_values = [teacher_values[i * layers_per_block + layers_per_block - 1]
                                    for i in range(student_layer_num)]

                    if student_config.distillation_config["use_hidden_states"]:
                        new_teacher_reps = [teacher_reps[i * layers_per_block + layers_per_block - 1]
                                    for i in range(student_layer_num)]
                        new_student_reps = student_reps

                if student_config.distillation_config["use_attention_scores"]:
                    att_loss = transformer_losses.compute_loss(student_atts, new_teacher_atts, loss_name="attention_loss")

                if student_config.distillation_config["use_hidden_states"]:
                    if use_projection:
                        rep_loss = transformer_losses.compute_loss(project(new_student_reps), new_teacher_reps, loss_name="hidden_state_loss")
                    else:
                        rep_loss = transformer_losses.compute_loss(new_student_reps, new_teacher_reps, loss_name="hidden_state_loss")

                if student_config.distillation_config["use_embedding_states"]:
                    if use_projection:
                        rep_loss += transformer_losses.compute_loss(project(student_embeddings), teacher_embeddings, loss_name="embedding_state_loss")
                    else:
                        rep_loss += transformer_losses.compute_loss(student_embeddings, teacher_embeddings, loss_name="embedding_state_loss")

                if student_config.distillation_config["use_value_states"]:
                    value_loss = transformer_losses.compute_loss(student_values, new_teacher_values, loss_name="value_state_loss")

                loss = att_loss + rep_loss + value_loss


            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            tr_att_loss += att_loss.item() / args.gradient_accumulation_steps
            if student_config.distillation_config["use_hidden_states"]:
                tr_rep_loss += rep_loss.item() / args.gradient_accumulation_steps
            if student_config.distillation_config["use_value_states"]:
                tr_value_loss += value_loss.item() / args.gradient_accumulation_steps
            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            if use_projection:
                torch.nn.utils.clip_grad_norm_(chain(student_model.parameters(), project.parameters()), args.max_grad_norm, error_if_nonfinite=False)
            else:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm, error_if_nonfinite=False)

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            local_step += 1

            if local_step % args.gradient_accumulation_steps == 0:
                scheduler.step()
                if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                global_step = optimizer.param_groups[0]["step"] if "step" in optimizer.param_groups[0] else 0

                if (global_step % args.steps_per_epoch) > 0:
                    mean_loss = tr_loss / (global_step % args.steps_per_epoch)
                    mean_att_loss = tr_att_loss / (global_step % args.steps_per_epoch)
                    mean_rep_loss = tr_rep_loss / (global_step % args.steps_per_epoch)
                    value_loss = tr_value_loss / (global_step % args.steps_per_epoch)

                if (global_step + 1) % args.eval_step == 0 and is_main_process():
                    result = {}
                    result['global_step'] = global_step
                    result['lr'] = optimizer.param_groups[0]["lr"]
                    result['loss'] = mean_loss
                    result['att_loss'] = mean_att_loss
                    result['rep_loss'] = mean_rep_loss
                    result['value_loss'] = value_loss
                    result['perf'] = (global_step + 1) * get_world_size() * args.train_batch_size * args.gradient_accumulation_steps / (time.time() - iter_start)
                    output_eval_file = os.path.join(args.output_dir, "log.txt")
                    if is_main_process():
                        with open(output_eval_file, "a") as writer:
                            logger.info("***** Eval results *****")
                            for key in sorted(result.keys()):
                                logger.info("  %s = %s", key, str(result[key]))
                                writer.write("%s = %s\n" % (key, str(result[key])))

                        # Save a trained model
                        model_name = "{}".format(WEIGHTS_NAME)

                        logging.info("** ** * Saving fine-tuned model ** ** * ")
                        # Only save the model it-self
                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                        if use_projection:
                            project_to_save = project.module if hasattr(project, 'module') else project

                        output_model_file = os.path.join(args.output_dir, model_name)
                        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                        output_project_file = os.path.join(args.output_dir, "project.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        if use_projection:
                            torch.save(project_to_save.state_dict(), output_project_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir)

                        if oncloud:
                            logging.info(mox.file.list_directory(args.output_dir, recursive=True))
                            logging.info(mox.file.list_directory('.', recursive=True))
                            mox.file.copy_parallel(args.output_dir, args.data_url)
                            mox.file.copy_parallel('.', args.data_url)

    model_name = "{}".format(WEIGHTS_NAME)
    logging.info("** ** * Saving fine-tuned model ** ** * ")
    model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

    if use_projection:
        project_to_save = project.module if hasattr(project, 'module') else project
        output_project_file = os.path.join(args.output_dir, "project.bin")
        if is_main_process():
            torch.save(project_to_save.state_dict(), output_project_file)

    output_model_file = os.path.join(args.output_dir, model_name)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    if is_main_process():
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

    if oncloud:
        logging.info(mox.file.list_directory(args.output_dir, recursive=True))
        logging.info(mox.file.list_directory('.', recursive=True))
        mox.file.copy_parallel(args.output_dir, args.data_url)
        mox.file.copy_parallel('.', args.data_url)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total time taken:", time.time() - start_time)
