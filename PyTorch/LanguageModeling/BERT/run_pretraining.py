# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#==================
import csv
import os
import logging
import argparse
import random
import h5py
from tqdm import tqdm, trange
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import math
from apex import amp



from tokenization import BertTokenizer
from modeling import BertForPreTraining, BertConfig
from optimization import BertAdam, BertAdam_FP16

# from fused_adam_local import FusedAdamBert
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from apex.optimizers import FusedAdam #, FP16_Optimizer
#from apex.optimizers import FusedAdam
from apex.parallel import DistributedDataParallel as DDP
from schedulers import LinearWarmUpScheduler

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        self.input_ids = np.asarray(f["input_ids"][:]).astype(np.int64)#[num_instances x max_seq_length])
        self.input_masks = np.asarray(f["input_mask"][:]).astype(np.int64) #[num_instances x max_seq_length]
        self.segment_ids = np.asarray(f["segment_ids"][:]).astype(np.int64) #[num_instances x max_seq_length]
        self.masked_lm_positions = np.asarray(f["masked_lm_positions"][:]).astype(np.int64) #[num_instances x max_pred_length]
        self.masked_lm_ids= np.asarray(f["masked_lm_ids"][:]).astype(np.int64) #[num_instances x max_pred_length]
        self.next_sentence_labels = np.asarray(f["next_sentence_labels"][:]).astype(np.int64) # [num_instances]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_ids)

    def __getitem__(self, index):
        
        input_ids= torch.from_numpy(self.input_ids[index]) # [max_seq_length]
        input_mask = torch.from_numpy(self.input_masks[index]) #[max_seq_length]
        segment_ids = torch.from_numpy(self.segment_ids[index])# [max_seq_length]
        masked_lm_positions = torch.from_numpy(self.masked_lm_positions[index]) #[max_pred_length]
        masked_lm_ids = torch.from_numpy(self.masked_lm_ids[index]) #[max_pred_length]
        next_sentence_labels = torch.from_numpy(np.asarray(self.next_sentence_labels[index])) #[1]
         
        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        if len((masked_lm_positions == 0).nonzero()) != 0:
          index = (masked_lm_positions == 0).nonzero()[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels]

def main():    

    print("IN NEW MAIN XD\n")
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
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
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=10.0,
                        help='frequency of logging loss.')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=2000,
                        help="Number of update steps until a model checkpoint is saved to disk.")


    args = parser.parse_args()


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert(torch.cuda.is_available())

    if args.local_rank == -1:
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
                            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps



    if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (os.listdir(args.output_dir) and os.listdir(args.output_dir)!=['logfile.txt']):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if not args.resume_from_checkpoint:
        os.makedirs(args.output_dir, exist_ok=True)

    # Prepare model
    config = BertConfig.from_json_file(args.config_file)
    model = BertForPreTraining(config)


    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if args.resume_step == -1:
            model_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
            args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])
        
        global_step = args.resume_step

        checkpoint = torch.load(os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step)), map_location="cpu")
        model.load_state_dict(checkpoint['model'], strict=False)

        print("resume step from ", args.resume_step)

    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    

    if args.fp16:

        optimizer = FusedAdam(optimizer_grouped_parameters,
                                    lr=args.learning_rate,
                                    #warmup=args.warmup_proportion,
                                    #t_total=args.max_steps,
                                    bias_correction=False,
                                    weight_decay=0.01,
                                    max_grad_norm=1.0)

        if args.loss_scale == 0:
            # optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False, loss_scale="dynamic")
        else:
            # optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False, loss_scale=args.loss_scale)

        scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion, total_steps=args.max_steps)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                warmup=args.warmup_proportion,
                                t_total=args.max_steps)
        


    if args.resume_from_checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])  # , strict=False)
       

        
    if args.local_rank != -1:
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
   
    files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    files.sort()

    num_files = len(files)
      

    logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_data))
    logger.info("  Batch size = %d", args.train_batch_size)
    print("  LR = ", args.learning_rate)
    

    model.train()
    print("Training. . .")

    most_recent_ckpts_paths = []

    print("Training. . .")
    tr_loss = 0.0 # total added training loss
    average_loss = 0.0 # averaged loss every args.log_freq steps
    epoch = 0
    training_steps = 0
    while True:
        if not args.resume_from_checkpoint:
            random.shuffle(files)
            f_start_id = 0
        else:
            f_start_id = checkpoint['files'][0]
            files = checkpoint['files'][1:]
            args.resume_from_checkpoint = False
        for f_id in range(f_start_id, len(files)):
            data_file = files[f_id]
            logger.info("file no %s file %s" %(f_id, data_file))
            train_data = pretraining_dataset(input_file=data_file, max_pred_length=args.max_predictions_per_seq)

            if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size * n_gpu, num_workers=4, pin_memory=True)
            else:
            train_sampler = DistributedSampler(train_data)

            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
            for step, batch in enumerate(tqdm(train_dataloader, desc="File Iteration")):
            
                training_steps += 1
                batch = [t.to(device) for t in batch]
                input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch#\
                loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_labels, checkpoint_activations=args.checkpoint_activations)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                #   optimizer.backward(loss)
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                tr_loss += loss
                average_loss += loss.item()

                if training_steps % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                
            

                if training_steps == 1 * args.gradient_accumulation_steps:
                    logger.info("Step:{} Average Loss = {} Step Loss = {} LR {}".format(global_step, average_loss, 
                                                                                loss.item(), optimizer.param_groups[0]['lr']))

                if training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                    logger.info("Step:{} Average Loss = {} Step Loss = {} LR {}".format(global_step,  average_loss / args.log_freq, 
                                                                                loss.item(), optimizer.param_groups[0]['lr']))
                    average_loss = 0


                if global_step >= args.max_steps or training_steps == 1 * args.gradient_accumulation_steps or training_steps % (args.num_steps_per_checkpoint * args.gradient_accumulation_steps) == 0:
                    if (not torch.distributed.is_initialized() or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)):
                        # Save a trained model
                        logger.info("** ** * Saving fine - tuned model ** ** * ")
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step))
                       
                        torch.save({'model' : model_to_save.state_dict(), 
                                'optimizer' : optimizer.state_dict(), 
                                'files' : [f_id] + files }, output_save_file)
                                
                        most_recent_ckpts_paths.append(output_save_file)
                        if len(most_recent_ckpts_paths) > 3:
                            ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                            os.remove(ckpt_to_be_removed)

                    if global_step >= args.max_steps:
                        tr_loss = tr_loss * args.gradient_accumulation_steps / training_steps
                        if (torch.distributed.is_initialized()):
                            tr_loss /= torch.distributed.get_world_size()
                            torch.distributed.all_reduce(tr_loss)
                        logger.info("Total Steps:{} Final Loss = {}".format(training_steps, tr_loss.item()))
                        return
            del train_dataloader
            del train_sampler
            del train_data       
            #for obj in gc.get_objects():
            #  if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #    del obj

            torch.cuda.empty_cache()
        epoch += 1


if __name__ == "__main__":
    main()
