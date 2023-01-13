# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import os
import re
from collections import OrderedDict
from pathlib import Path

import amp_C
import numpy as np
import torch
import torch.distributed as dist

from .metrics import word_error_rate
from common.utils import print_once


def to_gpu(batch, fp16=False, bf16=False):
    assert not (fp16 and bf16)
    for k, v in batch['net_input'].items():
        if fp16 and v.dtype is torch.float:
            batch['net_input'][k] = v.cuda(non_blocking=True).half()
        elif bf16 and v.dtype is torch.float:
            batch['net_input'][k] = v.cuda(non_blocking=True).to(dtype=torch.bfloat16)
        else:
            batch['net_input'][k] = v.cuda(non_blocking=True)


def init_multi_tensor_ema(model, ema_model):
    model_weights = list(model.state_dict().values())
    ema_model_weights = list(ema_model.state_dict().values())
    ema_overflow_buf = torch.cuda.IntTensor([0])
    return model_weights, ema_model_weights, ema_overflow_buf


def apply_multi_tensor_ema(decay, model_weights, ema_model_weights,
                           overflow_buf):
    amp_C.multi_tensor_axpby(
        65536, overflow_buf,
        [ema_model_weights, model_weights, ema_model_weights],
        decay, 1-decay, -1)


def apply_ema(model, ema_model, decay, patch_conv_wn=False):
    if not decay:
        return

    if patch_conv_wn:
        torch.nn.utils.remove_weight_norm(model.encoder.pos_conv[0])

    sd = getattr(model, 'module', model).state_dict()

    for k, v in ema_model.state_dict().items():
        v.copy_(decay * v + (1 - decay) * sd[k])

    if patch_conv_wn:
        torch.nn.utils.weight_norm(
            model.encoder.pos_conv[0], name="weight", dim=2)


def add_ctc_blank(symbols):
    return symbols + ['<BLANK>']


def ctc_decoder_predictions_tensor(tensor, labels, blank_id=None):
    """
    Takes output of greedy ctc decoder and performs ctc decoding algorithm to
    remove duplicates and special symbol. Returns prediction
    Args:
        tensor: model output tensor
        label: A list of labels
    Returns:
        prediction
    """
    if blank_id is None:
        blank_id = len(labels) - 1
    hypotheses = []
    labels_map = {i: labels[i] for i in range(len(labels))}
    prediction_cpu_tensor = tensor.long().cpu()
    # iterate over batch
    for prediction in prediction_cpu_tensor:
        prediction = prediction.numpy().tolist()
        # CTC decoding procedure
        decoded_prediction = []
        previous = blank_id
        for p in prediction:
            if (p != previous or previous == blank_id) and p != blank_id:
                decoded_prediction.append(p)
            previous = p
        hypothesis = ''.join([labels_map[c] for c in decoded_prediction])
        hypotheses.append(hypothesis)
    return hypotheses


def greedy_wer(preds, tgt, tgt_lens, labels):
    """
    Takes output of greedy ctc decoder and performs ctc decoding algorithm to
    remove duplicates and special symbol. Prints wer and prediction examples
    to screen.
    Args:
        tensors: A list of 3 tensors (predictions, targets, target_lengths)
        labels: A list of labels

    Returns:
        word error rate
    """
    with torch.no_grad():
        references = gather_transcripts([tgt], [tgt_lens], labels)
        hypotheses = ctc_decoder_predictions_tensor(preds, labels)

    wer, _, _ = word_error_rate(hypotheses, references)
    return wer, hypotheses[0], references[0]


def gather_losses(losses_list):
    return [torch.mean(torch.stack(losses_list))]


def gather_predictions(predictions_list, labels, blank_id=None):
    results = []
    for prediction in predictions_list:
        results += ctc_decoder_predictions_tensor(prediction, labels=labels,
                                                  blank_id=blank_id)
    return results


def gather_transcripts(transcript_list, transcript_len_list, labels):
    results = []
    labels_map = {i: labels[i] for i in range(len(labels))}
    # iterate over workers
    for txt, lens in zip(transcript_list, transcript_len_list):
        for t, l in zip(txt.long().cpu(), lens.long().cpu()):
            t = list(t.numpy())
            results.append(''.join([labels_map[c] for c in t[:l]]))
    return results


def process_evaluation_batch(tensors, global_vars, labels):
    """
    Processes results of an iteration and saves it in global_vars
    Args:
        tensors: dictionary with results of an evaluation iteration,
        e.g., loss, predictions, transcript, and output
        global_vars: dictionary where processes results of iteration are saved
        labels: A list of labels
    """
    for kv, v in tensors.items():
        if kv.startswith('loss'):
            global_vars['EvalLoss'] += gather_losses(v)
        elif kv.startswith('predictions'):
            global_vars['preds'] += gather_predictions(v, labels)
        elif kv.startswith('transcript_length'):
            transcript_len_list = v
        elif kv.startswith('transcript'):
            transcript_list = v
        elif kv.startswith('output'):
            global_vars['logits'] += v

    global_vars['txts'] += gather_transcripts(
        transcript_list, transcript_len_list, labels)


def process_evaluation_epoch(aggregates):
    """
    Processes results from each worker and combine to final result.
    Args:
        aggregates: dictionary containing information of entire evaluation
    Return:
        wer: final word error rate
        loss: final loss
    """
    if 'losses' in aggregates:
        eloss = torch.mean(torch.stack(aggregates['losses'])).item()
    else:
        eloss = None
    hypotheses = aggregates['preds']
    references = aggregates['txts']
    ids = aggregates['ids']

    wer, scores, num_words = word_error_rate(hypotheses, references)
    multi_gpu = dist.is_initialized()
    if multi_gpu:
        if eloss is not None:
            eloss /= dist.get_world_size()
            eloss_tensor = torch.tensor(eloss).cuda()
            dist.all_reduce(eloss_tensor)
            eloss = eloss_tensor.item()

        scores_tensor = torch.tensor(scores).cuda().unsqueeze(-1)
        num_words_tensor = torch.tensor(num_words).cuda().unsqueeze(-1)
        ids_tensor = torch.tensor(ids).cuda().unsqueeze(-1)

        result_tensor = torch.cat(
            [scores_tensor, num_words_tensor, ids_tensor], dim=-1)
        result_tensor_list = [torch.zeros_like(result_tensor)
                              for i in range(dist.get_world_size())]
        dist.all_gather(result_tensor_list, result_tensor)
        if dist.get_rank() == 0:
            agg_results = torch.cat(result_tensor_list, dim=0)
            agg_ids = set()
            agg_score, agg_num_words = 0, 0
            for x in agg_results.cpu().numpy():
                score, num_words, sample_id = x
                if sample_id in agg_ids:
                    continue
                else:
                    agg_ids.add(sample_id)
                    agg_score += score
                    agg_num_words += num_words
            wer = 1.0 * agg_score / agg_num_words
    return wer, eloss


def num_weights(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def load_wrapped_state(model, state_dict, strict=True):
    if model is None:
        return

    unwrap_ddp = lambda model: getattr(model, 'module', model)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    unwrap_ddp(unwrap_ddp(model)).load_state_dict(state_dict, strict=strict)


class Checkpointer:

    def __init__(self, args, model_name):
        self.no_save = args.no_save
        self.save_dir = args.output_dir
        self.keep_milestones = args.keep_milestones
        self.model_name = model_name
        self.output_labels = None  # for supervised training

        pattern = f'{self.model_name}_update*.pt'
        tracked = [(int(re.search('update(\d+)\.pt', str(f)).group(1)), f)
                   for f in Path(args.output_dir).rglob(pattern)]
        self.tracked = OrderedDict(sorted(tracked, key=lambda t: t[0]))

        fpath = (self.last_checkpoint() if args.resume else None) or args.ckpt

        if fpath is not None:
            print_once(f'Loading model from {fpath}')
            self.last_state = torch.load(fpath, map_location="cpu")
        else:
            self.last_state = None

    def maybe_save(self, model, ema_model, optimizer, scaler, train_state,
                   step, epoch, val_losses, val_wer, args):
        """Saves model checkpoint for inference/resuming training.

        Args:
            model: the model, optionally wrapped by DistributedDataParallel
            ema_model: model with averaged weights, can be None
            optimizer: optimizer
            epoch (int): epoch during which the model is saved
            step (int): number of steps since beginning of training
            best_wer (float): lowest recorded WER on the dev set
            is_best (bool, optional): set name of checkpoint to 'best'
                and overwrite the previous one
        """

        if epoch == 0 or args.no_save:
            return
        if args.local_rank != 0 or int(os.environ.get('RANK', 0)) != 0:
            return

        if args.mode == "finetune":
            is_best_ckpt = val_wer[0] < train_state["best_val_wer"]
        elif args.mode == "pretrain":
            is_best_ckpt = val_losses[0] < train_state["best_val_loss"]

        if not is_best_ckpt and epoch % args.save_frequency != 0:
            return

        unwrap_ = lambda model: getattr(model, 'module', model)
        unwrap_ddp = lambda model: unwrap_(unwrap_(model))
        state_dict = lambda m: m.state_dict() if m is not None else None
        type_name = lambda m: None if m is None else type(m).__name__

        val_wer = val_wer or [float("inf")]  # wer absent in pretraining
        train_state.update({
            'optimizer_type': type_name(optimizer),
            'scaler_type': type_name(scaler),
            'step': step,
            'epoch': epoch + 1,  # fairseq compat; restart at the next epoch
            'best_val_wer': min(val_wer[0], train_state["best_val_wer"]),
            'best_val_loss': min(val_losses[0], train_state['best_val_loss']),
        })

        state = {
            'args': args.__dict__,
            'model': state_dict(unwrap_ddp(model)),
            'ema_model': state_dict(unwrap_ddp(ema_model)),
            'optimizer': state_dict(optimizer),
            'scaler': state_dict(scaler),
            'train_state': train_state,
            **({'output_labels': self.output_labels} if self.output_labels else {}),
        }

        if is_best_ckpt:
            fpath = Path(self.save_dir, f"{self.model_name}_best.pt")
            print_once(f"Saving {fpath}...")
            torch.save(state, fpath)

        fpath = Path(self.save_dir, f"{self.model_name}_update{step}.pt")
        print_once(f"Saving {fpath}...")
        torch.save(state, fpath)

        # keep checkpoints with steps closest to milestones
        for_keeps = set()
        if len(self.tracked) > 0:
            tracked = np.array(list(self.tracked.keys()))
            for milestone in self.keep_milestones:
                st = tracked[np.argmin(np.abs(tracked - milestone))]
                for_keeps.add(st)

        # remove old checkpoints; keep milestones and the last two
        self.tracked[step] = fpath
        for epoch in set(list(self.tracked)[:-2]) - for_keeps:
            try:
                os.remove(self.tracked[epoch])
            except:
                pass
            del self.tracked[epoch]

    def maybe_load_state(self, model=None, ema_model=None, optimizer=None,
                         scaler=None, train_state=None, train_loader=None):

        if self.last_state is None:
            return

        if model is not None:
            load_wrapped_state(model, self.last_state['model'])

        if ema_model is not None:
            if checkpoint.get('ema_model', None) is not None:
                load_wrapped_state(ema_model, self.last_state['ema_model'])
            else:
                print_once('WARNING: EMA weights not found in the ckpt.')
                print_once('WARNING: Initializing EMA model with main model.')

                # https://github.com/pytorch/pytorch/issues/28594
                model.remove_conv_wn()
                load_wrapped_state(ema_model, model.state_dict())
                model.apply_conv_wn()

        if optimizer is not None:
            if 'last_optimizer_state' in self.last_state:
                optimizer.load_state_dict(
                    self.last_state['last_optimizer_state'])

            elif 'optimizer' in self.last_state:
                optimizer.load_state_dict(self.last_state['optimizer'])
            else:
                raise ValueError('Optimizer state not found')

        if scaler is not None:
            if 'scaler' in self.last_state:
                scaler.load_state_dict(self.last_state['scaler'])
            elif 'amp' in self.last_state:
                scaler.load_state_dict(self.last_state['amp'])
            else:
                raise ValueError('Scaler state not found')

        if train_state is not None:

            if 'train_state' in self.last_state:
                train_state.update(self.last_state['train_state'])

            if 'extra_state' in self.last_state:
                extra_state = self.last_state['extra_state']
                train_state.update({
                    'epoch': extra_state['train_iterator']['epoch'],
                    'best_val_loss': extra_state['best']
                })

                if 'optimizer_history' in extra_state:
                    train_state['step'] = (extra_state['optimizer_history']
                                                      [-1]['num_updates']),

        if train_loader is not None and 'extra_state' in self.last_state:
            state = self.last_state['extra_state']['train_iterator']
            train_loader.load_state_dict(state)

    def last_checkpoint(self):
        tracked = list(self.tracked.values())
        for fpath in reversed(tracked):
            try:
                torch.load(fpath, map_location='cpu')
                return fpath
            except:
                print_once(f'Checkpoint {fpath} appears corrupted.')

        return None
