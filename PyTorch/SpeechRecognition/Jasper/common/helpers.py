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

import glob
import os
import re
from collections import OrderedDict

import torch
import torch.distributed as dist

from .metrics import word_error_rate


def print_once(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)


def add_ctc_blank(symbols):
    return symbols + ['<BLANK>']


def ctc_decoder_predictions_tensor(tensor, labels):
    """
    Takes output of greedy ctc decoder and performs ctc decoding algorithm to
    remove duplicates and special symbol. Returns prediction
    Args:
        tensor: model output tensor
        label: A list of labels
    Returns:
        prediction
    """
    blank_id = len(labels) - 1
    hypotheses = []
    labels_map = {i: labels[i] for i in range(len(labels))}
    prediction_cpu_tensor = tensor.long().cpu()
    # iterate over batch
    for ind in range(prediction_cpu_tensor.shape[0]):
        prediction = prediction_cpu_tensor[ind].numpy().tolist()
        # CTC decoding procedure
        decoded_prediction = []
        previous = len(labels) - 1 # id of a blank symbol
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
    remove duplicates and special symbol. Prints wer and prediction examples to screen
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


def gather_predictions(predictions_list, labels):
    results = []
    for prediction in predictions_list:
        results += ctc_decoder_predictions_tensor(prediction, labels=labels)
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
        tensors: dictionary with results of an evaluation iteration, e.g. loss, predictions, transcript, and output
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


def process_evaluation_epoch(aggregates, tag=None):
    """
    Processes results from each worker at the end of evaluation and combine to final result
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

    wer, scores, num_words = word_error_rate(hypotheses, references)
    multi_gpu = dist.is_initialized()
    if multi_gpu:
        if eloss is not None:
            eloss /= dist.get_world_size()
            eloss_tensor = torch.tensor(eloss).cuda()
            dist.all_reduce(eloss_tensor)
            eloss = eloss_tensor.item()

        scores_tensor = torch.tensor(scores).cuda()
        dist.all_reduce(scores_tensor)
        scores = scores_tensor.item()
        num_words_tensor = torch.tensor(num_words).cuda()
        dist.all_reduce(num_words_tensor)
        num_words = num_words_tensor.item()
        wer = scores * 1.0 / num_words
    return wer, eloss


def num_weights(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def convert_v1_state_dict(state_dict):
    rules = [
        ('^jasper_encoder.encoder.', 'encoder.layers.'),
        ('^jasper_decoder.decoder_layers.', 'decoder.layers.'),
    ]
    ret = {}
    for k, v in state_dict.items():
        if k.startswith('acoustic_model.'):
            continue
        if k.startswith('audio_preprocessor.'):
            continue
        for pattern, to in rules:
            k = re.sub(pattern, to, k)
        ret[k] = v

    return ret


class Checkpointer(object):

    def __init__(self, save_dir, model_name, keep_milestones=[100, 200, 300]):
        self.save_dir = save_dir
        self.keep_milestones = keep_milestones
        self.model_name = model_name

        tracked = [
            (int(re.search('epoch(\d+)_', f).group(1)), f)
            for f in glob.glob(f'{save_dir}/{self.model_name}_epoch*_checkpoint.pt')]
        tracked = sorted(tracked, key=lambda t: t[0])
        self.tracked = OrderedDict(tracked)

    def save(self, model, ema_model, optimizer, scaler, epoch, step, best_wer,
             is_best=False):
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
        rank = 0
        if dist.is_initialized():
            dist.barrier()
            rank = dist.get_rank()

        if rank != 0:
            return

        # Checkpoint already saved
        if not is_best and epoch in self.tracked:
            return

        unwrap_ddp = lambda model: getattr(model, 'module', model)
        state = {
            'epoch': epoch,
            'step': step,
            'best_wer': best_wer,
            'state_dict': unwrap_ddp(model).state_dict(),
            'ema_state_dict': unwrap_ddp(ema_model).state_dict() if ema_model is not None else None,
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
        }

        if is_best:
            fpath = os.path.join(
                self.save_dir, f"{self.model_name}_best_checkpoint.pt")
        else:
            fpath = os.path.join(
                self.save_dir, f"{self.model_name}_epoch{epoch}_checkpoint.pt")

        print_once(f"Saving {fpath}...")
        torch.save(state, fpath)

        if not is_best:
            # Remove old checkpoints; keep milestones and the last two
            self.tracked[epoch] = fpath
            for epoch in set(list(self.tracked)[:-2]) - set(self.keep_milestones):
                try:
                    os.remove(self.tracked[epoch])
                except:
                    pass
                del self.tracked[epoch]

    def last_checkpoint(self):
        tracked = list(self.tracked.values())

        if len(tracked) >= 1:
            try:
                torch.load(tracked[-1], map_location='cpu')
                return tracked[-1]
            except:
                print_once(f'Last checkpoint {tracked[-1]} appears corrupted.')

        elif len(tracked) >= 2:
            return tracked[-2]
        else:
            return None

    def load(self, fpath, model, ema_model, optimizer, scaler, meta):

        print_once(f'Loading model from {fpath}')
        checkpoint = torch.load(fpath, map_location="cpu")

        unwrap_ddp = lambda model: getattr(model, 'module', model)
        state_dict = convert_v1_state_dict(checkpoint['state_dict'])
        unwrap_ddp(model).load_state_dict(state_dict, strict=True)

        if ema_model is not None:
            if checkpoint.get('ema_state_dict') is not None:
                key = 'ema_state_dict'
            else:
                key = 'state_dict'
                print_once('WARNING: EMA weights not found in the checkpoint.')
                print_once('WARNING: Initializing EMA model with regular params.')
            state_dict = convert_v1_state_dict(checkpoint[key])
            unwrap_ddp(ema_model).load_state_dict(state_dict, strict=True)

        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])

        meta['start_epoch'] = checkpoint.get('epoch')
        meta['best_wer'] = checkpoint.get('best_wer', meta['best_wer'])
