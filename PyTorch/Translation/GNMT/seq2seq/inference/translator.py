# Copyright (c) 2017 Elad Hoffer
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import subprocess
import time

import torch
import torch.distributed as dist

import seq2seq.data.config as config
import seq2seq.utils as utils
from seq2seq.inference.beam_search import SequenceGenerator


def gather_predictions(preds):
    world_size = utils.get_world_size()
    if world_size > 1:
        all_preds = [preds.new(preds.size(0), preds.size(1)) for i in range(world_size)]
        dist.all_gather(all_preds, preds)
        preds = torch.cat(all_preds)
    return preds


def run_sacrebleu(test_path, reference_path):
    """
    Executes sacrebleu and returns BLEU score.

    :param test_path: path to the test file
    :param reference_path: path to the reference file
    """
    sacrebleu_params = '--score-only -lc --tokenize intl'
    logging.info(f'Running sacrebleu (parameters: {sacrebleu_params})')
    sacrebleu = subprocess.run([f'sacrebleu --input {test_path} \
                                {reference_path} {sacrebleu_params}'],
                               stdout=subprocess.PIPE, shell=True)
    test_bleu = round(float(sacrebleu.stdout.strip()), 2)
    return test_bleu


class Translator:
    """
    Translator runs validation on test dataset, executes inference, optionally
    computes BLEU score using sacrebleu.
    """
    def __init__(self,
                 model,
                 tokenizer,
                 loader=None,
                 beam_size=5,
                 len_norm_factor=0.6,
                 len_norm_const=5.0,
                 cov_penalty_factor=0.1,
                 max_seq_len=50,
                 print_freq=1,
                 reference=None,
                 ):

        self.model = model
        self.tokenizer = tokenizer
        self.loader = loader
        self.insert_target_start = [config.BOS]
        self.insert_src_start = [config.BOS]
        self.insert_src_end = [config.EOS]
        self.batch_first = model.batch_first
        self.beam_size = beam_size
        self.print_freq = print_freq
        self.reference = reference

        self.distributed = (utils.get_world_size() > 1)

        self.generator = SequenceGenerator(
            model=self.model,
            beam_size=beam_size,
            max_seq_len=max_seq_len,
            len_norm_factor=len_norm_factor,
            len_norm_const=len_norm_const,
            cov_penalty_factor=cov_penalty_factor)

    def run(self, calc_bleu=True, epoch=None, iteration=None, eval_path=None,
            summary=False, warmup=0, reference_path=None):
        """
        Runs translation on test dataset.

        :param calc_bleu: if True compares results with reference and computes
            BLEU score
        :param epoch: index of the current epoch
        :param iteration: index of the current iteration
        :param eval_path: path to the file for saving results
        :param summary: if True prints summary
        :param reference_path: path to the file with reference translation
        """
        if reference_path is None:
            reference_path = self.reference

        device = next(self.model.parameters()).device

        test_bleu = torch.tensor([0.], device=device)

        rank = utils.get_rank()
        logging.info(f'Running evaluation on test set')
        self.model.eval()

        output, eval_stats = self.evaluate(self.loader, epoch, iteration,
                                           warmup, summary)
        output = output[:len(self.loader.dataset)]
        output = self.loader.dataset.unsort(output)

        if rank == 0 and eval_path:
            with open(eval_path, 'w') as eval_file:
                lines = [line + '\n' for line in output]
                eval_file.writelines(lines)
            if calc_bleu:
                test_bleu[0] = run_sacrebleu(eval_path, reference_path)
                if summary:
                    logging.info(f'BLEU on test dataset: {test_bleu[0]:.2f}')

        utils.barrier()
        logging.info(f'Finished evaluation on test set')

        if self.distributed:
            dist.broadcast(test_bleu, 0)

        if calc_bleu:
            eval_stats['bleu'] = test_bleu[0].item()
        else:
            eval_stats['bleu'] = None

        return output, eval_stats

    def evaluate(self, loader, epoch=0, iteration=0, warmup=0, summary=False):
        """
        Runs evaluation on test dataset.

        :param epoch: index of the current epoch
        :param iteration: index of the current iteration
        :param summary: if True prints summary
        """
        device = next(self.model.parameters()).device

        batch_time = utils.AverageMeter(warmup, keep=True)
        tot_tok_per_sec = utils.AverageMeter(warmup, keep=True)
        iterations = utils.AverageMeter()
        enc_seq_len = utils.AverageMeter()
        dec_seq_len = utils.AverageMeter()
        stats = {}

        batch_size = loader.batch_size
        global_batch_size = batch_size * utils.get_world_size()
        beam_size = self.beam_size

        bos = [self.insert_target_start] * (batch_size * beam_size)
        bos = torch.tensor(bos, dtype=torch.int64, device=device)
        if self.batch_first:
            bos = bos.view(-1, 1)
        else:
            bos = bos.view(1, -1)

        if beam_size == 1:
            generator = self.generator.greedy_search
        else:
            generator = self.generator.beam_search

        output = []

        for i, (src, indices) in enumerate(loader):
            translate_timer = time.time()
            src, src_length = src
            stats['total_enc_len'] = int(src_length.sum())

            src = src.to(device)
            src_length = src_length.to(device)

            with torch.no_grad():
                context = self.model.encode(src, src_length)
                context = [context, src_length, None]
                preds, lengths, counter = generator(batch_size, bos, context)

            stats['total_dec_len'] = lengths.sum().item()
            stats['iters'] = counter

            indices = torch.tensor(indices).to(preds)
            preds = preds.scatter(0, indices.unsqueeze(1).expand_as(preds), preds)
            preds = gather_predictions(preds).cpu()

            for pred in preds:
                pred = pred.tolist()
                detok = self.tokenizer.detokenize(pred)
                output.append(detok)

            elapsed = time.time() - translate_timer
            batch_time.update(elapsed, batch_size)

            total_tokens = stats['total_dec_len'] + stats['total_enc_len']
            ttps = total_tokens / elapsed
            tot_tok_per_sec.update(ttps, batch_size)

            iterations.update(stats['iters'])
            enc_seq_len.update(stats['total_enc_len'] / batch_size, batch_size)
            dec_seq_len.update(stats['total_dec_len'] / batch_size, batch_size)

            if i % self.print_freq == self.print_freq - 1:
                log = []
                log += f'TEST '
                if epoch is not None:
                    log += f'[{epoch}]'
                if iteration is not None:
                    log += f'[{iteration}]'
                log += f'[{i}/{len(loader)}]\t'
                log += f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                log += f'Decoder iters {iterations.val:.1f} ({iterations.avg:.1f})\t'
                log += f'Tok/s {tot_tok_per_sec.val:.0f} ({tot_tok_per_sec.avg:.0f})'
                log = ''.join(log)
                logging.info(log)

        tot_tok_per_sec.reduce('sum')
        enc_seq_len.reduce('mean')
        dec_seq_len.reduce('mean')
        batch_time.reduce('mean')
        iterations.reduce('sum')

        if summary and utils.get_rank() == 0:
            time_per_sentence = (batch_time.avg / global_batch_size)
            log = []
            log += f'TEST SUMMARY:\n'
            log += f'Lines translated: {len(loader.dataset)}\t'
            log += f'Avg total tokens/s: {tot_tok_per_sec.avg:.0f}\n'
            log += f'Avg time per batch: {batch_time.avg:.3f} s\t'
            log += f'Avg time per sentence: {1000*time_per_sentence:.3f} ms\n'
            log += f'Avg encoder seq len: {enc_seq_len.avg:.2f}\t'
            log += f'Avg decoder seq len: {dec_seq_len.avg:.2f}\t'
            log += f'Total decoder iterations: {int(iterations.sum)}'
            log = ''.join(log)
            logging.info(log)

        eval_stats = {}
        eval_stats['tokens_per_sec'] = tot_tok_per_sec.avg
        eval_stats['runtimes'] = batch_time.vals
        eval_stats['throughputs'] = tot_tok_per_sec.vals

        return output, eval_stats
