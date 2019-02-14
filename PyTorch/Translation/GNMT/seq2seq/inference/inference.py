import contextlib
import logging
import os
import subprocess
import time

import torch
import torch.distributed as dist

import seq2seq.data.config as config
from seq2seq.inference.beam_search import SequenceGenerator
from seq2seq.utils import AverageMeter
from seq2seq.utils import barrier
from seq2seq.utils import get_rank
from seq2seq.utils import get_world_size


def gather_predictions(preds):
    world_size = get_world_size()
    if world_size > 1:
        all_preds = [preds.new(preds.size(0), preds.size(1)) for i in range(world_size)]
        dist.all_gather(all_preds, preds)
        preds = torch.cat(all_preds)
    return preds


class Translator:
    """
    Translator runs validation on test dataset, executes inference, optionally
    computes BLEU score using sacrebleu.
    """
    def __init__(self,
                 model,
                 tokenizer,
                 loader,
                 beam_size=5,
                 len_norm_factor=0.6,
                 len_norm_const=5.0,
                 cov_penalty_factor=0.1,
                 max_seq_len=50,
                 cuda=False,
                 print_freq=1,
                 dataset_dir=None,
                 save_path=None,
                 target_bleu=None):

        self.model = model
        self.tokenizer = tokenizer
        self.loader = loader
        self.insert_target_start = [config.BOS]
        self.insert_src_start = [config.BOS]
        self.insert_src_end = [config.EOS]
        self.batch_first = model.batch_first
        self.cuda = cuda
        self.beam_size = beam_size
        self.print_freq = print_freq
        self.dataset_dir = dataset_dir
        self.target_bleu = target_bleu
        self.save_path = save_path

        self.distributed = (get_world_size() > 1)

        self.generator = SequenceGenerator(
            model=self.model,
            beam_size=beam_size,
            max_seq_len=max_seq_len,
            cuda=cuda,
            len_norm_factor=len_norm_factor,
            len_norm_const=len_norm_const,
            cov_penalty_factor=cov_penalty_factor)

    def build_eval_path(self, epoch, iteration):
        """
        Appends index of the current epoch and index of the current iteration
        to the name of the file with results.

        :param epoch: index of the current epoch
        :param iteration: index of the current iteration
        """
        if iteration is not None:
            eval_fname = f'eval_epoch_{epoch}_iter_{iteration}'
        else:
            eval_fname = f'eval_epoch_{epoch}'
        eval_path = os.path.join(self.save_path, eval_fname)
        return eval_path

    def run(self, calc_bleu=True, epoch=None, iteration=None, eval_path=None,
            summary=False, reference_path=None):
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
        if self.cuda:
            test_bleu = torch.cuda.FloatTensor([0])
            break_training = torch.cuda.LongTensor([0])
        else:
            test_bleu = torch.FloatTensor([0])
            break_training = torch.LongTensor([0])

        if eval_path is None:
            eval_path = self.build_eval_path(epoch, iteration)
        detok_eval_path = eval_path + '.detok'

        with contextlib.suppress(FileNotFoundError):
            os.remove(eval_path)
            os.remove(detok_eval_path)

        rank = get_rank()
        logging.info(f'Running evaluation on test set')
        self.model.eval()
        torch.cuda.empty_cache()

        output = self.evaluate(epoch, iteration, summary)
        output = output[:len(self.loader.dataset)]
        output = self.loader.dataset.unsort(output)

        if rank == 0:
            with open(eval_path, 'a') as eval_file:
                eval_file.writelines(output)
            if calc_bleu:
                self.run_detokenizer(eval_path)
                test_bleu[0] = self.run_sacrebleu(detok_eval_path, reference_path)
                if summary:
                    logging.info(f'BLEU on test dataset: {test_bleu[0]:.2f}')

                if self.target_bleu and test_bleu[0] >= self.target_bleu:
                    logging.info(f'Target accuracy reached')
                    break_training[0] = 1

        barrier()
        torch.cuda.empty_cache()
        logging.info(f'Finished evaluation on test set')

        if self.distributed:
            dist.broadcast(break_training, 0)
            dist.broadcast(test_bleu, 0)

        return test_bleu[0].item(), break_training[0].item()

    def evaluate(self, epoch, iteration, summary):
        """
        Runs evaluation on test dataset.

        :param epoch: index of the current epoch
        :param iteration: index of the current iteration
        :param summary: if True prints summary
        """
        batch_time = AverageMeter(False)
        tot_tok_per_sec = AverageMeter(False)
        iterations = AverageMeter(False)
        enc_seq_len = AverageMeter(False)
        dec_seq_len = AverageMeter(False)
        stats = {}

        output = []

        for i, (src, indices) in enumerate(self.loader):
            translate_timer = time.time()
            src, src_length = src

            batch_size = self.loader.batch_size
            global_batch_size = batch_size * get_world_size()
            beam_size = self.beam_size

            bos = [self.insert_target_start] * (batch_size * beam_size)
            bos = torch.LongTensor(bos)
            if self.batch_first:
                bos = bos.view(-1, 1)
            else:
                bos = bos.view(1, -1)

            src_length = torch.LongTensor(src_length)
            stats['total_enc_len'] = int(src_length.sum())

            if self.cuda:
                src = src.cuda()
                src_length = src_length.cuda()
                bos = bos.cuda()

            with torch.no_grad():
                context = self.model.encode(src, src_length)
                context = [context, src_length, None]

                if beam_size == 1:
                    generator = self.generator.greedy_search
                else:
                    generator = self.generator.beam_search
                preds, lengths, counter = generator(batch_size, bos, context)

            stats['total_dec_len'] = lengths.sum().item()
            stats['iters'] = counter

            indices = torch.tensor(indices).to(preds)
            preds = preds.scatter(0, indices.unsqueeze(1).expand_as(preds), preds)

            preds = gather_predictions(preds).cpu()

            for pred in preds:
                pred = pred.tolist()
                detok = self.tokenizer.detokenize(pred)
                output.append(detok + '\n')

            elapsed = time.time() - translate_timer
            batch_time.update(elapsed, batch_size)

            total_tokens = stats['total_dec_len'] + stats['total_enc_len']
            ttps = total_tokens / elapsed
            tot_tok_per_sec.update(ttps, batch_size)

            iterations.update(stats['iters'])
            enc_seq_len.update(stats['total_enc_len'] / batch_size, batch_size)
            dec_seq_len.update(stats['total_dec_len'] / batch_size, batch_size)

            if i % self.print_freq == 0:
                log = []
                log += f'TEST '
                if epoch is not None:
                    log += f'[{epoch}]'
                if iteration is not None:
                    log += f'[{iteration}]'
                log += f'[{i}/{len(self.loader)}]\t'
                log += f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                log += f'Decoder iters {iterations.val:.1f} ({iterations.avg:.1f})\t'
                log += f'Tok/s {tot_tok_per_sec.val:.0f} ({tot_tok_per_sec.avg:.0f})'
                log = ''.join(log)
                logging.info(log)

        tot_tok_per_sec.reduce('sum')
        enc_seq_len.reduce('mean')
        dec_seq_len.reduce('mean')
        batch_time.reduce('mean')
        iterations.reduce('sum')

        if summary and get_rank() == 0:
            time_per_sentence = (batch_time.avg / global_batch_size)
            log = []
            log += f'TEST SUMMARY:\n'
            log += f'Lines translated: {len(self.loader.dataset)}\t'
            log += f'Avg total tokens/s: {tot_tok_per_sec.avg:.0f}\n'
            log += f'Avg time per batch: {batch_time.avg:.3f} s\t'
            log += f'Avg time per sentence: {1000*time_per_sentence:.3f} ms\n'
            log += f'Avg encoder seq len: {enc_seq_len.avg:.2f}\t'
            log += f'Avg decoder seq len: {dec_seq_len.avg:.2f}\t'
            log += f'Total decoder iterations: {int(iterations.sum)}'
            log = ''.join(log)
            logging.info(log)

        return output

    def run_detokenizer(self, eval_path):
        """
        Executes moses detokenizer on eval_path file and saves result to
        eval_path + ".detok" file.

        :param eval_path: path to the tokenized input
        """
        logging.info('Running detokenizer')
        detok_path = os.path.join(self.dataset_dir, config.DETOKENIZER)
        detok_eval_path = eval_path + '.detok'

        with open(detok_eval_path, 'w') as detok_eval_file, \
                open(eval_path, 'r') as eval_file:
            subprocess.run(['perl', f'{detok_path}'], stdin=eval_file,
                           stdout=detok_eval_file, stderr=subprocess.DEVNULL)

    def run_sacrebleu(self, detok_eval_path, reference_path):
        """
        Executes sacrebleu and returns BLEU score.

        :param detok_eval_path: path to the test file
        :param reference_path: path to the reference file
        """
        if reference_path is None:
            reference_path = os.path.join(self.dataset_dir,
                                          config.TGT_TEST_TARGET_FNAME)
        sacrebleu_params = '--score-only -lc --tokenize intl'
        logging.info(f'Running sacrebleu (parameters: {sacrebleu_params})')
        sacrebleu = subprocess.run([f'sacrebleu --input {detok_eval_path} \
                                    {reference_path} {sacrebleu_params}'],
                                   stdout=subprocess.PIPE, shell=True)
        test_bleu = float(sacrebleu.stdout.strip())
        return test_bleu
