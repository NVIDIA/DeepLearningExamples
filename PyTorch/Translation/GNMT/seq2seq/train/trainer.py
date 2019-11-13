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
import os
import time
from itertools import cycle

import numpy as np
import torch
import torch.optim
import torch.utils.data
from apex.parallel import DistributedDataParallel
from apex import amp

from seq2seq.train.fp_optimizers import FP16Optimizer
from seq2seq.train.fp_optimizers import FP32Optimizer
from seq2seq.train.fp_optimizers import AMPOptimizer
from seq2seq.train.lr_scheduler import WarmupMultiStepLR
from seq2seq.utils import AverageMeter
from seq2seq.utils import sync_workers


class Seq2SeqTrainer:
    """
    Seq2SeqTrainer
    """
    def __init__(self,
                 model,
                 criterion,
                 opt_config,
                 scheduler_config,
                 print_freq=10,
                 save_freq=1000,
                 grad_clip=float('inf'),
                 save_info={},
                 save_dir='.',
                 train_iterations=0,
                 checkpoint_filename='checkpoint%s.pth',
                 keep_checkpoints=5,
                 math='fp32',
                 loss_scaling={},
                 intra_epoch_eval=0,
                 prealloc_mode='always',
                 iter_size=1,
                 translator=None,
                 verbose=False):
        """
        Constructor for the Seq2SeqTrainer.

        :param model: model to train
        :param criterion: criterion (loss function)
        :param opt_config: dictionary with options for the optimizer
        :param scheduler_config: dictionary with options for the learning rate
            scheduler
        :param print_freq: prints short summary every 'print_freq' iterations
        :param save_freq: saves checkpoint every 'save_freq' iterations
        :param grad_clip: coefficient for gradient clipping
        :param save_info: dict with additional state stored in each checkpoint
        :param save_dir: path to the directiory for checkpoints
        :param train_iterations: total number of training iterations to execute
        :param checkpoint_filename: name of files with checkpoints
        :param keep_checkpoints: max number of checkpoints to keep
        :param math: arithmetic type
        :param loss_scaling: options for dynamic loss scaling
        :param intra_epoch_eval: number of additional eval runs within each
            training epoch
        :param prealloc_mode: controls preallocation,
            choices=['off', 'once', 'always']
        :param iter_size: number of iterations between weight updates
        :param translator: instance of Translator, runs inference on test set
        :param verbose: enables verbose logging
        """
        super(Seq2SeqTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.epoch = 0
        self.save_info = save_info
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_counter = 0
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint_counter = cycle(range(keep_checkpoints))
        self.opt_config = opt_config
        self.device = next(model.parameters()).device
        self.print_freq = print_freq
        self.verbose = verbose
        self.loss = None
        self.translator = translator
        self.intra_epoch_eval = intra_epoch_eval
        self.iter_size = iter_size
        self.prealloc_mode = prealloc_mode
        self.preallocated = False

        self.distributed = torch.distributed.is_initialized()
        self.batch_first = model.batch_first

        params = self.model.parameters()

        if math == 'manual_fp16':
            self.fp_optimizer = FP16Optimizer(
                self.model, grad_clip,
                loss_scale=loss_scaling['init_scale'],
                dls_upscale_interval=loss_scaling['upscale_interval']
                )
            params = self.fp_optimizer.fp32_params
        elif math == 'fp32':
            self.fp_optimizer = FP32Optimizer(self.model, grad_clip)

        opt_name = opt_config.pop('optimizer')
        self.optimizer = torch.optim.__dict__[opt_name](params, **opt_config)
        logging.info(f'Using optimizer: {self.optimizer}')

        self.scheduler = WarmupMultiStepLR(self.optimizer, train_iterations,
                                           **scheduler_config)

        if math == 'fp16':
            self.model, self.optimizer = amp.initialize(
                self.model,
                self.optimizer,
                cast_model_outputs=torch.float16,
                keep_batchnorm_fp32=False,
                opt_level='O2')

            self.fp_optimizer = AMPOptimizer(
                self.model,
                grad_clip,
                loss_scale=loss_scaling['init_scale'],
                dls_upscale_interval=loss_scaling['upscale_interval']
                )

        if self.distributed:
            self.model = DistributedDataParallel(self.model)

    def iterate(self, src, tgt, update=True, training=True):
        """
        Performs one iteration of the training/validation.

        :param src: batch of examples from the source language
        :param tgt: batch of examples from the target language
        :param update: if True: optimizer does update of the weights
        :param training: if True: executes optimizer
        """
        src, src_length = src
        tgt, tgt_length = tgt
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        src_length = src_length.to(self.device)

        num_toks = {}
        num_toks['tgt'] = int(sum(tgt_length - 1))
        num_toks['src'] = int(sum(src_length))

        if self.batch_first:
            output = self.model(src, src_length, tgt[:, :-1])
            tgt_labels = tgt[:, 1:]
            T, B = output.size(1), output.size(0)
        else:
            output = self.model(src, src_length, tgt[:-1])
            tgt_labels = tgt[1:]
            T, B = output.size(0), output.size(1)

        loss = self.criterion(output.view(T * B, -1),
                              tgt_labels.contiguous().view(-1))

        loss_per_batch = loss.item()
        loss /= (B * self.iter_size)

        if training:
            self.fp_optimizer.step(loss, self.optimizer, self.scheduler,
                                   update)

        loss_per_token = loss_per_batch / num_toks['tgt']
        loss_per_sentence = loss_per_batch / B

        return loss_per_token, loss_per_sentence, num_toks

    def feed_data(self, data_loader, training=True):
        """
        Runs training or validation on batches from data_loader.

        :param data_loader: data loader
        :param training: if True runs training else runs validation
        """
        if training:
            assert self.optimizer is not None
            eval_fractions = np.linspace(0, 1, self.intra_epoch_eval+2)[1:-1]
            iters_with_update = len(data_loader) // self.iter_size
            eval_iters = (eval_fractions * iters_with_update).astype(int)
            eval_iters = eval_iters * self.iter_size
            eval_iters = set(eval_iters)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_per_token = AverageMeter()
        losses_per_sentence = AverageMeter()

        tot_tok_time = AverageMeter()
        src_tok_time = AverageMeter()
        tgt_tok_time = AverageMeter()

        batch_size = data_loader.batch_size

        end = time.time()
        for i, (src, tgt) in enumerate(data_loader):
            self.save_counter += 1
            # measure data loading time
            data_time.update(time.time() - end)

            update = False
            if i % self.iter_size == self.iter_size - 1:
                update = True

            # do a train/evaluate iteration
            stats = self.iterate(src, tgt, update, training=training)
            loss_per_token, loss_per_sentence, num_toks = stats

            # measure accuracy and record loss
            losses_per_token.update(loss_per_token, num_toks['tgt'])
            losses_per_sentence.update(loss_per_sentence, batch_size)

            # measure elapsed time
            elapsed = time.time() - end
            batch_time.update(elapsed)
            src_tok_time.update(num_toks['src'] / elapsed)
            tgt_tok_time.update(num_toks['tgt'] / elapsed)
            tot_num_toks = num_toks['tgt'] + num_toks['src']
            tot_tok_time.update(tot_num_toks / elapsed)
            self.loss = losses_per_token.avg

            if training and i in eval_iters:
                eval_fname = f'eval_epoch_{self.epoch}_iter_{i}'
                eval_path = os.path.join(self.save_dir, eval_fname)
                _, eval_stats = self.translator.run(
                    calc_bleu=True,
                    epoch=self.epoch,
                    iteration=i,
                    eval_path=eval_path,
                    )
                test_bleu = eval_stats['bleu']

                log = []
                log += [f'TRAIN [{self.epoch}][{i}/{len(data_loader)}]']
                log += [f'BLEU: {test_bleu:.2f}']
                log = '\t'.join(log)
                logging.info(log)

                self.model.train()
                self.preallocate(data_loader.batch_size,
                                 data_loader.dataset.max_len, training=True)

            if i % self.print_freq == 0:
                phase = 'TRAIN' if training else 'VALIDATION'
                log = []
                log += [f'{phase} [{self.epoch}][{i}/{len(data_loader)}]']
                log += [f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})']
                log += [f'Data {data_time.val:.2e} ({data_time.avg:.2e})']
                log += [f'Tok/s {tot_tok_time.val:.0f} ({tot_tok_time.avg:.0f})']
                if self.verbose:
                    log += [f'Src tok/s {src_tok_time.val:.0f} ({src_tok_time.avg:.0f})']
                    log += [f'Tgt tok/s {tgt_tok_time.val:.0f} ({tgt_tok_time.avg:.0f})']
                    log += [f'Loss/sentence {losses_per_sentence.val:.1f} ({losses_per_sentence.avg:.1f})']
                log += [f'Loss/tok {losses_per_token.val:.4f} ({losses_per_token.avg:.4f})']
                if training:
                    lr = self.optimizer.param_groups[0]['lr']
                    log += [f'LR {lr:.3e}']
                log = '\t'.join(log)
                logging.info(log)

            save_chkpt = (self.save_counter % self.save_freq) == (self.save_freq - 1)
            if training and save_chkpt:
                self.save_counter = 0
                self.save_info['iteration'] = i
                identifier = next(self.checkpoint_counter, -1)
                if identifier != -1:
                    with sync_workers() as rank:
                        if rank == 0:
                            self.save(identifier=identifier)

            end = time.time()

        tot_tok_time.reduce('sum')
        losses_per_token.reduce('mean')

        return losses_per_token.avg, tot_tok_time.avg

    def preallocate(self, batch_size, max_length, training):
        """
        Generates maximum sequence length batch and runs forward and backward
        pass without updating model parameters.

        :param batch_size: batch size for preallocation
        :param max_length: max sequence length for preallocation
        :param training: if True preallocates memory for backward pass
        """
        if self.prealloc_mode == 'always' or (self.prealloc_mode == 'once' and
                                              not self.preallocated):
            logging.info('Executing preallocation')
            torch.cuda.empty_cache()

            src_length = torch.full((batch_size,), max_length,
                                    dtype=torch.int64)
            tgt_length = torch.full((batch_size,), max_length,
                                    dtype=torch.int64)

            if self.batch_first:
                shape = (batch_size, max_length)
            else:
                shape = (max_length, batch_size)

            src = torch.full(shape, 4, dtype=torch.int64)
            tgt = torch.full(shape, 4, dtype=torch.int64)
            src = src, src_length
            tgt = tgt, tgt_length
            self.iterate(src, tgt, update=False, training=training)
            self.model.zero_grad()
            self.preallocated = True

    def optimize(self, data_loader):
        """
        Sets model in training mode, preallocates memory and runs training on
        data provided by data_loader.

        :param data_loader: data loader
        """
        torch.set_grad_enabled(True)
        self.model.train()
        self.preallocate(data_loader.batch_size, data_loader.dataset.max_len,
                         training=True)

        output = self.feed_data(data_loader, training=True)

        self.model.zero_grad()
        return output

    def evaluate(self, data_loader):
        """
        Sets model in eval mode, disables gradients, preallocates memory and
        runs validation on data provided by data_loader.

        :param data_loader: data loader
        """
        torch.set_grad_enabled(False)
        self.model.eval()
        self.preallocate(data_loader.batch_size, data_loader.dataset.max_len,
                         training=False)

        output = self.feed_data(data_loader, training=False)

        self.model.zero_grad()
        return output

    def load(self, filename):
        """
        Loads checkpoint from filename.

        :param filename: path to the checkpoint file
        """
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location={'cuda:0': 'cpu'})
            if self.distributed:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.fp_optimizer.initialize_model(self.model)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.epoch = checkpoint['epoch']
            self.loss = checkpoint['loss']
            logging.info(f'Loaded checkpoint {filename} (epoch {self.epoch})')
        else:
            logging.error(f'Invalid checkpoint: {filename}')

    def save(self, identifier=None, is_best=False, save_all=False):
        """
        Stores checkpoint to a file.

        :param identifier: identifier for periodic checkpoint
        :param is_best: if True stores checkpoint to 'model_best.pth'
        :param save_all: if True stores checkpoint after completed training
            epoch
        """

        def write_checkpoint(state, filename):
            filename = os.path.join(self.save_dir, filename)
            logging.info(f'Saving model to {filename}')
            torch.save(state, filename)

        if self.distributed:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        state = {
            'epoch': self.epoch,
            'state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss': getattr(self, 'loss', None),
        }
        state = dict(list(state.items()) + list(self.save_info.items()))

        if identifier is not None:
            filename = self.checkpoint_filename % identifier
            write_checkpoint(state, filename)

        if is_best:
            filename = 'model_best.pth'
            write_checkpoint(state, filename)

        if save_all:
            filename = f'checkpoint_epoch_{self.epoch:03d}.pth'
            write_checkpoint(state, filename)
