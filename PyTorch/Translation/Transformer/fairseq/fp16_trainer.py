# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
#-------------------------------------------------------------------------
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

"""
Train a network on multiple GPUs.
"""

import torch
import ctypes

from fairseq import optim, utils
from fairseq.meters import AverageMeter
from fairseq.optim import lr_scheduler
from fairseq.trainer import Trainer

lib = ctypes.cdll.LoadLibrary(None)
lib.THCudaHalfTensor_normall.argtypes=[ctypes.c_void_p, ctypes.c_void_p]#, ctypes.c_float]
lib.THCudaHalfTensor_normall.restype = ctypes.c_float
#function object lib.THCudaHalfTensor_normall can be cached, and fused_norm implemented as a class that has it as one of the attributes, but this will do.

def fused_norm(input):
    if input.type() == 'torch.cuda.HalfTensor':
        return lib.THCudaHalfTensor_normall(torch.cuda._state_cdata, input._cdata, 16384) #yes, 16384 is half 2 if you stare at it long enough
    else:
        return input.norm()


class DynamicLossScaler:

    def __init__(self, init_scale=2.**15, scale_factor=2., scale_window=2000):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self._iter = 0
        self._last_overflow_iter = -1

    def update_scale(self, overflow):
        if overflow:
            self.loss_scale /= self.scale_factor
            self._last_overflow_iter = self._iter
        elif (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
        self._iter += 1

    @staticmethod
    def has_overflow(grad_norm):
        # detect inf and nan
        if grad_norm == float('inf') or grad_norm != grad_norm:
            return True
        return False


class FP16Trainer(Trainer):
    """Modified trainer for FP16.

    We maintain two copies of the model's parameters, both in FP16 and FP32.
    We do forward/backward with FP16 and compute the loss + optimize with FP32.
    """

    def __init__(self, args, task, model, criterion):
        super().__init__(args, task, model, criterion)

        # convert model to FP16 (but keep criterion FP32)
        self.model.half()

        # dynamically scale loss to reduce overflow
        self.scaler = DynamicLossScaler(init_scale=2.**7)
        self.meters['loss_scale'] = AverageMeter()

        self.grad_denom = 1.0

        if self.args.enable_parallel_backward_allred_opt:
            import numpy as np
            self._reduction_stream = torch.cuda.Stream()

            self._flat_grads_parallel = torch.tensor([], dtype=torch.float16).cuda()
            self._grads_info = []
            grads_size = 0
            p_offset = 0
            for p_i, p in enumerate([p for p in self.model.parameters() if p.requires_grad]):
                p_grads_size = np.prod(list(p.size()))
                grads_size += p_grads_size
                # register hooks
                def wrapper(param, param_i, param_grads_size, param_offset):
                    def allreduce_hook(grad):
                        self._do_allreduce(param_i, param_grads_size, param_offset, grad)

                    if param.requires_grad:
                        param.register_hook(allreduce_hook)
                # print(p_i, p.size(), p_grads_size, p_offset)
                self._grads_info.append({"param_grads_size":p_grads_size, "param_offset":p_offset})
                wrapper(p, p_i, p_grads_size, p_offset)
                p_offset += p_grads_size
            self._flat_grads_parallel.resize_(grads_size)
            # print(grads_size, len(self._flat_grads_parallel), self._flat_grads_parallel.dtype, self._flat_grads_parallel.get_device())

            self._allreduce_flush_min_threshold = self.args.parallel_backward_allred_opt_threshold
            print("| parallel all-reduce ENABLED. all-reduce threshold: " + str(self._allreduce_flush_min_threshold))
            self._grads_generated = [False]*len(self._grads_info)
            self._allreduce_processed_idx = len(self._grads_info)-1

            if self.args.enable_parallel_backward_allred_opt_correctness_check:
                self._num_grads_generated = 0
                self._all_grads_generated = False
                self._allreduce_schedule = []

    def _get_flush_bucket(self):
        # print([1 if x else 0 for x in self._grads_generated])
        flush_bucket = []

        size = 0
        allreduce_processed_idx_list = []
        allreduce_processed_end_idx = self._allreduce_processed_idx
        remaining_grads_for_allreduce = self._grads_generated[allreduce_processed_end_idx-len(self._grads_generated)::-1]
        # print([1 if x else 0 for x in remaining_grads_for_allreduce])
        for s in remaining_grads_for_allreduce:
            # print(s,allreduce_processed_end_idx,size)
            if s:
                allreduce_processed_idx_list.append(allreduce_processed_end_idx)
                size += self._grads_info[allreduce_processed_end_idx]["param_grads_size"]
                allreduce_processed_end_idx -= 1
            else:
                break

        # print(size, allreduce_processed_idx_list)
        ignore_threshold = all(self._grads_generated)

        if size >= self._allreduce_flush_min_threshold or ignore_threshold:
            # for i in allreduce_processed_idx_list:
            #     print(i, self._grads_info[i]["param_grads_size"], self._grads_info[i]["param_offset"],size)
            if allreduce_processed_idx_list:
                start = self._grads_info[(allreduce_processed_idx_list[-1])]["param_offset"]
                end = start + size
                # print("->", start, end)
                flush_bucket = [start, end]

            self._allreduce_processed_idx = allreduce_processed_end_idx
            if self._allreduce_processed_idx < 0:
                # reset
                self._grads_generated = [False]*len(self._grads_info)
                self._allreduce_processed_idx = len(self._grads_info)-1

        return flush_bucket

    def _do_allreduce(self, param_i, param_grads_size, param_offset, grad):
        if self._last_step == False:

            # # ----------------------
            # # debugging: do all-reduce in the same stream
            # print(self._last_step, self._grads_total, len(self._backward_grads_schedule), param_i, param_offset, param_grads_size, grad.size(), grad.numel(), grad.dtype)
            # self._flat_grads_parallel[param_offset:param_offset+param_grads_size].copy_(grad.view(-1))
            # self._flat_grads_parallel[param_offset:param_offset+param_grads_size].div_(self.args.distributed_world_size)
            # torch.distributed.all_reduce(self._flat_grads_parallel[param_offset:param_offset+param_grads_size])
            # # ----------------------

            # # ----------------------
            # # option #1: send per-layer gradients
            # torch.div(grad.view(-1), self.args.distributed_world_size, out=self._flat_grads_parallel[param_offset:param_offset+param_grads_size])
            # orig_stream = torch.cuda.current_stream()
            # self._reduction_stream.wait_stream(orig_stream)
            # with torch.cuda.stream(self._reduction_stream):
            #     torch.distributed.all_reduce(self._flat_grads_parallel[param_offset:param_offset+param_grads_size])
            # # ----------------------

            # ----------------------
            # option #2: bucket all-reduce based on threshold
            torch.div(grad.view(-1), self.args.distributed_world_size, out=self._flat_grads_parallel[param_offset:param_offset+param_grads_size])
            self._grads_generated[param_i]=True
            flush_bucket = self._get_flush_bucket()
            if flush_bucket:
                start = flush_bucket[0]
                end = flush_bucket[1]
                # print("->", start, end)

                if self.args.enable_parallel_backward_allred_opt_correctness_check and not self._all_grads_generated:
                    self._allreduce_schedule.append(flush_bucket)

                orig_stream = torch.cuda.current_stream()
                self._reduction_stream.wait_stream(orig_stream)
                with torch.cuda.stream(self._reduction_stream):
                    torch.distributed.all_reduce(self._flat_grads_parallel[start:end])


            if self.args.enable_parallel_backward_allred_opt_correctness_check:
                self._num_grads_generated += 1
                if self._num_grads_generated == len(self._grads_info):
                    self._all_grads_generated = True
            # ----------------------


    def _build_optimizer(self):
        # create FP32 copy of parameters and grads
        params = [p for p in self.model.parameters() if p.requires_grad]
        total_param_size = sum(p.data.numel() for p in params)
        self.fp32_params = params[0].new(0).float().new(total_param_size)
        offset = 0
        for p in params:
            numel = p.data.numel()
            self.fp32_params[offset:offset+numel].copy_(p.data.view(-1))
            offset += numel
        self.fp32_params = torch.nn.Parameter(self.fp32_params)
        #self.fp32_params.grad = self.fp32_params.data.new(total_param_size)

        # create optimizer using the copied FP32 params
        self._optimizer = optim.build_optimizer(self.args, [self.fp32_params])
        self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        extra_state['loss_scale'] = self.scaler.loss_scale
        super().save_checkpoint(filename, extra_state)

    def load_checkpoint(self, filename):
        """Load all training state from a checkpoint file."""
        extra_state = super().load_checkpoint(filename)
        if extra_state is not None and 'loss_scale' in extra_state:
            self.scaler.loss_scale = extra_state['loss_scale']
        return extra_state

    def zero_grad(self):
        # zero both the FP16 and FP32 grads
#        self.model.zero_grad()      # FP16
#        self.optimizer.zero_grad()  # FP32
#        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for p in self.model.parameters():
            p.grad = None





    def _backward(self, loss):
        self.meters['loss_scale'].reset()
        self.meters['loss_scale'].update(self.scaler.loss_scale)
        if loss is not None:
            # dynamically rescale loss to stay in FP16 range
            loss = loss * self.scaler.loss_scale
        return super()._backward(loss)

    def _all_reduce_and_rescale(self, grad_denom, has_grad = True, ooms = 0):
        # undo effect of dynamic loss scaling on gradients
        self.grad_denom = grad_denom * self.scaler.loss_scale

        if self.args.distributed_world_size > 1:
            self.grad_denom /= self.args.distributed_world_size - ooms

            if not self.args.enable_parallel_backward_allred_opt or self._last_step:
                # flatten grads into a single buffer

                self._flat_grads = self._get_flat_grads(out=None, has_grad = has_grad)

                # scale gradients to avoid overflow in all-reduce
                self._flat_grads.div_(self.args.distributed_world_size)

                # all-reduce flat grads
                torch.distributed.all_reduce(self._flat_grads)
            else:
                torch.cuda.current_stream().wait_stream(self._reduction_stream)
                self._flat_grads = self._flat_grads_parallel

                if self.args.enable_parallel_backward_allred_opt_correctness_check:
                    # # ----------------------
                    # # option #1: send per-layer gradients
                    # grads = self._get_grads()
                    # offset = 0
                    # for g in grads:
                    #     numel = g.numel()
                    #     out = grads[0].new(numel).zero_()
                    #     out.copy_(g.view(-1))
                    #     out.div_(self.args.distributed_world_size)
                    #     torch.distributed.all_reduce(out)
                    #     is_parallel_grads_finite = torch.all(torch.isfinite(self._flat_grads_parallel[offset:offset+numel]))
                    #     is_out_finite = torch.all(torch.isfinite(out))
                    #     assert(is_out_finite == is_parallel_grads_finite)
                    #     if not is_out_finite:
                    #         print("| OVERLAP-CHECK: check inf/nan detected. this batch should be skipped")
                    #     else:
                    #         if not torch.all(torch.eq(out, self._flat_grads_parallel[offset:offset+numel])):
                    #             print(out[0:10], self._flat_grads_parallel[offset:offset+10])
                    #             # for i,_ in enumerate(out):
                    #             #     if out[i] != self._flat_grads_parallel[i]:
                    #             #         print(i,out[i],self._flat_grads_parallel[i])
                    #             raise RuntimeError('w-gradients received in parallel vs. end differ')
                    #     offset += numel
                    # # ----------------------

                    # ----------------------
                    # option #2: bucket all-reduce based on threshold
                    # print(self._allreduce_schedule)
                    out = self._get_flat_grads()
                    out.div_(self.args.distributed_world_size)
                    grads_size = 0
                    for s in self._allreduce_schedule:
                        start = s[0]
                        end = s[1]
                        assert(end > start)
                        grads_size += (end - start)
                        torch.distributed.all_reduce(out[start:end])
                        is_parallel_grads_finite = torch.all(torch.isfinite(self._flat_grads_parallel[start:end]))
                        is_out_finite = torch.all(torch.isfinite(out[start:end]))
                        assert(is_out_finite == is_parallel_grads_finite)
                        if not is_out_finite:
                            print("| OVERLAP-CHECK: check inf/nan detected. this batch should be skipped")
                        else:
                            if not torch.all(torch.eq(out[start:end], self._flat_grads_parallel[start:end])):
                                print(start, end, out[start:end], self._flat_grads_parallel[start:end])
                                raise RuntimeError('w-gradients received in parallel vs. end differ')
                    assert(grads_size == len(self._flat_grads_parallel))
                    # ----------------------
        else:
            # flatten grads into a single buffer
            self._flat_grads = self._get_flat_grads(out=None, has_grad = has_grad)

        # rescale and clip grads
        grad_norm = fused_norm(self._flat_grads)

        # detect overflow and adjust loss scale
        overflow = DynamicLossScaler.has_overflow(grad_norm)
        self.scaler.update_scale(overflow)
        if overflow:
            if self.scaler.loss_scale <= self.args.min_loss_scale:
                raise Exception((
                    'Minimum loss scale reached ({}). Your loss is probably exploding. '
                    'Try lowering the learning rate, using gradient clipping or '
                    'increasing the batch size.'
                ).format(self.args.min_loss_scale))
            raise OverflowError('setting loss scale to: ' + str(self.scaler.loss_scale))

        return grad_norm

    def _opt(self):
        # take an optimization step using the FP32 params and grads
        #super()._opt()
        new_params = self._flat_grads.new_empty(self._flat_grads.size())
        self.optimizer.optimizer.step(closure=None, grads=[self._flat_grads], output_params=[new_params], scale=self.grad_denom)
        self.zero_grad()
        self._num_updates += 1

        # update learning rate
        self.lr_scheduler.step_update(self._num_updates)

        # copy FP32 params back into FP16 model
        offset = 0
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            numel = p.data.numel()
            p.data.copy_(new_params[offset:offset+numel].view_as(p.data))
            offset += numel
