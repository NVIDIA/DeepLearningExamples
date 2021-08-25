# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import torch
import focal_loss_cuda


class FocalLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cls_output, cls_targets_at_level, num_positives_sum,
                num_real_classes, alpha, gamma, label_smoothing=0.0):
        loss, partial_grad = focal_loss_cuda.forward(cls_output,
                                                     cls_targets_at_level,
                                                     num_positives_sum,
                                                     num_real_classes,
                                                     alpha, gamma,
                                                     label_smoothing)

        ctx.save_for_backward(partial_grad, num_positives_sum)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        partial_grad, num_positives_sum = ctx.saved_tensors

        # The backward kernel is actually in-place to save memory space,
        # partial_grad and grad_input are the same tensor.
        grad_input = focal_loss_cuda.backward(grad_loss, partial_grad,
                                              num_positives_sum)

        return grad_input, None, None, None, None, None, None
