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
# limitations under the License.import torch

import torch
from torch.autograd.variable  import Variable
import fused_relu_dropout_cuda

class FusedReluDropout(torch.autograd.Function) :
    @staticmethod
    def forward(ctx, input, prob) :
        dropout_prob = 1. - prob
        output,mask = fused_relu_dropout_cuda.forward(input, dropout_prob)
        scale = 1./dropout_prob
        scale_save = Variable(torch.tensor([scale]))
        ctx.save_for_backward(mask, scale_save);
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output) :
        mask,scale = ctx.saved_tensors
        grad_input = fused_relu_dropout_cuda.backward(grad_output, mask, scale[0])
        return grad_input, None

fused_relu_dropout = FusedReluDropout.apply
