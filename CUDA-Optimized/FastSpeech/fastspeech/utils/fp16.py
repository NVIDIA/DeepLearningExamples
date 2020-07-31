# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn as nn

"""
Revised based on apex/apex/amp/_initialize.py
"""

def _applier(value, fn):
    if isinstance(value, torch.cuda.FloatTensor):
        return fn(value)
    elif isinstance(value, torch.cuda.HalfTensor):
        return fn(value)
    elif isinstance(value, dict):
        return dict({k : _applier(v, fn) for k, v in value.items()})
    elif isinstance(value, tuple):
        return tuple(_applier(v, fn) for v in value)
    else:
        return value

def _cast_module_to_half(module, op_list):

    for op in op_list:
        if isinstance(module, op):
            module.half()
            module.register_forward_pre_hook(lambda module, input: _applier(input, lambda x: x.half()))
            module.register_forward_hook(lambda module, input, output: _applier(output, lambda x: x.float()))
            break
    else:
        for child in module.children():
            _cast_module_to_half(child, op_list)

    return module

def cast_model_to_half(model, op_list=[nn.Linear, nn.Conv1d]):
    model = _cast_module_to_half(model, op_list)
    return model