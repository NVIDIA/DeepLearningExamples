# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


import logging
from typing import Iterable
import torch

def initialize_module(module: torch.nn.Module, inputs: Iterable[torch.Tensor]) -> None:
    """Use given sample input to initialize the module. 
    Module must implement method called `initialize` which takes list of input tensors
    """
    assert hasattr(module, 'initialize')
    assert len(inputs) == 1, f'{len(inputs)} inputs'
    assert module.initialized.item() == 0, 'initialized'
    module.initialize(*inputs)
    assert module.initialized.item() == 1, 'not initialized'


def initialize(model: torch.nn.Module, single_batch: Iterable[torch.Tensor]) -> None:
    """Initialize all sub-modules in the model given the sample input batch."""
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, 'initialize'):
            logging.info(f'marking {name} for initialization')
            hook = module.register_forward_pre_hook(initialize_module)
            hooks.append(hook)
    _ = model(*single_batch)
    logging.info('all modules initialized, removing hooks')
    for hook in hooks:
        hook.remove()
