# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from collections import OrderedDict

class DistillHooks():

    """Implements hooks that can extract any intermediate
    output/state in a model's forward pass for distillation.
    """
    def __init__(self, config):

        """
        Intermediate states extracted by `self.child_to_main_hook`
        are saved in `module.distill_states_dict`

        Intermediate nn.Module states extracted by `self.nn_module_hook`
        as listed in `self.nn_module_states` are saved in `self.nn_module_states`
        """

        #list of nn_module_names to register extraction hooks on in `self.register_nn_module_hook`
        self.nn_module_names = config["nn_module_names"]
        #Dictionary to store states extracted from nn module using `self.nn_module_hook`
        self.nn_module_states = {}

    def nn_module_hook(self, name):
        """
        Method to cache output on nn.Module(s)
        """
        def hook(module, input, output):
            self.nn_module_states[name] = output
        return hook

    def register_nn_module_hook(self, module, input):
        """
        Method to register hook on nn.module directly.
        With this method, user can obtain output from
        nn.module without having to explicity add lines
        to cache state in the nn.module itself, or if user
        has no access to the `fwd` method of the module
        Example: models from torchvision

        Typically used in models where model definition
        or the forward pass in inaccessible. Intermediate
        states will be stored in self.nn_module_state
        with key being the name of the module.

        Hook has to be deleted after the very first forward pass
        to avoid registering `nn_module_hook` on modules listed in
        `self.nn_module_names` with every fwd pass

        Example:
                model = MyModel()
                distill_hooks = DistillHooks(config)
                model_pre_hook = model.register_forward_pre_hook(distill_hooks.register_nn_module_hook)
                for idx, batch in enumerate(train_dataloader):
                
                if idx == 1:
                    model_pre_hook.remove()
                
        """

        for name, i in module.named_modules():
            if name in self.nn_module_names:
                i.register_forward_hook(self.nn_module_hook(name))
                print("registered `nn_module_hook` on", name)

    def child_to_main_hook(self, module, input, output):
        """
        Method to recursively fetch all intermediate states cached in children modules and store in parent module
        """
        module.distill_states_dict = OrderedDict()
        for name, i in module.named_modules():
            if hasattr(i, 'distill_state_dict'):
                module.distill_states_dict[name] = i.distill_state_dict


def flatten_states(state_dict, state_name):
    """
    Method to iterate across all intermediate states cached in a dictionary,
    extract a certain state based on `state_name` and append to a list
    """
    extracted_states = []
    for key, value in state_dict.items():
        if state_name in value:
            extracted_states.append(value[state_name])
    return extracted_states
