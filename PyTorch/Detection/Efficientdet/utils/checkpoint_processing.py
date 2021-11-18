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

import argparse
import torch
from collections import OrderedDict


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--checkpoint_path', default='/checkpoints/model_best.pth.tar', help='path to checkpoint')
parser.add_argument('--state_dict_path', default='/checkpoints/Effdet_B0.pth', help='path to save processed checkpoint state_dict to.')

args = parser.parse_args()
ckpt = torch.load(args.checkpoint_path)

print("Checkpoint keys {}".format([k for k in ckpt.keys()]))

if 'state_dict_ema' in ckpt:
    print("... state_dict found in ckpt")
    state_dict = ckpt['state_dict_ema']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # strip `module.` prefix
        if k.startswith('module'):
            name = k[7:]
        elif k.startswith('model'):
            name = k[6:]
        else:
            name = k
        new_state_dict[name] = v
    print("... state_dict saving")
    torch.save(new_state_dict, args.state_dict_path)
print("...End process")
