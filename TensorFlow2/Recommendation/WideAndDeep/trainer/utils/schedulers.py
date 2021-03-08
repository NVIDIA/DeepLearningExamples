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

from tensorflow.python.keras import backend as K


def get_schedule(args, steps_per_epoch):
    assert args.deep_warmup_epochs <= args.num_epochs, 'Number of warmup epochs cannot be higher than training epochs'
    base_lr = args.deep_learning_rate
    warmup_steps = args.deep_warmup_epochs * steps_per_epoch
    bound_epoch = args.deep_warmup_epochs + (args.num_epochs - args.deep_warmup_epochs) / 2
    boundaries = [bound_epoch * steps_per_epoch]
    values = [base_lr / 4, base_lr / 8]

    def schedule(optimizer, current_step):
        current_step = max(1, current_step)

        if current_step < warmup_steps:
            warmup_lr = base_lr * current_step / warmup_steps
            K.set_value(optimizer.lr, K.get_value(warmup_lr))
        else:
            for index, bound in enumerate(boundaries):
                if current_step <= bound:
                    K.set_value(optimizer.lr, K.get_value(values[index]))
                    return
            K.set_value(optimizer.lr, K.get_value(values[-1]))
        return

    return schedule
