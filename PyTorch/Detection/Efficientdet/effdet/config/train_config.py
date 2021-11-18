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

from omegaconf import OmegaConf


def default_detection_train_config():
    # FIXME currently using args for train config, will revisit, perhaps move to Hydra
    h = OmegaConf.create()

    # dataset
    h.skip_crowd_during_training = True

    # augmentation
    h.input_rand_hflip = True
    h.train_scale_min = 0.1
    h.train_scale_max = 2.0
    h.autoaugment_policy = None

    # optimization
    h.momentum = 0.9
    h.learning_rate = 0.08
    h.lr_warmup_init = 0.008
    h.lr_warmup_epoch = 1.0
    h.first_lr_drop_epoch = 200.0
    h.second_lr_drop_epoch = 250.0
    h.clip_gradients_norm = 10.0
    h.num_epochs = 300

    # regularization l2 loss.
    h.weight_decay = 4e-5

    h.lr_decay_method = 'cosine'
    h.moving_average_decay = 0.9998
    h.ckpt_var_scope = None

    return h
