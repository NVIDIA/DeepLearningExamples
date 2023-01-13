# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Any, List

from collections.abc import Collection
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from omegaconf import II, OmegaConf

@dataclass
class FairseqLambConfig(FairseqDataclass):
    lamb_betas: Any = field(
        default=(0.9, 0.999), metadata={"help": "betas for lamb optimizer"}
    )
    lamb_eps: float = field(
        default=1e-8, metadata={"help": "epsilon for lamb optimizer"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    lr: List[float] = II("optimization.lr")


@register_optimizer("lamb", dataclass=FairseqLambConfig)
class FairseqLAMB(FairseqOptimizer):
    """LAMB optimizer."""

    def __init__(self, cfg: FairseqLambConfig, params):
        super().__init__(cfg)
        try:
            from apex.optimizers import FusedLAMB

            self._optimizer = FusedLAMB(params, **self.optimizer_config)
        except ImportError:
            raise ImportError("Please install apex to use LAMB optimizer")

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0] if isinstance(self.cfg.lr, Collection) else self.cfg.lr,
            "betas": eval(self.cfg.lamb_betas) if isinstance(self.cfg.lamb_betas, str)
                else OmegaConf.to_container(self.cfg.lamb_betas),
            "eps": self.cfg.lamb_eps,
            "weight_decay": self.cfg.weight_decay,
        }

    @property
    def supports_flat_params(self):
        return False
