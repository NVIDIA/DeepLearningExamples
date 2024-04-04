# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class LauncherConfig:
    _target_: str = (
        "hydra_plugins.hydra_torchrun_launcher.distributed_launcher.TorchDistributedLauncher"
    )
    min_nodes: int = 1
    max_nodes: int = 1
    nproc_per_node: int = 8
    rdzv_id: str = 'none'
    role: str = 'default'
    rdzv_endpoint: str = '127.0.0.1:29500'
    rdzv_backend: str = 'static'
    rdzv_timeout: int = -1
    max_restarts: int = 0
    monitor_interval: int = 5
    log_dir = None
    redirects: str = '0'
    tee: str = '0'


ConfigStore.instance().store(
    group="hydra/launcher", name="torchrun", node=LauncherConfig
)
