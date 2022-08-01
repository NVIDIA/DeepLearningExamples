# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

def get_config(config_name, config_path, override_list=None, return_hydra_config=False):
    GlobalHydra.instance().clear()
    initialize(config_path=config_path)
    cfg = compose(config_name, return_hydra_config=return_hydra_config, overrides=override_list)
    if return_hydra_config:
        HydraConfig().cfg = cfg
        OmegaConf.resolve(cfg)
    return cfg
