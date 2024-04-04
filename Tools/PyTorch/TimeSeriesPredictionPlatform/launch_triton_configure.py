# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

# SPDX-License-Identifier: Apache-2.0

import warnings

import hydra

warnings.filterwarnings("ignore")


@hydra.main(config_path="conf/", config_name="converter_config")
def main(cfg):
    print(cfg)
    cfg.deployment.config.checkpoint=cfg.checkpoint
    hydra.utils.call(cfg, _recursive_=False)


if __name__ == "__main__":
    main()
