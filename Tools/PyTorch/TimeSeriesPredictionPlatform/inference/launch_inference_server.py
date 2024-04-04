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
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple
import dllogger
import shutil
import hydra
from triton.dataloader import get_dataloader_fn
def run_server_launch(config):
    cfg = config
    # export model
    output_path = os.path.join(cfg.checkpoint, "deployment")
    tspp_main_dir = os.path.sep + os.path.join(*(os.getcwd().split(os.path.sep)[:-3]))

    # get the actual model name
    if not os.path.isdir(os.path.join(output_path, "navigator_workspace")) or not os.path.isdir(
        os.path.join(output_path, "navigator_workspace/model-store")
    ):
        if os.path.isdir(os.path.join(output_path, "navigator_workspace/final-model-store")):
            shutil.copytree(os.path.join(output_path, "navigator_workspace/final-model-store"), os.path.join(output_path, "navigator_workspace/model-store"))
        else:
            assert (
                False
            ), "This checkpoint directory is not configured correctly, there should be a dir/deployment/navigator_workspace/model-store/ directory"
    files_in_store = list(os.listdir(os.path.join(output_path, "navigator_workspace/model-store")))
    if len(files_in_store) < 1:
        assert False, "There needs to be exactly 1 model in the model-store directory"
    model_name = cfg.get("model_name") if cfg.get("model_name", None) else files_in_store[0]
    # deploy
    subprocess.run(["bash", "inference/deploy.sh", output_path, str(cfg.gpu)], cwd=tspp_main_dir, check=True)
