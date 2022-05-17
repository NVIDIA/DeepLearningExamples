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

import json
import os
import pathlib
from pathlib import Path
import shutil
import urllib.request
from typing import Any, Callable
from zipfile import ZipFile
from tqdm.auto import tqdm

# Predefined model config files
MODEL_ZOO_KEYS_B1_NGC = {}
MODEL_ZOO_KEYS_B1_NGC["GV100"] = {}
# GPUNet-0: 0.62ms on GV100
MODEL_ZOO_KEYS_B1_NGC["GV100"]["0.65ms"] = "https://api.ngc.nvidia.com/v2/models/nvidia/dle/gpunet_0_pyt_ckpt/versions/21.12.0_amp/zip"
# GPUNet-1: 0.85ms on GV100
MODEL_ZOO_KEYS_B1_NGC["GV100"]["0.85ms"] = "https://api.ngc.nvidia.com/v2/models/nvidia/dle/gpunet_1_pyt_ckpt/versions/21.12.0_amp/zip"
# GPUNet-2: 1.76ms on GV100
MODEL_ZOO_KEYS_B1_NGC["GV100"]["1.75ms"] = "https://api.ngc.nvidia.com/v2/models/nvidia/dle/gpunet_2_pyt_ckpt/versions/21.12.0_amp/zip"
# GPUNet-D1: 1.25ms on GV100
MODEL_ZOO_KEYS_B1_NGC["GV100"]["1.25ms-D"] = "https://api.ngc.nvidia.com/v2/models/nvidia/dle/gpunet_d1_pyt_ckpt/versions/21.12.0_amp/zip"
# GPUNet-D2: 2.25ms on GV100
MODEL_ZOO_KEYS_B1_NGC["GV100"]["2.25ms-D"] = "https://api.ngc.nvidia.com/v2/models/nvidia/dle/gpunet_d2_pyt_ckpt/versions/21.12.0_amp/zip"

# GPUNet-P0: 0.5ms on GV100
MODEL_ZOO_KEYS_B1_NGC["GV100"]["0.5ms-D"] = "https://api.ngc.nvidia.com/v2/models/nvidia/dle/gpunet_p0_pyt_ckpt/versions/21.12.0_amp/zip"
# GPUNet-P1: 0.8ms on GV100
MODEL_ZOO_KEYS_B1_NGC["GV100"]["0.8ms-D"] = "https://api.ngc.nvidia.com/v2/models/nvidia/dle/gpunet_p1_pyt_ckpt/versions/21.12.0_amp/zip"
MODEL_ZOO_BATCH_NGC = {
    "1": MODEL_ZOO_KEYS_B1_NGC,
}

MODEL_ZOO_NAME2TYPE_B1 = {}
MODEL_ZOO_NAME2TYPE_B1["GPUNet-0"] = "0.65ms"
MODEL_ZOO_NAME2TYPE_B1["GPUNet-1"] = "0.85ms"
MODEL_ZOO_NAME2TYPE_B1["GPUNet-2"] = "1.75ms"
MODEL_ZOO_NAME2TYPE_B1["GPUNet-P0"] = "0.5ms-D"
MODEL_ZOO_NAME2TYPE_B1["GPUNet-P1"] = "0.8ms-D"
MODEL_ZOO_NAME2TYPE_B1["GPUNet-D1"] = "1.25ms-D"
MODEL_ZOO_NAME2TYPE_B1["GPUNet-D2"] = "2.25ms-D"

def get_model_list(batch: int = 1):
    """Get a list of models in model zoo."""
    batch = str(batch)
    err_msg = "Batch {} is not yet optimized.".format(batch)
    assert batch in MODEL_ZOO_BATCH_NGC.keys(), err_msg
    return list(MODEL_ZOO_BATCH_NGC[batch].keys())




def get_configs(
    batch: int = 1,
    latency: str = "GPUNet_1ms",
    gpuType: str = "GV100",
    config_root_dir: str = "./configs",
    download: bool = True
):
    """Get file with model config (downloads if necessary)."""
    batch = str(batch)
    errMsg0 = "Batch {} not found, available batches are {}".format(
        batch, list(MODEL_ZOO_BATCH_NGC.keys())
    )
    assert batch in MODEL_ZOO_BATCH_NGC.keys(), errMsg0

    availGPUs = list(MODEL_ZOO_BATCH_NGC[batch].keys())
    errMsg1 = "GPU {} not found, available GPUs are {}".format(gpuType, availGPUs)
    assert gpuType in availGPUs, errMsg1

    errMsg2 = "Latency {} not found, available Latencies are {}".format(
        latency, list(MODEL_ZOO_BATCH_NGC[batch][gpuType])
    )
    assert latency in MODEL_ZOO_BATCH_NGC[batch][gpuType].keys(), errMsg2

    print("testing:", " batch=", batch, " latency=", latency, " gpu=", gpuType)
    
    configPath = config_root_dir + "/batch" + str(batch)
    configPath += "/" + gpuType + "/" + latency + ".json"
    checkpointPath = config_root_dir + "/batch" + str(batch) + "/"
    checkpointPath += gpuType + "/"
    ngcCheckpointPath = Path(checkpointPath)
    checkpointPath += latency + ".pth.tar"
    ngcUrl = MODEL_ZOO_BATCH_NGC[batch][gpuType][latency]
    if download:
        download_checkpoint_ngc(ngcUrl, ngcCheckpointPath)
    with open(configPath) as configFile:
        modelJSON = json.load(configFile)
        configFile.close()

    return modelJSON, checkpointPath



def unzip(checkpoint_path: pathlib.Path, archive_path: pathlib.Path) -> None:
    """
    Unzip acrhive to provided path

    Args:
        checkpoint_path: Path where archive has to be unpacked
        archive_path: Path to archive Archive filename

    Returns:
        None
    """
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path, "r") as zf:
        zf.extractall(path=checkpoint_path)
    archive_path.unlink()


def download_progress(t: Any) -> Callable:
    """
    Progress bar

    Args:
        t: progress

    Returns:
        Callable
    """
    last_b = [0]

    def update_to(b: int = 1, bsize: int = 1, tsize: int = None):
        if tsize not in (None, -1):
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def download_checkpoint_ngc(checkpoint_url: str, checkpoint_path: pathlib.Path) -> None:
    """
    Download checkpoint from given url to provided path
    Args:
        checkpoint_url: Url from which checkpoint has to be downloaded
        checkpoint_path: Path where checkpoint has to be stored

    Returns:
        None
    """
    with tqdm(unit="B") as t:
        reporthook = download_progress(t)
        result = urllib.request.urlretrieve(checkpoint_url, reporthook=reporthook)

    filename = result[0]

    file_path = pathlib.Path(filename)
    assert file_path.is_file() or file_path.is_dir(), "Checkpoint was not downloaded"

    shutil.move(file_path, checkpoint_path.parent / file_path.name)

    archive_path = checkpoint_path.parent / file_path.name
    unzip(checkpoint_path, archive_path)


