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

import os
import sys
import urllib.request
from zipfile import ZipFile
import torch
from torch.utils.data import DataLoader
NGC_CHECKPOINT_URLS = {}
NGC_CHECKPOINT_URLS["electricity"] = "https://api.ngc.nvidia.com/v2/models/nvidia/dle/tft_base_pyt_ckpt_ds-electricity/versions/22.11.0_amp/zip"
NGC_CHECKPOINT_URLS["traffic"] = "https://api.ngc.nvidia.com/v2/models/nvidia/dle/tft_base_pyt_ckpt_ds-traffic/versions/22.11.0_amp/zip"
def _download_checkpoint(checkpoint, force_reload):
    model_dir = os.path.join(torch.hub._get_torch_home(), 'checkpoints')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    ckpt_file = os.path.join(model_dir, os.path.basename(checkpoint))
    if not os.path.exists(ckpt_file) or force_reload:
        sys.stderr.write('Downloading checkpoint from {}\n'.format(checkpoint))
        urllib.request.urlretrieve(checkpoint, ckpt_file)
    with ZipFile(ckpt_file, "r") as zf:
        zf.extractall(path=model_dir)
    return os.path.join(model_dir, "checkpoint.pt")

def nvidia_tft(pretrained=True, **kwargs):
    from .modeling import TemporalFusionTransformer
    """Constructs a TFT model.
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com
    Args (type[, default value]):
        pretrained (bool, True): If True, returns a pretrained model.
        dataset (str, 'electricity'): loads selected model type electricity or traffic. Defaults to electricity
    """
    ds_type = kwargs.get("dataset", "electricity")
    ckpt = _download_checkpoint(NGC_CHECKPOINT_URLS[ds_type], True)
    state_dict = torch.load(ckpt)
    config = state_dict['config']
    
    model = TemporalFusionTransformer(config)
    if pretrained:
        model.load_state_dict(state_dict['model'])
    model.eval()
    return model

def nvidia_tft_data_utils(**kwargs):

    from .data_utils import TFTDataset
    from .configuration import ElectricityConfig
    class Processing:
        @staticmethod
        def download_data(path):
            if not os.path.exists(os.path.join(path, "raw")):
                os.makedirs(os.path.join(path, "raw"), exist_ok=True)
            dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
            ckpt_file = os.path.join(path, "raw/electricity.zip")
            if not os.path.exists(ckpt_file):
                sys.stderr.write('Downloading checkpoint from {}\n'.format(dataset_url))
                urllib.request.urlretrieve(dataset_url, ckpt_file)
            with ZipFile(ckpt_file, "r") as zf:
                zf.extractall(path=os.path.join(path, "raw/electricity/"))

        @staticmethod
        def preprocess(path):
            config = ElectricityConfig()
            if not os.path.exists(os.path.join(path, "processed")):
                os.makedirs(os.path.join(path, "processed"), exist_ok=True)
            from data_utils import standarize_electricity as standarize
            from data_utils import preprocess
            standarize(os.path.join(path, "raw/electricity"))
            preprocess(os.path.join(path, "raw/electricity/standarized.csv"), os.path.join(path, "processed/electricity_bin/"), config)


        @staticmethod
        def get_batch(path):
            config = ElectricityConfig()
            test_split = TFTDataset(os.path.join(path, "processed/electricity_bin/", "test.csv"), config)
            data_loader = DataLoader(test_split, batch_size=16, num_workers=0)
            for i, batch in enumerate(data_loader):
                if i == 40:
                    break
            return batch

    return Processing()

