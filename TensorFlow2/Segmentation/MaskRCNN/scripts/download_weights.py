# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
""" Scripts that downloads pretrained weights for ResNet50 backbone. """

import argparse
from os import path
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

RESNET_NAME = 'NVIDIA ResNet50 v1.5'
RESNET_URL = 'https://api.ngc.nvidia.com/v2/models/nvidia/rn50_tf_amp_ckpt/versions/20.06.0/zip'
RESNET_DIR = 'rn50_tf_amp_ckpt_v20.06.0'


if __name__ == '__main__':

    # cli
    parser = argparse.ArgumentParser(
        description='NVIDIA MaskRCNN TF2 backbone checkpoint download and conversion'
    )
    parser.add_argument('--save_dir', type=str, default='/weights',
                        help='Directory to which the checkpoint will be saved')
    parser.add_argument('--download_url', type=str, default=RESNET_URL,
                        help='Override checkpoint download url')
    params = parser.parse_args()

    resnet_dir = path.join(params.save_dir, RESNET_DIR)

    # download and extract
    print(f'Downloading and extracting {RESNET_NAME} checkpoint from {params.download_url}')

    with urlopen(params.download_url) as zip_stream:
        with ZipFile(BytesIO(zip_stream.read())) as zip_file:
            zip_file.extractall(resnet_dir)

    print(f'{RESNET_NAME} checkpoint was extracted to {resnet_dir}')
