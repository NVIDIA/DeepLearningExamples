# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

import sys
import wget

from pathlib import Path


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


class GLUEDownloader:

    def __init__(self, save_path):
        self.save_path = save_path + '/glue'

    def download(self, task_name):
        mkdir(self.save_path)
        if task_name in {'mrpc', 'mnli'}:
            task_name = task_name.upper()
        elif task_name == 'cola':
            task_name = 'CoLA'
        else:  # SST-2
            assert task_name == 'sst-2'
            task_name = 'SST'
        wget.download(
            'https://gist.githubusercontent.com/roclark/9ab385e980c5bdb9e15ecad5963848e0/raw/c9dcc44a6e1336d2411e3333c25bcfd507c39c81/download_glue_data.py',
            out=self.save_path,
        )
        sys.path.append(self.save_path)
        import download_glue_data
        download_glue_data.main(
            ['--data_dir', self.save_path, '--tasks', task_name])
        sys.path.pop()
