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

from GooglePretrainedWeightDownloader import GooglePretrainedWeightDownloader
from NVIDIAPretrainedWeightDownloader import NVIDIAPretrainedWeightDownloader
from WikiDownloader import WikiDownloader
from BooksDownloader import BooksDownloader
from GLUEDownloader import GLUEDownloader
from SquadDownloader import SquadDownloader


class Downloader:

    def __init__(self, dataset_name, save_path):
        self.dataset_name = dataset_name
        self.save_path = save_path

    def download(self):
        if self.dataset_name == 'bookscorpus':
            self.download_bookscorpus()

        elif self.dataset_name == 'wikicorpus_en':
            self.download_wikicorpus('en')

        elif self.dataset_name == 'wikicorpus_zh':
            self.download_wikicorpus('zh')

        elif self.dataset_name == 'google_pretrained_weights':
            self.download_google_pretrained_weights()

        elif self.dataset_name == 'nvidia_pretrained_weights':
            self.download_nvidia_pretrained_weights()

        elif self.dataset_name in {'mrpc', 'sst-2'}:
            self.download_glue(self.dataset_name)

        elif self.dataset_name == 'squad':
            self.download_squad()

        elif self.dataset_name == 'all':
            self.download_bookscorpus()
            self.download_wikicorpus('en')
            self.download_wikicorpus('zh')
            self.download_google_pretrained_weights()
            self.download_nvidia_pretrained_weights()
            self.download_glue('mrpc')
            self.download_glue('sst-2')
            self.download_squad()

        else:
            print(self.dataset_name)
            assert False, 'Unknown dataset_name provided to downloader'

    def download_bookscorpus(self):
        downloader = BooksDownloader(self.save_path)
        downloader.download()

    def download_wikicorpus(self, language):
        downloader = WikiDownloader(language, self.save_path)
        downloader.download()

    def download_google_pretrained_weights(self):
        downloader = GooglePretrainedWeightDownloader(self.save_path)
        downloader.download()

    def download_nvidia_pretrained_weights(self):
        downloader = NVIDIAPretrainedWeightDownloader(self.save_path)
        downloader.download()

    def download_glue(self, task_name):
        downloader = GLUEDownloader(self.save_path)
        downloader.download(task_name)

    def download_squad(self):
        downloader = SquadDownloader(self.save_path)
        downloader.download()
