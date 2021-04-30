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

import bz2
import os
import urllib.request
import subprocess
import sys
import subprocess

class WikiDownloader:
    def __init__(self, language, save_path):
        self.save_path = save_path + '/wikicorpus_' + language

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.language = language
        # Use a mirror from https://dumps.wikimedia.org/mirrors.html if the below links do not work
        self.download_urls = {
            'en' : 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2',
            'zh' : 'https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2'
        }

        self.checksum_urls = {
            'en': 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-md5sums.txt',
            'zh': 'https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-md5sums.txt'
        }

        self.output_files = {
            'en' : 'wikicorpus_en.xml.bz2',
            'zh' : 'wikicorpus_zh.xml.bz2'
        }

        self.checksum_files = {
            'en': 'enwiki-latest-md5sums.txt',
            'zh': 'zhwiki-latest-md5sums.txt'
        }

    def _compare_hashes(self, data_hash, checksum_file):
        # If the local archive hash is in the checksums file, then the dataset
        # is valid.
        with open(checksum_file, 'r') as checksums:
            if data_hash in checksums.read():
                return True
        return False

    def _valid_download(self, dataset_filename):
        # If the dataset archive already exists, download the latest checksum
        # file from the data repository and compare the archive's hash with the
        # hashes in the checksums file.
        checksum_url = self.checksum_urls[self.language]
        checksum_file = self.checksum_files[self.language]
        output_file = os.path.join(self.save_path, checksum_file)

        print('** Download file already exists, checking if valid')
        cmd = ['wget', checksum_url, f'--output-document={output_file}']
        status = subprocess.run(cmd)
        if status.returncode != 0:
            raise RuntimeError('Unable to download Wiki checksum file')

        cmd = ['md5sum', dataset_filename]
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                universal_newlines=True)
        if result.returncode == 0:
            # The dataset hash is the first sub-string in the output.
            data_hash = result.stdout.strip().split()[0]
            if not self._compare_hashes(data_hash, output_file):
                print('** Invalid archive file. Redownloading')
                return False
            return True
        else:
            raise RuntimeError('Unable to find hash for Wiki dataset')
        return False

    def download(self):
        if self.language in self.download_urls:
            url = self.download_urls[self.language]
            filename = os.path.join(self.save_path,
                                    self.output_files[self.language])

            print('Downloading:', url)
            if os.path.isfile(filename) and self._valid_download(filename):
                print('** Dataset hashes match, skipping download')
            else:
                cmd = ['wget', url, '--output-document={}'.format(filename)]
                print('Running:', cmd)
                status = subprocess.run(cmd)
                if status.returncode != 0:
                    raise RuntimeError('Wiki download not successful')

            # Always unzipping since this is relatively fast and will overwrite
            print('Unzipping:', self.output_files[self.language])
            subprocess.run('bzip2 -dk ' + filename, shell=True, check=True)

        else:
            assert False, 'WikiDownloader not implemented for this language yet.'
