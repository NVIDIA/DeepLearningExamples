# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
from os.path import basename, normpath
import urllib.request
import tarfile
import zipfile
from tqdm import tqdm
import itertools

from glob import glob
import logging

LOG = logging.getLogger("VAE")


def download_movielens(data_dir):
    destination_filepath = os.path.join(data_dir, 'ml-20m/download/ml-20m.zip')
    if not glob(destination_filepath):
        ml_20m_download_url = 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
        download_file(ml_20m_download_url, destination_filepath)

    LOG.info("Extracting")
    extract_file(destination_filepath, to_directory=os.path.join(data_dir, 'ml-20m/extracted'))


def download_file(url, filename):
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    u = urllib.request.urlopen(url)
    with open(filename, 'wb') as f:
        meta = u.info()
        if (meta.get_all("Content-Length")):
            file_size = int(meta.get_all("Content-Length")[0])
            pbar = tqdm(
                total=file_size,
                desc=basename(normpath(filename)),
                unit='B',
                unit_scale=True)

            file_size_dl = 0
            block_sz = 8192
            while True:
                buff = u.read(block_sz)
                if not buff:
                    break
                pbar.update(len(buff))
                file_size_dl += len(buff)
                f.write(buff)
            pbar.close()
        else:
            LOG.warning("No content length information")
            file_size_dl = 0
            block_sz = 8192
            for cyc in itertools.cycle('/–\\|'):
                buff = u.read(block_sz)
                if not buff:
                    break
                print(cyc, end='\r')
                file_size_dl += len(buff)
                f.write(buff)


def extract_file(path, to_directory):
    """
    Extract file
    :param path: Path to compressed file
    :param to_directory: Directory that is going to store extracte files
    """
    if (path.endswith("tar.gz")):
        tar = tarfile.open(path, "r:gz")
        tar.extractall(path=to_directory)
        tar.close()
    elif (path.endswith("tar")):
        tar = tarfile.open(path, "r:")
        tar.extractall(path=to_directory)
        tar.close()
    elif (path.endswith("zip")):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(to_directory)
    else:
        raise Exception(
            "Could not extract {} as no appropriate extractor is found".format(path))
