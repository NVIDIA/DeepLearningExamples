# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#!/usr/bin/env python

import hashlib
import requests
import os
import tarfile
import tqdm

def download_file(url, dest_folder, fname, overwrite=False):
    fpath = os.path.join(dest_folder, fname)
    if os.path.isfile(fpath):
        if overwrite:
            print("Overwriting existing file")
        else:
            print("File exists, skipping download.")
            return

    tmp_fpath = fpath + '.tmp'

    if not os.path.exists(os.path.dirname(tmp_fpath)):
        os.makedirs(os.path.dirname(tmp_fpath))

    r = requests.get(url, stream=True)
    file_size = int(r.headers['Content-Length'])
    chunk_size = 1024 * 1024  # 1MB
    total_chunks = int(file_size / chunk_size)

    with open(tmp_fpath, 'wb') as fp:
        content_iterator = r.iter_content(chunk_size=chunk_size)
        chunks = tqdm.tqdm(content_iterator, total=total_chunks,
                           unit='MB', desc=fpath, leave=True)
        for chunk in chunks:
            fp.write(chunk)

    os.rename(tmp_fpath, fpath)


def md5_checksum(fpath, target_hash):
    file_hash = hashlib.md5()
    with open(fpath, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024*1024), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest() == target_hash


def extract(fpath, dest_folder):
    if fpath.endswith('.tar.gz'):
        mode = 'r:gz'
    elif fpath.endswith('.tar'):
        mode = 'r:'
    else:
        raise IOError('fpath has unknown extention: %s' % fpath)

    with tarfile.open(fpath, mode) as tar:
        members = tar.getmembers()
        for member in tqdm.tqdm(iterable=members, total=len(members), leave=True):
            tar.extract(path=dest_folder, member=member)
