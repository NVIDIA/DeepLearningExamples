#!/usr/bin/env python3

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

import argparse
import hashlib
import os
import requests
import tarfile

from tqdm import tqdm


urls = {
    "dev-clean": ("http://www.openslr.org/resources/12/dev-clean.tar.gz", "42e2234ba48799c1f50f24a7926300a1"),
    "dev-other": ("http://www.openslr.org/resources/12/dev-other.tar.gz", "c8d0bcc9cca99d4f8b62fcc847357931"),
    "test-clean": ("http://www.openslr.org/resources/12/test-clean.tar.gz", "32fa31d27d2e1cad72775fee3f4849a9"),
    "test-other": ("http://www.openslr.org/resources/12/test-other.tar.gz", "fb5a50374b501bb3bac4815ee91d3135"),
    "train-clean-100": ("http://www.openslr.org/resources/12/train-clean-100.tar.gz", "2a93770f6d5c6c964bc36631d331a522"),
    "train-clean-360": ("http://www.openslr.org/resources/12/train-clean-360.tar.gz", "c0e676e450a7ff2f54aeade5171606fa"),
    "train-other-500": ("http://www.openslr.org/resources/12/train-other-500.tar.gz", "d1a0fd59409feb2c614ce4d30c387708"),
}


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
        chunks = tqdm(content_iterator, total=total_chunks,
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
        for member in tqdm(iterable=members, total=len(members), leave=True):
            tar.extract(path=dest_folder, member=member)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Download, verify and extract dataset files')
    parser.add_argument('dest', type=str,
                        help='Download destnation folder.')
    parser.add_argument('-e', type=str, default=None,
                        help='Extraction destnation folder. Defaults to download folder if not provided')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip downloading the files')
    parser.add_argument('--skip_checksum', action='store_true',
                        help='Skip checksum')
    parser.add_argument('--skip_extract', action='store_true',
                        help='Skip extracting files')
    parser.add_argument('--subsets', type=str, nargs="+", choices=list(urls.keys()),
                        default=list(urls.keys()), help='Subsets to download')
    args = parser.parse_args()
    args.e = args.e or args.dest

    print("\nNOTE: Depending on the selected subsets and connection bandwith "
          "this process might take a few hours.\n")

    for subset in args.subsets:
        url, md5 = urls[subset]

        if not args.skip_download:
            fname = url.split('/')[-1]
            print("Downloading %s:" % fname)
            download_file(url=url, dest_folder=args.dest, fname=fname)
        else:
            print("Skipping file download")

        if not args.skip_checksum:
            fname = url.split('/')[-1]
            fpath = os.path.join(args.dest, fname)
            print("Verifing %s: " % fname, end='')
            ret = md5_checksum(fpath=fpath, target_hash=md5)
            print("Passed" if ret else "Failed")
        else:
            print("Skipping checksum")

        if not args.skip_extract:
            fname = url.split('/')[-1]
            fpath = os.path.join(args.dest, fname)
            print("Decompressing %s:" % fpath)
            extract(fpath=fpath, dest_folder=args.e)
        else:
            print("Skipping file extraction")
