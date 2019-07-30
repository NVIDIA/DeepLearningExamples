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

import os
import argparse
import pandas as pd

from download_utils import download_file, md5_checksum, extract

parser = argparse.ArgumentParser(description='Download, verify and extract dataset files')
parser.add_argument('csv', type=str,
                    help='CSV file with urls and checksums to download.')
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
args = parser.parse_args()
args.e = args.e or args.dest


df = pd.read_csv(args.csv, delimiter=',')


if not args.skip_download:
    for url in df.url:
        fname = url.split('/')[-1]
        print("Downloading %s:" % fname)
        download_file(url=url, dest_folder=args.dest, fname=fname)
else:
    print("Skipping file download")


if not args.skip_checksum:
    for index, row in df.iterrows():
        url = row['url']
        md5 = row['md5']
        fname = url.split('/')[-1]
        fpath = os.path.join(args.dest, fname)
        print("Verifing %s: " % fname, end='')
        ret = md5_checksum(fpath=fpath, target_hash=md5)
        print("Passed" if ret else "Failed")
else:
    print("Skipping checksum")


if not args.skip_extract:
    for url in df.url:
        fname = url.split('/')[-1]
        fpath = os.path.join(args.dest, fname)
        print("Decompressing %s:" % fpath)
        extract(fpath=fpath, dest_folder=args.e)
else:
    print("Skipping file extraction")
