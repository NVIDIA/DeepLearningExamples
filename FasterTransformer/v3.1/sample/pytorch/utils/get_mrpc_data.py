# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import shutil
import argparse
import tempfile
import urllib.request
import zipfile

MRPC_TRAIN = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt'
MRPC_TEST = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt'

def format_mrpc(mrpc_dir):
    print("Processing MRPC...")
    mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
    mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
    urllib.request.urlretrieve(MRPC_TRAIN, mrpc_train_file)
    urllib.request.urlretrieve(MRPC_TEST, mrpc_test_file)
    assert os.path.isfile(mrpc_train_file), "Train data not found at %s" % mrpc_train_file
    assert os.path.isfile(mrpc_test_file), "Test data not found at %s" % mrpc_test_file
    urllib.request.urlretrieve('https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc', os.path.join(mrpc_dir, "dev_ids.tsv"))

    dev_ids = []
    with open(os.path.join(mrpc_dir, "dev_ids.tsv"), encoding="utf8") as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split('\t'))

    with open(mrpc_train_file, encoding="utf8") as data_fh, \
         open(os.path.join(mrpc_dir, "train.tsv"), 'w', encoding="utf8") as train_fh, \
         open(os.path.join(mrpc_dir, "dev.tsv"), 'w', encoding="utf8") as dev_fh:
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split('\t')
            if [id1, id2] in dev_ids:
                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            else:
                train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

    with open(mrpc_test_file, encoding="utf8") as data_fh, \
            open(os.path.join(mrpc_dir, "test.tsv"), 'w', encoding="utf8") as test_fh:
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split('\t')
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))
    print("\tCompleted!")

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='directory to save data to', type=str, default='glue_data')
    args = parser.parse_args(arguments)

    format_mrpc(args.data_dir)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))