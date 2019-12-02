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
import os
import tarfile

from google_drive_downloader import GoogleDriveDownloader as gdd

PARSER = argparse.ArgumentParser(description="V-Net medical")

PARSER.add_argument('--data_dir',
                    type=str,
                    default='./data',
                    help="""Directory where to download the dataset""")

PARSER.add_argument('--dataset',
                    type=str,
                    default='hippocampus',
                    help="""Dataset to download""")


def main():
    FLAGS = PARSER.parse_args()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    filename = ''

    if FLAGS.dataset == 'hippocampus':
        filename = 'Task04_Hippocampus.tar'
        gdd.download_file_from_google_drive(file_id='1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C',
                                            dest_path=os.path.join(FLAGS.data_dir, filename),
                                            unzip=False)

    print('Unpacking...')

    tf = tarfile.open(os.path.join(FLAGS.data_dir, filename))
    tf.extractall(path=FLAGS.data_dir)

    print('Cleaning up...')

    os.remove(os.path.join(FLAGS.data_dir, filename))

    print("Finished downloading files for V-Net medical to {}".format(FLAGS.data_dir))


if __name__ == '__main__':
    main()
