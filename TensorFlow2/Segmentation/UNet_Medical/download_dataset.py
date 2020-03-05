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

PARSER = argparse.ArgumentParser(description="U-Net medical")

PARSER.add_argument('--data_dir',
                    type=str,
                    default='./data',
                    help="""Directory where to download the dataset""")


def main():
    FLAGS = PARSER.parse_args()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    os.system('wget http://brainiac2.mit.edu/isbi_challenge/sites/default/files/train-volume.tif -P {}'.format(FLAGS.data_dir))
    os.system('wget http://brainiac2.mit.edu/isbi_challenge/sites/default/files/train-labels.tif -P {}'.format(FLAGS.data_dir))
    os.system('wget http://brainiac2.mit.edu/isbi_challenge/sites/default/files/test-volume.tif -P {}'.format(FLAGS.data_dir))

    print("Finished downloading files for U-Net medical to {}".format(FLAGS.data_dir))


if __name__ == '__main__':
    main()
