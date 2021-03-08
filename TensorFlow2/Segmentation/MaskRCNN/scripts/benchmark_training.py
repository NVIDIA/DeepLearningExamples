# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
""" Scripts that simplifies running training benchmark """

import argparse
import os
import shutil
import subprocess


def main():
    # CLI flags
    parser = argparse.ArgumentParser(description="MaskRCNN train benchmark")
    parser.add_argument('--gpus', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--data_dir', type=str, default='/data')
    parser.add_argument('--model_dir', type=str, default='/tmp/model')
    parser.add_argument('--weights_dir', type=str, default='/model')

    flags = parser.parse_args()
    main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../mask_rcnn_main.py'))

    # build command
    cmd = (
        f'horovodrun -np {flags.gpus} '
        f'python {main_path}'
        f' --mode train'
        f' --model_dir "{flags.model_dir}"'
        f' --checkpoint "{os.path.join(flags.weights_dir, "resnet/resnet-nhwc-2018-02-07/model.ckpt-112603")}"'
        f' --training_file_pattern "{os.path.join(flags.data_dir, "train*.tfrecord")}"'
        f' --init_learning_rate 0.04'
        f' --total_steps 200'
        f' --use_batched_nms'
        f' --noeval_after_training'
        f' --nouse_custom_box_proposals_op'
        f' --xla'
        f' --train_batch_size {flags.batch_size}'
        f' {"--amp" if flags.amp else ""}'
    )

    # print command
    line = '-' * shutil.get_terminal_size()[0]
    print(line, cmd, line, sep='\n')

    # run model
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
