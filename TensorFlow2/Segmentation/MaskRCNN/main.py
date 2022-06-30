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
"""Training script for Mask-RCNN."""
import logging
import os
from argparse import Namespace

from mrcnn_tf2.runtime.run import run_training, run_inference, run_evaluation
from mrcnn_tf2.utils.dllogger import LoggingBackend

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'

import dllogger

from mrcnn_tf2.arguments import PARSER
from mrcnn_tf2.config import CONFIG
from mrcnn_tf2.dataset import Dataset


def main():

    # setup params
    arguments = PARSER.parse_args()
    params = Namespace(**{**vars(CONFIG), **vars(arguments)})

    # setup logging
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.DEBUG if params.verbose else logging.INFO,
        format='{asctime} {levelname:.1} {name:15} {message}',
        style='{'
    )

    # remove custom tf handler that logs to stderr
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').handlers.clear()

    # setup dllogger
    dllogger.init(backends=[
        dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE, filename=params.log_file, append=True),
        LoggingBackend(verbosity=dllogger.Verbosity.VERBOSE)
    ])
    dllogger.log(step='PARAMETER', data=vars(params))

    # setup dataset
    dataset = Dataset(params)

    if params.mode == 'train':
        run_training(dataset, params)
    if params.mode == 'eval':
        run_evaluation(dataset, params)
    if params.mode == 'infer':
        run_inference(dataset, params)


if __name__ == '__main__':
    main()
