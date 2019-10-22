# Copyright 2017-2018 The Apache Software Foundation
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -----------------------------------------------------------------------
#
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

import os
import sys
import argparse
import logging
import mxnet as mx
import numpy as np

import data, dali
import fit
import models

def parse_args():
    parser = argparse.ArgumentParser(description="Train classification models on ImageNet",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    models.add_model_args(parser)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    dali.add_dali_args(parser)
    data.add_data_aug_args(parser)
    return parser.parse_args()

def setup_logging(args):
    head = '{asctime}:{levelname}: {message}'
    logging.basicConfig(level=logging.DEBUG, format=head, style='{',
                        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(args.log)])
    logging.info('Start with arguments {}'.format(args))

if __name__ == '__main__':
    args = parse_args()
    setup_logging(args)

    model = models.get_model(**vars(args))
    data_loader = data.get_data_loader(args)

    fit.fit(args, model, data_loader)
