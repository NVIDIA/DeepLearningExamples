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


from argparse import ArgumentParser
from vae.load.preprocessing import load_and_parse_ML_20M
import numpy as np

parser = ArgumentParser(description="Prepare data for VAE training")
parser.add_argument('--data_dir', default='/data', type=str,
                    help='Directory for storing the training data')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed')
args = parser.parse_args()

print('Preprocessing seed: ', args.seed)
np.random.seed(args.seed)

# load dataset
(train_data,
 validation_data_input,
 validation_data_true,
 test_data_input,
 test_data_true) = load_and_parse_ML_20M(args.data_dir)
