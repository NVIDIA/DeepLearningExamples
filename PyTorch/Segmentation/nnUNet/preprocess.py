# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from data_preprocessing.preprocessor import Preprocessor
from utils.utils import get_task_code

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--data", type=str, default="/data", help="Path to data directory")
parser.add_argument("--results", type=str, default="/data", help="Path for saving results directory")
parser.add_argument(
    "--exec_mode",
    type=str,
    default="training",
    choices=["training", "val", "test"],
    help="Mode for data preprocessing",
)
parser.add_argument("--dilation", action="store_true", help="Perform morphological label dilation")
parser.add_argument("--task", type=str, help="Number of task to be run. MSD uses numbers 01-10")
parser.add_argument("--dim", type=int, default=3, choices=[2, 3], help="Data dimension to prepare")
parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs for data preprocessing")


if __name__ == "__main__":
    args = parser.parse_args()
    start = time.time()
    Preprocessor(args).run()
    task_code = get_task_code(args)
    path = os.path.join(args.data, task_code)
    if args.exec_mode == "test":
        path = os.path.join(path, "test")
    end = time.time()
    print(f"Preprocessing time: {(end - start):.2f}")
