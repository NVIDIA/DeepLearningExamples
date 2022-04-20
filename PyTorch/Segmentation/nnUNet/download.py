# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from subprocess import call

from data_preprocessing.configs import task

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--task", type=str, required=True, help="Task to download")
parser.add_argument("--results", type=str, default="/data", help="Directory for data storage")

if __name__ == "__main__":
    args = parser.parse_args()
    tar_file = task[args.task] + ".tar"
    file_path = os.path.join(args.results, tar_file)
    call(f"aws s3 cp s3://msd-for-monai-eu/{tar_file} --no-sign-request {args.results}", shell=True)
    call(f"tar -xf {file_path} -C {args.results}", shell=True)
    call(f"rm -rf {file_path}", shell=True)
