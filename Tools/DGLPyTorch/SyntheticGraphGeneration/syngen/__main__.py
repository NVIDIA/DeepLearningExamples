# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import logging
import sys
import traceback

from syngen.cli import get_parser


logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)
log = logger


def get_args():
    parser = get_parser()

    try:
        args = parser.parse_args()
    except TypeError:
        parser.print_help()
        sys.exit(0)

    return args, sys.argv


def main():
    args, argv = get_args()
    log.info("=========================================")
    log.info("|    Synthetic Graph Generation Tool    |")
    log.info("=========================================")

    try:
        _ = args.action(args)
    except Exception as error:
        print(f"{error}")
        traceback.print_tb(error.__traceback__)
    sys.exit(0)


if __name__ == "__main__":
    main()
