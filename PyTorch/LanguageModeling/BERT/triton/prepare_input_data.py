#!/usr/bin/env python3

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

import argparse
import json
import logging
import os
from pathlib import Path

from tqdm import tqdm

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = Path(__file__).parent.name

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "0"


from .deployment_toolkit.args import ArgParserGenerator  # noqa: E402  module level import not at top of file
from .deployment_toolkit.core import (  # noqa: E402  module level import not at top of file
    DATALOADER_FN_NAME,
    load_from_file,
)

LOGGER = logging.getLogger("prepare_input_data")


def _parse_and_validate_args():
    parser = argparse.ArgumentParser(description="Dump local inference output of given model", allow_abbrev=False)
    parser.add_argument("--dataloader", help="Path to python file containing dataloader.", required=True)
    parser.add_argument("--input-data-dir", help="Path to dir where output files will be stored", required=True)
    parser.add_argument("-v", "--verbose", help="Verbose logs", action="store_true", default=False)

    args, *_ = parser.parse_known_args()

    get_dataloader_fn = load_from_file(args.dataloader, label="dataloader", target=DATALOADER_FN_NAME)
    ArgParserGenerator(get_dataloader_fn).update_argparser(parser)

    args = parser.parse_args()
    return args


def main():
    args = _parse_and_validate_args()

    log_level = logging.INFO if not args.verbose else logging.DEBUG
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    LOGGER.info("args:")
    for key, value in vars(args).items():
        LOGGER.info(f"    {key} = {value}")

    data = []
    get_dataloader_fn = load_from_file(args.dataloader, label="dataloader", target=DATALOADER_FN_NAME)
    dataloader_fn = ArgParserGenerator(get_dataloader_fn).from_args(args)
    LOGGER.info("Data loader initialized; Creating benchmark data")
    for _, x, _ in tqdm(dataloader_fn(), unit="batch", mininterval=10):
        for input__0, input__1, input__2 in zip(x["input__0"], x["input__1"], x["input__2"]):
            data.append(
                {
                    "input__0": input__0.tolist(),
                    "input__1": input__1.tolist(),
                    "input__2": input__2.tolist(),
                }
            )

    LOGGER.info("Dumping data")
    with open(Path(args.input_data_dir) / "data.json", "w") as fd:
        fd.write(json.dumps({"data": data}))
    LOGGER.info("Dumped")


if __name__ == "__main__":
    main()
