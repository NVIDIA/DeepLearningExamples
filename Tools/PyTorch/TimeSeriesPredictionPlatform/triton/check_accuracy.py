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
import logging
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "1"

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = Path(__file__).parent.name

from .deployment_toolkit.args import ArgParserGenerator  # noqa: E402  module level import not at top of file
from .deployment_toolkit.core import (  # noqa: E402  module level import not at top of file
    DATALOADER_FN_NAME,
    BaseLoader,
    BaseRunner,
    Model,
    load_from_file,
)
from .deployment_toolkit.extensions import loaders, runners  # noqa: E402  module level import not at top of file
from .model import get_model

LOGGER = logging.getLogger("check_accuracy")

def _get_args():
    parser = argparse.ArgumentParser(
        description="Script for checking accuracy of export and conversion.", allow_abbrev=False
    )
    parser.add_argument("--native-model", help="Path to native model", required=True)
    parser.add_argument("--native-type", help="Native model type", required=True)
    parser.add_argument("--export-model", help="Path to exported model", required=True)
    parser.add_argument("--export-type", help="Exported model type", required=True)
    parser.add_argument("--convert-model", help="Path to converted model", required=True)
    parser.add_argument("--convert-type", help="Converted model type", required=True)
    parser.add_argument("--dataloader", help="Path to python module containing data loader", required=True)
    parser.add_argument("-v", "--verbose", help="Verbose logs", action="store_true", default=False)
    parser.add_argument(
        "--ignore-unknown-parameters",
        help="Ignore unknown parameters (argument often used in CI where set of arguments is constant)",
        action="store_true",
        default=False,
    )

    args, unparsed_args = parser.parse_known_args()

    Loader: BaseLoader = loaders.get(args.native_type)
    ArgParserGenerator(Loader, module_path=args.native_model).update_argparser(parser)
    Runner: BaseRunner = runners.get(args.native_type)
    ArgParserGenerator(Runner).update_argparser(parser)

    Loader: BaseLoader = loaders.get(args.export_type)
    ArgParserGenerator(Loader, module_path=args.export_model).update_argparser(parser)
    Runner: BaseRunner = runners.get(args.export_type)
    ArgParserGenerator(Runner).update_argparser(parser)

    if args.convert_type != 'trt':
        Loader: BaseLoader = loaders.get(args.convert_type)
        ArgParserGenerator(Loader, module_path=args.convert_model).update_argparser(parser)
        Runner: BaseRunner = runners.get(args.convert_type)
        ArgParserGenerator(Runner).update_argparser(parser)


    if args.dataloader is not None:
        get_dataloader_fn = load_from_file(args.dataloader, label="dataloader", target=DATALOADER_FN_NAME)
        ArgParserGenerator(get_dataloader_fn).update_argparser(parser)

    if args.ignore_unknown_parameters:
        args, unknown_args = parser.parse_known_args()
        LOGGER.warning(f"Got additional args {unknown_args}")
    else:
        args = parser.parse_args()
    return args


def main():
    args = _get_args()

    log_level = logging.INFO if not args.verbose else logging.DEBUG
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    LOGGER.info("args:")
    for key, value in vars(args).items():
        LOGGER.info(f"    {key} = {value}")
    
    LOGGER.info(f"Loading {args.native_model}")
    Runner: BaseRunner = runners.get(args.native_type)

    runner_native = ArgParserGenerator(Runner).from_args(args)
    model_native, _ = get_model(model_dir= args.native_model)
    model_native = Model(handle=model_native, precision=None, inputs=None, outputs=['target__0'])



    LOGGER.info(f"Loading {args.export_model}")
    Loader: BaseLoader = loaders.get(args.export_type)
    Runner: BaseRunner = runners.get(args.export_type)

    loader = ArgParserGenerator(Loader, module_path=args.export_model).from_args(args)
    runner_export = ArgParserGenerator(Runner).from_args(args)
    model_export = loader.load(args.export_model)

    if args.convert_type != 'trt':
        LOGGER.info(f"Loading {args.convert_model}")
        Loader: BaseLoader = loaders.get(args.convert_type)
        Runner: BaseRunner = runners.get(args.convert_type)

        loader = ArgParserGenerator(Loader, module_path=args.convert_model).from_args(args)
        runner_convert = ArgParserGenerator(Runner).from_args(args)
        model_convert = loader.load(args.convert_model)

    get_dataloader_fn = load_from_file(args.dataloader, label="dataloader", target=DATALOADER_FN_NAME)
    dataloader_fn = ArgParserGenerator(get_dataloader_fn).from_args(args)
    
    ids, x, y_real = next(dataloader_fn())
    with runner_native.init_inference(model=model_native) as runner_session:
        y_pred_native = runner_session(x)
    del model_native
    del runner_native
    with runner_export.init_inference(model=model_export) as runner_session:
        y_pred_export = runner_session(x)
    del model_export
    del runner_export
    e1 = [np.linalg.norm(y_pred_native[k]-y_pred_export[k]) for k in y_pred_native.keys()]
    assert all([i < 1e-3 for i in e1]), "Error between native and export is {}, limit is 1e-3".format(e1)
    if args.convert_type != 'trt':
        with runner_convert.init_inference(model=model_convert) as runner_session:
            y_pred_convert = runner_session(x)
        e2 = [np.linalg.norm(y_pred_convert[k]-y_pred_export[k]) for k in y_pred_native.keys()]
        assert all([i < 1e-3 for i in e2]), "Error between export and convert is {}, limit is 1e-3".format(e2)

    


if __name__ == "__main__":
    main()
