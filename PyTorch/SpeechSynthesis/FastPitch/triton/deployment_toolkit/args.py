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
import inspect
import logging
from typing import Any, Callable, Dict, Optional, Union

from .core import GET_ARGPARSER_FN_NAME, load_from_file

LOGGER = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def filter_fn_args(args: Union[dict, argparse.Namespace], fn: Callable) -> dict:
    signature = inspect.signature(fn)
    parameters_names = list(signature.parameters)
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    args = {k: v for k, v in args.items() if k in parameters_names}
    return args


def add_args_for_fn_signature(parser, fn) -> argparse.ArgumentParser:
    parser.conflict_handler = "resolve"
    signature = inspect.signature(fn)
    for parameter in signature.parameters.values():
        if parameter.name in ["self", "args", "kwargs"]:
            continue
        argument_kwargs = {}
        if parameter.annotation != inspect.Parameter.empty:
            if parameter.annotation == bool:
                argument_kwargs["type"] = str2bool
                argument_kwargs["choices"] = [0, 1]
            elif isinstance(parameter.annotation, type(Optional[Any])):
                types = [type_ for type_ in parameter.annotation.__args__ if not isinstance(None, type_)]
                if len(types) != 1:
                    raise RuntimeError(
                        f"Could not prepare argument parser for {parameter.name}: {parameter.annotation} in {fn}"
                    )
                argument_kwargs["type"] = types[0]
            else:
                argument_kwargs["type"] = parameter.annotation

        if parameter.default != inspect.Parameter.empty:
            if parameter.annotation == bool:
                argument_kwargs["default"] = str2bool(parameter.default)
            else:
                argument_kwargs["default"] = parameter.default
        else:
            argument_kwargs["required"] = True
        name = parameter.name.replace("_", "-")
        LOGGER.debug(f"Adding argument {name} with {argument_kwargs}")
        parser.add_argument(f"--{name}", **argument_kwargs)
    return parser


class ArgParserGenerator:
    def __init__(self, cls_or_fn, module_path: Optional[str] = None):
        self._cls_or_fn = cls_or_fn

        self._handle = cls_or_fn if inspect.isfunction(cls_or_fn) else getattr(cls_or_fn, "__init__")
        input_is_python_file = module_path and module_path.endswith(".py")
        self._input_path = module_path if input_is_python_file else None
        self._required_fn_name_for_signature_parsing = getattr(
            cls_or_fn, "required_fn_name_for_signature_parsing", None
        )

    def update_argparser(self, parser):
        name = self._handle.__name__
        group_parser = parser.add_argument_group(name)
        add_args_for_fn_signature(group_parser, fn=self._handle)
        self._update_argparser(group_parser)

    def get_args(self, args: argparse.Namespace):
        filtered_args = filter_fn_args(args, fn=self._handle)

        tmp_parser = argparse.ArgumentParser(allow_abbrev=False)
        self._update_argparser(tmp_parser)
        custom_names = [
            p.dest.replace("-", "_") for p in tmp_parser._actions if not isinstance(p, argparse._HelpAction)
        ]
        custom_params = {n: getattr(args, n) for n in custom_names}
        filtered_args = {**filtered_args, **custom_params}
        return filtered_args

    def from_args(self, args: Union[argparse.Namespace, Dict]):
        args = self.get_args(args)
        LOGGER.info(f"Initializing {self._cls_or_fn.__name__}({args})")
        return self._cls_or_fn(**args)

    def _update_argparser(self, parser):
        label = "argparser_update"
        if self._input_path:
            update_argparser_handle = load_from_file(self._input_path, label=label, target=GET_ARGPARSER_FN_NAME)
            if update_argparser_handle:
                update_argparser_handle(parser)
            elif self._required_fn_name_for_signature_parsing:
                fn_handle = load_from_file(
                    self._input_path, label=label, target=self._required_fn_name_for_signature_parsing
                )
                if fn_handle:
                    add_args_for_fn_signature(parser, fn_handle)
