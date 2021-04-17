from dataclasses import dataclass, asdict, replace
from typing import Optional, Callable
import os
import torch
import argparse


@dataclass
class ModelArch:
    pass


@dataclass
class ModelParams:
    def parser(self, name):
        return argparse.ArgumentParser(description=f"{name} arguments", add_help=False, usage="")


@dataclass
class OptimizerParams:
    pass


@dataclass
class Model:
    constructor: Callable
    arch: ModelArch
    params: Optional[ModelParams]
    optimizer_params: Optional[OptimizerParams] = None
    checkpoint_url: Optional[str] = None


class EntryPoint:
    def __init__(self, name: str, model: Model):
        self.name = name
        self.model = model

    def __call__(self, pretrained=False, pretrained_from_file=None, **kwargs):
        assert not (pretrained and (pretrained_from_file is not None))
        params = replace(self.model.params, **kwargs)

        model = self.model.constructor(arch=self.model.arch, **asdict(params))

        state_dict = None
        if pretrained:
            assert self.model.checkpoint_url is not None
            state_dict = torch.hub.load_state_dict_from_url(self.model.checkpoint_url, map_location=torch.device('cpu'))

        if pretrained_from_file is not None:
            if os.path.isfile(pretrained_from_file):
                print(
                    "=> loading pretrained weights from '{}'".format(
                        pretrained_from_file
                    )
                )
                state_dict = torch.load(pretrained_from_file, map_location=torch.device('cpu'))
            else:
                print(
                    "=> no pretrained weights found at '{}'".format(
                        pretrained_from_file
                    )
                )
        # Temporary fix to allow NGC checkpoint loading
        if state_dict is not None:
            state_dict = {
                k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()
            }
            state_dict = {
                k: v.view(v.shape[0], -1, 1, 1) if is_linear_se_weight(k, v) else v for k, v in state_dict.items()
            }

            model.load_state_dict(state_dict)
        return model

    def parser(self):
        if self.model.params is None: return None
        parser = self.model.params.parser(self.name)
        parser.add_argument(
            "--pretrained-from-file",
            default=None,
            type=str,
            metavar="PATH",
            help="load weights from local file",
        )
        if self.model.checkpoint_url is not None:
            parser.add_argument(
                "--pretrained",
                default=False,
                action="store_true",
                help="load pretrained weights from NGC"
            )

        return parser


def is_linear_se_weight(key, value):
    return (key.endswith('squeeze.weight') or key.endswith('expand.weight')) and len(value.shape) == 2


def create_entrypoint(m: Model):
    def _ep(**kwargs):
        params = replace(m.params, **kwargs)
        return m.constructor(arch=m.arch, **asdict(params))

    return _ep
