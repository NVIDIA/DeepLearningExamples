from dataclasses import dataclass, asdict, replace
from .common import (
    SequentialSqueezeAndExcitationTRT,
    SequentialSqueezeAndExcitation,
    SqueezeAndExcitation,
    SqueezeAndExcitationTRT,
)
from typing import Optional, Callable
import os
import torch
import argparse
from functools import partial


@dataclass
class ModelArch:
    pass


@dataclass
class ModelParams:
    def parser(self, name):
        return argparse.ArgumentParser(
            description=f"{name} arguments", add_help=False, usage=""
        )


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


def torchhub_docstring(name: str):
    return f"""Constructs a {name} model.
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com
    Args:
        pretrained (bool, True): If True, returns a model pretrained on IMAGENET dataset.
    """

class EntryPoint:
    @staticmethod
    def create(name: str, model: Model):
        ep = EntryPoint(name, model)
        ep.__doc__ = torchhub_docstring(name)
        return ep

    def __init__(self, name: str, model: Model):
        self.name = name
        self.model = model

    def __call__(
        self,
        pretrained=True,
        pretrained_from_file=None,
        state_dict_key_map_fn=None,
        **kwargs,
    ):
        assert not (pretrained and (pretrained_from_file is not None))
        params = replace(self.model.params, **kwargs)

        model = self.model.constructor(arch=self.model.arch, **asdict(params))

        state_dict = None
        if pretrained:
            assert self.model.checkpoint_url is not None
            state_dict = torch.hub.load_state_dict_from_url(
                self.model.checkpoint_url,
                map_location=torch.device("cpu"),
                progress=True,
            )

        if pretrained_from_file is not None:
            if os.path.isfile(pretrained_from_file):
                print(
                    "=> loading pretrained weights from '{}'".format(
                        pretrained_from_file
                    )
                )
                state_dict = torch.load(
                    pretrained_from_file, map_location=torch.device("cpu")
                )
            else:
                print(
                    "=> no pretrained weights found at '{}'".format(
                        pretrained_from_file
                    )
                )

        if state_dict is not None:
            state_dict = {
                k[len("module.") :] if k.startswith("module.") else k: v
                for k, v in state_dict.items()
            }

            def reshape(t, conv):
                if conv:
                    if len(t.shape) == 4:
                        return t
                    else:
                        return t.view(t.shape[0], -1, 1, 1)
                else:
                    if len(t.shape) == 4:
                        return t.view(t.shape[0], t.shape[1])
                    else:
                        return t

            if state_dict_key_map_fn is not None:
                state_dict = {
                    state_dict_key_map_fn(k): v for k, v in state_dict.items()
                }

            if hasattr(model, "ngc_checkpoint_remap"):
                remap_fn = model.ngc_checkpoint_remap(url=self.model.checkpoint_url)
                state_dict = {remap_fn(k): v for k, v in state_dict.items()}

            def _se_layer_uses_conv(m):
                return any(
                    map(
                        partial(isinstance, m),
                        [
                            SqueezeAndExcitationTRT,
                            SequentialSqueezeAndExcitationTRT,
                        ],
                    )
                )

            state_dict = {
                k: reshape(
                    v,
                    conv=_se_layer_uses_conv(
                        dict(model.named_modules())[".".join(k.split(".")[:-2])]
                    ),
                )
                if is_se_weight(k, v)
                else v
                for k, v in state_dict.items()
            }

            model.load_state_dict(state_dict)
        return model

    def parser(self):
        if self.model.params is None:
            return None
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
                help="load pretrained weights from NGC",
            )

        return parser


def is_se_weight(key, value):
    return key.endswith("squeeze.weight") or key.endswith("expand.weight")


def create_entrypoint(m: Model):
    def _ep(**kwargs):
        params = replace(m.params, **kwargs)
        return m.constructor(arch=m.arch, **asdict(params))

    return _ep
