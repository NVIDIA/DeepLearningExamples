import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any
import yaml

from main import main, add_parser_arguments, available_models
import torch.backends.cudnn as cudnn

import argparse


def get_config_path():
    return Path(os.path.dirname(os.path.abspath(__file__))) / "configs.yml"


if __name__ == "__main__":
    yaml_cfg_parser = argparse.ArgumentParser(add_help=False)
    yaml_cfg_parser.add_argument(
        "--cfg_file",
        default=get_config_path(),
        type=str,
        help="path to yaml config file",
    )
    yaml_cfg_parser.add_argument("--model", default=None, type=str, required=True)
    yaml_cfg_parser.add_argument("--mode", default=None, type=str, required=True)
    yaml_cfg_parser.add_argument("--precision", default=None, type=str, required=True)
    yaml_cfg_parser.add_argument("--platform", default=None, type=str, required=True)
    yaml_cfg_parser.add_argument("--batch_size", default=256, type=int, required=False)
    yaml_cfg_parser.add_argument("--learning_rate", default=0.1, type=float, required=False)
    yaml_cfg_parser.add_argument("--memory_format", default="", type=str, required=False)

    yaml_args, rest = yaml_cfg_parser.parse_known_args()

    with open(yaml_args.cfg_file, "r") as cfg_file:
        config = yaml.load(cfg_file, Loader=yaml.FullLoader)
    
    # arch: se-resnext101-32x4d
    # batch_size: 112
    # memory_format: nhwc

    if yaml_args.model not in config["models"] or config["models"][yaml_args.model] is None:
        config["models"][yaml_args.model] = dict()

    if yaml_args.platform not in config["models"][yaml_args.model]:
        config["models"][yaml_args.model][yaml_args.platform] = dict()

    config["models"][yaml_args.model][yaml_args.platform]['batch_size'] = yaml_args.batch_size

    if yaml_args.precision not in config["models"][yaml_args.model][yaml_args.platform]:
        config["models"][yaml_args.model][yaml_args.platform][yaml_args.precision] = dict()
        config["models"][yaml_args.model][yaml_args.platform][yaml_args.precision]['batch_size'] = yaml_args.batch_size
        config["models"][yaml_args.model][yaml_args.platform][yaml_args.precision]['arch'] = yaml_args.model
        config["models"][yaml_args.model][yaml_args.platform][yaml_args.precision]['lr'] = yaml_args.learning_rate

    if len(yaml_args.memory_format):
        config["models"][yaml_args.model][yaml_args.platform][yaml_args.precision]['memory_format'] = yaml_args.memory_format    

    cfg = {
        **config["precision"][yaml_args.precision],
        **config["platform"][yaml_args.platform],
        **config["models"][yaml_args.model][yaml_args.platform][yaml_args.precision],
        **config["mode"][yaml_args.mode],
    }

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    add_parser_arguments(parser)
    parser.set_defaults(**cfg)
    parser.add_argument("--raport_file", default=None, type=str, required=True)

    args, rest = parser.parse_known_args(rest)

    model_arch = available_models()[args.arch]
    model_args, rest = model_arch.parser().parse_known_args(rest)
    assert len(rest) == 0, f"Unknown args passed: {rest}"

    cudnn.benchmark = True

    main(args, model_args, model_arch)
