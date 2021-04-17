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

    yaml_args, rest = yaml_cfg_parser.parse_known_args()

    with open(yaml_args.cfg_file, "r") as cfg_file:
        config = yaml.load(cfg_file, Loader=yaml.FullLoader)
    
    cfg = {
        **config["precision"][yaml_args.precision],
        **config["platform"][yaml_args.platform],
        **config["models"][yaml_args.model][yaml_args.platform][yaml_args.precision],
        **config["mode"][yaml_args.mode],
    }

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    add_parser_arguments(parser)
    parser.set_defaults(**cfg)
    args, rest = parser.parse_known_args(rest)

    model_arch = available_models()[args.arch]
    model_args, rest = model_arch.parser().parse_known_args(rest)
    assert len(rest) == 0, f"Unknown args passed: {rest}"

    cudnn.benchmark = True

    main(args, model_args, model_arch)
