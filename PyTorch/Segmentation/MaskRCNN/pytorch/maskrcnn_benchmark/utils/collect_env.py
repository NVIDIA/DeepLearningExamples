# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import PIL

from torch.utils.collect_env import get_pretty_env_info


def get_pil_version():
    return "\n        Pillow ({})".format(PIL.__version__)


def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += get_pil_version()
    return env_str
