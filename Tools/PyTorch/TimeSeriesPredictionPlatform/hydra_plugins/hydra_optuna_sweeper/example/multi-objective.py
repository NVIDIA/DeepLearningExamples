# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Tuple

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="multi-objective-conf", config_name="config")
def binh_and_korn(cfg: DictConfig) -> Tuple[float, float]:
    x: float = cfg.x
    y: float = cfg.y

    v0 = 4 * x**2 + 4 * y**2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1


if __name__ == "__main__":
    binh_and_korn()
