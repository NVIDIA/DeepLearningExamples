# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def sphere(cfg: DictConfig) -> float:
    x: float = cfg.x
    y: float = cfg.y

    if cfg.get("error", False):
        raise RuntimeError("cfg.error is True")

    return x**2 + y**2


if __name__ == "__main__":
    sphere()
