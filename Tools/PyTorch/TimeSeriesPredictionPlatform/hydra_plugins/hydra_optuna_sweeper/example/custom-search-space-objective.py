# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import hydra
from omegaconf import DictConfig
from optuna.trial import Trial


@hydra.main(version_base=None, config_path="custom-search-space", config_name="config")
def multi_dimensional_sphere(cfg: DictConfig) -> float:
    w: float = cfg.w
    x: float = cfg.x
    y: float = cfg.y
    z: float = cfg.z
    return w**2 + x**2 + y**2 + z**2


def configure(cfg: DictConfig, trial: Trial) -> None:
    x_value = trial.params["x"]
    trial.suggest_float(
        "z",
        x_value - cfg.max_z_difference_from_x,
        x_value + cfg.max_z_difference_from_x,
    )
    trial.suggest_float("+w", 0.0, 1.0)  # note +w here, not w as w is a new parameter


if __name__ == "__main__":
    multi_dimensional_sphere()
