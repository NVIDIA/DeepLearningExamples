# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class MultiprocessingLauncherConf:
    _target_: str = "hydra_plugins.hydra_multiprocessing_launcher.multiprocessing_launcher.MultiprocessingLauncher"

    # maximum number of concurrently running jobs. if None, all CPUs are used
    n_jobs: Optional[int] = None


ConfigStore.instance().store(
    group="hydra/launcher",
    name="multiprocessing",
    node=MultiprocessingLauncherConf,
    provider="multiprocessing_launcher",
)
