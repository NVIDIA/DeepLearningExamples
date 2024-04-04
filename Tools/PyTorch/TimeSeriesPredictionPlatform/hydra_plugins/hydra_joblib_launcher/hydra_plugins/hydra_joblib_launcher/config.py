# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class JobLibLauncherConf:
    _target_: str = "hydra_plugins.hydra_joblib_launcher.joblib_launcher.JoblibLauncher"

    # maximum number of concurrently running jobs. if -1, all CPUs are used
    n_jobs: int = -1

    # allows to hard-code backend, otherwise inferred based on prefer and require
    backend: Optional[str] = None

    # processes or threads, soft hint to choose backend
    prefer: str = "processes"

    # null or sharedmem, sharedmem will select thread-based backend
    require: Optional[str] = None

    # if greater than zero, prints progress messages
    verbose: int = 0

    # timeout limit for each task. Unit dependent on backend implementation; miliseconds for loky.
    timeout: Optional[float] = None

    # number of batches to be pre-dispatched
    pre_dispatch: str = "2*n_jobs"

    # number of atomic tasks to dispatch at once to each worker
    batch_size: str = "auto"

    # path used for memmapping large arrays for sharing memory with workers
    temp_folder: Optional[str] = None

    # thresholds size of arrays that triggers automated memmapping
    max_nbytes: Optional[str] = None

    # memmapping mode for numpy arrays passed to workers
    mmap_mode: str = "r"


ConfigStore.instance().store(
    group="hydra/launcher",
    name="joblib",
    node=JobLibLauncherConf,
    provider="joblib_launcher",
)
