# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

from functools import partial
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, open_dict
from hydra.types import HydraContext
from hydra.core.singleton import Singleton
from hydra.core.hydra_config import HydraConfig
from hydra.types import TaskFunction
from hydra.core.utils import (
    JobReturn,
    configure_log,
    filter_overrides,
    run_job,
    setup_globals,
    env_override,
)

from torch.distributed.launcher.api import LaunchConfig, launch_agent
from torch.distributed.elastic.multiprocessing import Std

from .distributed_launcher import TorchDistributedLauncher

log = logging.getLogger(__name__)


def setup(
    launcher: TorchDistributedLauncher,
    *,
    hydra_context: HydraContext,
    task_function: TaskFunction,
    config: DictConfig,
) -> None:
    launcher.config = config
    launcher.hydra_context = hydra_context
    launcher.task_function = task_function

    c = config.hydra.launcher
    launcher.launch_config = LaunchConfig(
        min_nodes=c.min_nodes,
        max_nodes=c.max_nodes,
        nproc_per_node=c.nproc_per_node,
        run_id=c.rdzv_id,
        role=c.role,
        rdzv_endpoint=c.rdzv_endpoint,
        rdzv_backend=c.rdzv_backend,
        rdzv_configs={"rank": 0},
        max_restarts=c.max_restarts,
        monitor_interval=c.monitor_interval,
        # start_method: Works only with fork.
        # Spawn and forkserver require pickling which does't work inside wrapped function
        start_method="fork",
        redirects=Std.from_str(c.redirects),
        tee=Std.from_str(c.tee),
        log_dir=c.get("log_dir"),
    )


def launch(
    launcher: TorchDistributedLauncher,
    job_overrides: Sequence[Sequence[str]],
    initial_job_idx: int,
) -> Sequence[JobReturn]:
    """
    :param job_overrides: a List of List<String>, where each inner list is the arguments for one job run.
    :param initial_job_idx: Initial job idx in batch.
    :return: an array of return values from run_job with indexes corresponding to the input list indexes.
    """
    setup_globals()
    assert launcher.config is not None
    assert launcher.hydra_context is not None
    assert launcher.task_function is not None

    configure_log(launcher.config.hydra.hydra_logging, launcher.config.hydra.verbose)
    sweep_dir = Path(str(launcher.config.hydra.sweep.dir))
    sweep_dir.mkdir(parents=True, exist_ok=True)
    runs = []

    for idx, overrides in enumerate(job_overrides):
        idx = initial_job_idx + idx
        lst = " ".join(filter_overrides(overrides))
        log.info(f"\t#{idx} : {lst}")
        sweep_config = launcher.hydra_context.config_loader.load_sweep_config(
            launcher.config, list(overrides)
        )
        with open_dict(sweep_config):
            # This typically coming from the underlying scheduler (SLURM_JOB_ID for instance)
            # In that case, it will not be available here because we are still in the main process.
            # but instead should be populated remotely before calling the task_function.
            sweep_config.hydra.job.id = f"job_id_for_{idx}"
            sweep_config.hydra.job.num = idx

        HydraConfig.instance().set_config(sweep_config)
        launcher.singleton_state = Singleton.get_state()

        def _task_function(task_function, singleton_state, task_cfg):
            return launch_agent(
                launcher.launch_config,
                wrapped_task_function,
                [task_function, launcher.singleton_state, task_cfg],
            )

        _task_function = partial(
            _task_function, launcher.task_function, launcher.singleton_state
        )

        ret = run_job(
            hydra_context=launcher.hydra_context,
            task_function=_task_function,
            config=sweep_config,
            job_dir_key="hydra.sweep.dir",
            job_subdir_key="hydra.sweep.subdir",
        )

        # We assume that main process has rank 0
        ret.return_value = ret.return_value[0]
        runs.append(ret)
        configure_log(
            launcher.config.hydra.hydra_logging, launcher.config.hydra.verbose
        )
    return runs


def wrapped_task_function(task_function, singleton_state, task_cfg):
    Singleton.set_state(singleton_state)
    env_set = HydraConfig.instance().cfg.hydra.job.env_set
    with env_override(env_set):
        ret = task_function(task_cfg)
        return ret
