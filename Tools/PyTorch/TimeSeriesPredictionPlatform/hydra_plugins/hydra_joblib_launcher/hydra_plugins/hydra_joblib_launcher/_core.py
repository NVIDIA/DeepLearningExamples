# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from pathlib import Path
from typing import Any, Dict, Union, List, Sequence

from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from hydra.core.utils import (
    JobReturn,
    configure_log,
    filter_overrides,
    run_job,
    setup_globals,
)
from hydra.plugins.sweeper import ExperimentSequence
from hydra.types import HydraContext, TaskFunction
from joblib import Parallel, delayed  # type: ignore
from omegaconf import DictConfig, open_dict
import multiprocessing as mp

from .joblib_launcher import JoblibLauncher

log = logging.getLogger(__name__)


def execute_job(
    idx: int,
    overrides: Sequence[str],
    hydra_context: HydraContext,
    config: DictConfig,
    task_function: TaskFunction,
    singleton_state: Dict[Any, Any],
) -> JobReturn:
    """Calls `run_job` in parallel"""
    setup_globals()
    Singleton.set_state(singleton_state)

    sweep_config = hydra_context.config_loader.load_sweep_config(
        config, list(overrides)
    )
    with open_dict(sweep_config):
        sweep_config.hydra.job.id = "{}_{}".format(sweep_config.hydra.job.name, idx)
        sweep_config.hydra.job.num = idx
    HydraConfig.instance().set_config(sweep_config)

    ret = run_job(
        hydra_context=hydra_context,
        config=sweep_config,
        task_function=task_function,
        job_dir_key="hydra.sweep.dir",
        job_subdir_key="hydra.sweep.subdir",
    )

    return ret


def process_joblib_cfg(joblib_cfg: Dict[str, Any]) -> None:
    for k in ["pre_dispatch", "batch_size", "max_nbytes"]:
        if k in joblib_cfg.keys():
            try:
                val = joblib_cfg.get(k)
                if val:
                    joblib_cfg[k] = int(val)
            except ValueError:
                pass


def _batch_sequence(sequence, batch_size=1):
    while True:
        overrides = [experiment_config for _, experiment_config in zip(range(batch_size), sequence)]
        if overrides:
            yield overrides
        if len(overrides) != batch_size:
            raise StopIteration


def launch(
    launcher: JoblibLauncher,
    job_overrides: Union[Sequence[Sequence[str]], ExperimentSequence],
    initial_job_idx: int,
) -> Sequence[JobReturn]:
    """
    :param job_overrides: an Iterable of List<String>, where each inner list is the arguments for one job run.
    :param initial_job_idx: Initial job idx in batch.
    :return: an array of return values from run_job with indexes corresponding to the input list indexes.
    """
    setup_globals()
    assert launcher.config is not None
    assert launcher.task_function is not None
    assert launcher.hydra_context is not None

    configure_log(launcher.config.hydra.hydra_logging, launcher.config.hydra.verbose)
    sweep_dir = Path(str(launcher.config.hydra.sweep.dir))
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Joblib's backend is hard-coded to loky since the threading
    # backend is incompatible with Hydra
    joblib_cfg = launcher.joblib
    joblib_cfg["backend"] = "loky"
    process_joblib_cfg(joblib_cfg)
    singleton_state = Singleton.get_state()

    if isinstance(job_overrides, ExperimentSequence):
        log.info(
            "Joblib.Parallel({}) is launching {} jobs".format(
                ",".join([f"{k}={v}" for k, v in joblib_cfg.items()]),
                'generator of',
            )
        )
        batch_size = v if (v := joblib_cfg['n_jobs']) != -1 else mp.cpu_count()
        runs = []
        overrides = []
        for idx, overrides in enumerate(_batch_sequence(job_overrides, batch_size)):
            for i, override in enumerate(overrides):
                log.info("\t#{} : {}".format(idx*batch_size+i, " ".join(filter_overrides(override))))
            results = Parallel(**joblib_cfg)(
                delayed(execute_job)(
                    initial_job_idx + idx,
                    override,
                    launcher.hydra_context,
                    launcher.config,
                    launcher.task_function,
                    singleton_state,
                )
                for override in overrides
            )
            for experiment_result in zip(overrides, results):
                job_overrides.update_sequence(experiment_result)
    else:
        log.info(
            "Joblib.Parallel({}) is launching {} jobs".format(
                ",".join([f"{k}={v}" for k, v in joblib_cfg.items()]),
                len(job_overrides),
            )
        )
        log.info("Launching jobs, sweep output dir : {}".format(sweep_dir))
        for idx, overrides in enumerate(job_overrides):
            log.info("\t#{} : {}".format(idx, " ".join(filter_overrides(overrides))))

        runs = Parallel(**joblib_cfg)(
            delayed(execute_job)(
                initial_job_idx + idx,
                overrides,
                launcher.hydra_context,
                launcher.config,
                launcher.task_function,
                singleton_state,
            )
            for idx, overrides in enumerate(job_overrides)
        )

    assert isinstance(runs, List)
    for run in runs:
        assert isinstance(run, JobReturn)
    return runs
