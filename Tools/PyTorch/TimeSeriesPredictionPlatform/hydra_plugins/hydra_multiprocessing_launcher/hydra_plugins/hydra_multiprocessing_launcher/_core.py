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
from pathlib import Path
from typing import Any, Dict, Union, List, Sequence
from enum import Enum

import cloudpickle

from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from hydra.core.utils import (
    JobReturn,
    JobStatus,
    configure_log,
    filter_overrides,
    run_job,
    setup_globals,
)
from hydra.plugins.sweeper import ExperimentSequence
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, open_dict
import multiprocessing as mp
import multiprocessing.connection  # needed to use mp.connection

from .multiprocessing_launcher import MultiprocessingLauncher

log = logging.getLogger(__name__)


class WaitingStrategy(Enum):
    FIRST_COMPLETED = 'first_completed'
    ALL_COMPLETED = 'all_completed'


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


def _proxy_fn_call(results_queue, collection_lock, *args):
    args = [cloudpickle.loads(obj) for obj in args]
    result = execute_job(*args)
    with collection_lock:
        results_queue.put((int(mp.current_process().name), cloudpickle.dumps(result)))


def wait_for_results(running_tasks, 
                     results_queue, 
                     idx_to_process, 
                     collection_lock, 
                     return_when=WaitingStrategy.ALL_COMPLETED):
    if not running_tasks:
        return [], []
    # waiting_strategy = all if return_when is WaitingStrategy.ALL_COMPLETED else any
    keep_waiting = True 
    finished = []   
    results = []
    while keep_waiting:
        mp.connection.wait([p.sentinel for p in running_tasks])
        with collection_lock:
            while not results_queue.empty():
                idx, ret = results_queue.get()
                finished.append(idx_to_process[idx])
                results.append(cloudpickle.loads(ret))

            for p in running_tasks:
                if not p.is_alive() and p.exitcode != 0:
                    e = mp.ProcessError('Worker process terminated unexpectedly!')
                    ret = JobReturn()
                    ret.return_value = e
                    ret.status = JobStatus.FAILED
                    finished.append(p)
                    results.append(ret)
                if not p.is_alive():
                    p.join()

        if return_when is WaitingStrategy.ALL_COMPLETED:
            keep_waiting = len(results) != len(running_tasks)
        else:
            keep_waiting = len(results) == 0

    return finished, results


def launch(
    launcher: MultiprocessingLauncher,
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

    singleton_state = Singleton.get_state()
    batch_size = v if (v := launcher.mp_config['n_jobs']) else mp.cpu_count()

    runs = [None for _ in range(len(job_overrides))]
    log.info(
        "Multiprocessing({}) is launching {} jobs".format(
            ",".join([f"{k}={v}" for k, v in launcher.mp_config.items()]),
            'generator of' if isinstance(job_overrides, ExperimentSequence) else len(job_overrides),
        )
    )

    running_tasks = {}
    collection_lock = launcher.mp_context.Lock()
    results_queue = launcher.mp_context.Queue()
    idx_to_process = {}

    for idx, override in enumerate(job_overrides):
        log.info("\t#{} : {}".format(idx, " ".join(filter_overrides(override))))
        p = launcher.mp_context.Process(
                target=_proxy_fn_call,
                args=(results_queue,
                      collection_lock,
                      *[cloudpickle.dumps(obj)
                        for obj in (
                        initial_job_idx + idx,
                        override,
                        launcher.hydra_context,
                        launcher.config,
                        launcher.task_function,
                        singleton_state)]
                ),
                name=str(idx)
        )
        running_tasks[p] = (override, idx)
        idx_to_process[idx] = p
        p.start()

        if len(running_tasks) == batch_size:
            finished, results = wait_for_results(running_tasks, 
                                                 results_queue, 
                                                 idx_to_process, 
                                                 collection_lock, 
                                                 return_when=WaitingStrategy.FIRST_COMPLETED)

            overrides = [running_tasks[f] for f in finished]
            running_tasks = {task: running_tasks[task] for task in running_tasks if task not in finished}

            for (override, idx), res in zip(overrides, results):
                runs[idx] = res
                del idx_to_process[idx]
                if isinstance(job_overrides, ExperimentSequence):
                    try:
                        job_overrides.update_sequence((override, res))
                    except:
                        [p.terminate() for p in idx_to_process.values()]
                        raise
    
    finished, results = wait_for_results(running_tasks, 
                                         results_queue, 
                                         idx_to_process, 
                                         collection_lock, 
                                         return_when=WaitingStrategy.ALL_COMPLETED)

    overrides = [running_tasks[f] for f in finished]

    for (override, idx), res in zip(overrides, results):
        runs[idx] = res
        del idx_to_process[idx]
        if isinstance(job_overrides, ExperimentSequence):
            try:
                job_overrides.update_sequence((override, res))
            except:
                [p.terminate() for p in idx_to_process.values()]
                raise

    #launcher.executor.close()
    assert isinstance(runs, List)
    for run in runs:
        assert isinstance(run, JobReturn)
    return runs
