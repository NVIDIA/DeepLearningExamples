# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

import logging
import sys
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple
)

import optuna
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from optuna.trial import Trial
from hydra.plugins.sweeper import ExperimentSequence
from optuna.distributions import BaseDistribution
import torch

log = logging.getLogger(__name__)


def get_config(config_name, config_path, override_list=None, return_hydra_config=False):
    GlobalHydra.instance().clear()
    initialize(config_path=config_path)
    cfg = compose(config_name, return_hydra_config=return_hydra_config, overrides=override_list)
    if return_hydra_config:
        HydraConfig().cfg = cfg
        OmegaConf.resolve(cfg)
    return cfg


class TSPPOptunaExperimentSequence(ExperimentSequence):
    def __init__(self,
                 study,
                 num_experiments,
                 search_space_distributions,
                 fixed_params,
                 directions,
                 custom_search_space_extender,
                 max_failure_rate=0.0,
                 is_grid_sampler=False) -> None:
        self.study = study 
        self.num_experiments = num_experiments
        self.search_space_distributions = search_space_distributions
        self.fixed_params = fixed_params
        self.directions = directions
        self.custom_search_space_extender = custom_search_space_extender
        self.max_failure_rate = max_failure_rate
        self.fault_tolerance = int(num_experiments * max_failure_rate)
        self.is_grid_sampler = is_grid_sampler
        self.idx = -1
        self.override_trial_mapping = {}
        self.idle_devices = set(range(torch.cuda.device_count()))
        self.trial_device = {}

    def _configure_trial(
        self,
        trial: Trial,
        search_space_distributions: Dict[str, BaseDistribution],
        fixed_params: Dict[str, Any],
        gpu_id: int
    ) -> Sequence[str]:
        for param_name, distribution in search_space_distributions.items():
            assert type(param_name) is str
            trial._suggest(param_name, distribution)
        for param_name, value in fixed_params.items():
            trial.set_user_attr(param_name, value)

        if self.custom_search_space_extender:
            assert self.config is not None
            self.custom_search_space_extender(self.config, trial)

        overlap = trial.params.keys() & trial.user_attrs
        if len(overlap):
            raise ValueError(
                "Overlapping fixed parameters and search space parameters found!"
                f"Overlapping parameters: {list(overlap)}"
            )
        params = dict(trial.params)
        params.update(fixed_params)
        params['+hydra.device_id'] = gpu_id

        return tuple(f"{name}={val}" for name, val in params.items())

    def update_sequence(self, experiment_result: Tuple[Sequence[str], Any]):
        override, ret = experiment_result
        trial = self.override_trial_mapping[override]
        self.idle_devices.add(self.trial_device[trial])
        values: Optional[List[float]] = None
        state: optuna.trial.TrialState = optuna.trial.TrialState.COMPLETE
        try:
            if len(self.directions) == 1:
                try:
                    values = [float(ret.return_value)]
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Return value must be float-castable. Got '{ret.return_value}'."
                    ).with_traceback(sys.exc_info()[2])
            else:
                try:
                    values = [float(v) for v in ret.return_value]
                except (ValueError, TypeError):
                    raise ValueError(
                        "Return value must be a list or tuple of float-castable values."
                        f" Got '{ret.return_value}'."
                    ).with_traceback(sys.exc_info()[2])
                if len(values) != len(self.directions):
                    raise ValueError(
                        "The number of the values and the number of the objectives are"
                        f" mismatched. Expect {len(self.directions)}, but actually {len(values)}."
                    )

            try:
                self.study.tell(trial=trial, state=state, values=values)
            except RuntimeError as e:
                if (
                    self.is_grid_sampler
                    and "`Study.stop` is supposed to be invoked inside an objective function or a callback."
                    in str(e)
                ):
                    pass
                else:
                    raise e

        except Exception as e:
            state = optuna.trial.TrialState.FAIL
            self.study.tell(trial=trial, state=state, values=values)
            log.warning(f"Failed experiment: {e}")
            self.fault_tolerance -= 1

        # raise if too many failures
        if self.fault_tolerance < 0:
            log.error(
                f"Failed {int(self.num_experiments * self.max_failure_rate) + 1} times out of {self.num_experiments} "
                f"with max_failure_rate={self.max_failure_rate}."
            )
            ret.return_value  # delegate raising to JobReturn, with actual traceback

    def __next__(self) -> Sequence[str]:
        self.idx += 1
        if self.idx < self.num_experiments:
            trial = self.study.ask()
            assert len(self.idle_devices) > 0, 'Number of simultaneous experiments is greater than number of gpus'
            device_id = self.idle_devices.pop()
            self.trial_device[trial] = device_id
            override = self._configure_trial(trial, self.search_space_distributions, self.fixed_params, device_id)
            self.override_trial_mapping[override] = trial
            return override
        else:
            raise StopIteration
    
    def __len__(self):
        return self.num_experiments
