# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, List, Optional

import optuna
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.test_utils.test_utils import (
    TSweepRunner,
    chdir_plugin_root,
    run_process,
    run_python_script,
)
from omegaconf import DictConfig, OmegaConf
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    DiscreteUniformDistribution,
    IntLogUniformDistribution,
    IntUniformDistribution,
    LogUniformDistribution,
    UniformDistribution,
)
from optuna.samplers import RandomSampler
from pytest import mark, warns

from hydra_plugins.hydra_optuna_sweeper import _impl
from hydra_plugins.hydra_optuna_sweeper._impl import OptunaSweeperImpl
from hydra_plugins.hydra_optuna_sweeper.config import Direction
from hydra_plugins.hydra_optuna_sweeper.optuna_sweeper import OptunaSweeper

chdir_plugin_root()


def test_discovery() -> None:
    assert OptunaSweeper.__name__ in [
        x.__name__ for x in Plugins.instance().discover(Sweeper)
    ]


def check_distribution(expected: BaseDistribution, actual: BaseDistribution) -> None:
    if not isinstance(expected, CategoricalDistribution):
        assert expected == actual
        return

    assert isinstance(actual, CategoricalDistribution)
    # shuffle() will randomize the order of items in choices.
    assert set(expected.choices) == set(actual.choices)


@mark.parametrize(
    "input, expected",
    [
        (
            {"type": "categorical", "choices": [1, 2, 3]},
            CategoricalDistribution([1, 2, 3]),
        ),
        ({"type": "int", "low": 0, "high": 10}, IntUniformDistribution(0, 10)),
        (
            {"type": "int", "low": 0, "high": 10, "step": 2},
            IntUniformDistribution(0, 10, step=2),
        ),
        ({"type": "int", "low": 0, "high": 5}, IntUniformDistribution(0, 5)),
        (
            {"type": "int", "low": 1, "high": 100, "log": True},
            IntLogUniformDistribution(1, 100),
        ),
        ({"type": "float", "low": 0, "high": 1}, UniformDistribution(0, 1)),
        (
            {"type": "float", "low": 0, "high": 10, "step": 2},
            DiscreteUniformDistribution(0, 10, 2),
        ),
        (
            {"type": "float", "low": 1, "high": 100, "log": True},
            LogUniformDistribution(1, 100),
        ),
    ],
)
def test_create_optuna_distribution_from_config(input: Any, expected: Any) -> None:
    actual = _impl.create_optuna_distribution_from_config(input)
    check_distribution(expected, actual)


@mark.parametrize(
    "input, expected",
    [
        ("key=choice(1,2)", CategoricalDistribution([1, 2])),
        ("key=choice(true, false)", CategoricalDistribution([True, False])),
        ("key=choice('hello', 'world')", CategoricalDistribution(["hello", "world"])),
        ("key=shuffle(range(1,3))", CategoricalDistribution((1, 2))),
        ("key=range(1,3)", IntUniformDistribution(1, 3)),
        ("key=interval(1, 5)", UniformDistribution(1, 5)),
        ("key=int(interval(1, 5))", IntUniformDistribution(1, 5)),
        ("key=tag(log, interval(1, 5))", LogUniformDistribution(1, 5)),
        ("key=tag(log, int(interval(1, 5)))", IntLogUniformDistribution(1, 5)),
        ("key=range(0.5, 5.5, step=1)", DiscreteUniformDistribution(0.5, 5.5, 1)),
    ],
)
def test_create_optuna_distribution_from_override(input: Any, expected: Any) -> None:
    parser = OverridesParser.create()
    parsed = parser.parse_overrides([input])[0]
    actual = _impl.create_optuna_distribution_from_override(parsed)
    check_distribution(expected, actual)


@mark.parametrize(
    "input, expected",
    [
        (["key=choice(1,2)"], ({"key": CategoricalDistribution([1, 2])}, {})),
        (["key=5"], ({}, {"key": "5"})),
        (
            ["key1=choice(1,2)", "key2=5"],
            ({"key1": CategoricalDistribution([1, 2])}, {"key2": "5"}),
        ),
        (
            ["key1=choice(1,2)", "key2=5", "key3=range(1,3)"],
            (
                {
                    "key1": CategoricalDistribution([1, 2]),
                    "key3": IntUniformDistribution(1, 3),
                },
                {"key2": "5"},
            ),
        ),
    ],
)
def test_create_params_from_overrides(input: Any, expected: Any) -> None:
    actual = _impl.create_params_from_overrides(input)
    assert actual == expected


def test_launch_jobs(hydra_sweep_runner: TSweepRunner) -> None:
    sweep = hydra_sweep_runner(
        calling_file=None,
        calling_module="hydra.test_utils.a_module",
        config_path="configs",
        config_name="compose.yaml",
        task_function=None,
        overrides=[
            "hydra/sweeper=optuna",
            "hydra/launcher=basic",
            "hydra.sweeper.n_trials=8",
            "hydra.sweeper.n_jobs=3",
        ],
    )
    with sweep:
        assert sweep.returns is None


@mark.parametrize("with_commandline", (True, False))
def test_optuna_example(with_commandline: bool, tmpdir: Path) -> None:
    storage = "sqlite:///" + os.path.join(str(tmpdir), "test.db")
    study_name = "test-optuna-example"
    cmd = [
        "example/sphere.py",
        "--multirun",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.sweeper.n_trials=20",
        "hydra.sweeper.n_jobs=1",
        f"hydra.sweeper.storage={storage}",
        f"hydra.sweeper.study_name={study_name}",
        "hydra/sweeper/sampler=tpe",
        "hydra.sweeper.sampler.seed=123",
        "~z",
    ]
    if with_commandline:
        cmd += [
            "x=choice(0, 1, 2)",
            "y=0",  # Fixed parameter
        ]
    run_python_script(cmd)
    returns = OmegaConf.load(f"{tmpdir}/optimization_results.yaml")
    study = optuna.load_study(storage=storage, study_name=study_name)
    best_trial = study.best_trial
    assert isinstance(returns, DictConfig)
    assert returns.name == "optuna"
    assert returns["best_params"]["x"] == best_trial.params["x"]
    if with_commandline:
        assert "y" not in returns["best_params"]
        assert "y" not in best_trial.params
    else:
        assert returns["best_params"]["y"] == best_trial.params["y"]
    assert returns["best_value"] == best_trial.value
    # Check the search performance of the TPE sampler.
    # The threshold is the 95th percentile calculated with 1000 different seed values
    # to make the test robust against the detailed implementation of the sampler.
    # See https://github.com/facebookresearch/hydra/pull/1746#discussion_r681549830.
    assert returns["best_value"] <= 2.27


@mark.parametrize("num_trials", (10, 1))
def test_example_with_grid_sampler(
    tmpdir: Path,
    num_trials: int,
) -> None:
    storage = "sqlite:///" + os.path.join(str(tmpdir), "test.db")
    study_name = "test-grid-sampler"
    cmd = [
        "example/sphere.py",
        "--multirun",
        "--config-dir=tests/conf",
        "--config-name=test_grid",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.job.chdir=False",
        f"hydra.sweeper.n_trials={num_trials}",
        "hydra.sweeper.n_jobs=1",
        f"hydra.sweeper.storage={storage}",
        f"hydra.sweeper.study_name={study_name}",
    ]
    run_python_script(cmd)
    returns = OmegaConf.load(f"{tmpdir}/optimization_results.yaml")
    assert isinstance(returns, DictConfig)
    bv, bx, by, bz = (
        returns["best_value"],
        returns["best_params"]["x"],
        returns["best_params"]["y"],
        returns["best_params"]["z"],
    )
    if num_trials >= 12:
        assert bv == 1 and abs(bx) == 1 and by == 0
    else:
        assert bx in [-1, 1] and by in [-1, 0]
    assert bz in ["foo", "bar"]


@mark.parametrize("with_commandline", (True, False))
def test_optuna_multi_objective_example(with_commandline: bool, tmpdir: Path) -> None:
    cmd = [
        "example/multi-objective.py",
        "--multirun",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.sweeper.n_trials=20",
        "hydra.sweeper.n_jobs=1",
        "hydra/sweeper/sampler=random",
        "hydra.sweeper.sampler.seed=123",
    ]
    if with_commandline:
        cmd += [
            "x=range(0, 5)",
            "y=range(0, 3)",
        ]
    run_python_script(cmd)
    returns = OmegaConf.load(f"{tmpdir}/optimization_results.yaml")
    assert isinstance(returns, DictConfig)
    assert returns.name == "optuna"
    if with_commandline:
        for trial_x in returns["solutions"]:
            assert trial_x["params"]["x"] % 1 == 0
            assert trial_x["params"]["y"] % 1 == 0
            # The trials must not dominate each other.
            for trial_y in returns["solutions"]:
                assert not _dominates(trial_x, trial_y)
    else:
        for trial_x in returns["solutions"]:
            assert trial_x["params"]["x"] % 1 in {0, 0.5}
            assert trial_x["params"]["y"] % 1 in {0, 0.5}
            # The trials must not dominate each other.
            for trial_y in returns["solutions"]:
                assert not _dominates(trial_x, trial_y)


def _dominates(values_x: List[float], values_y: List[float]) -> bool:
    return all(x <= y for x, y in zip(values_x, values_y)) and any(
        x < y for x, y in zip(values_x, values_y)
    )


def test_optuna_custom_search_space_example(tmpdir: Path) -> None:
    max_z_difference_from_x = 0.3
    cmd = [
        "example/custom-search-space-objective.py",
        "--multirun",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.sweeper.n_trials=20",
        "hydra.sweeper.n_jobs=1",
        "hydra/sweeper/sampler=random",
        "hydra.sweeper.sampler.seed=123",
        f"max_z_difference_from_x={max_z_difference_from_x}",
    ]
    run_python_script(cmd)
    returns = OmegaConf.load(f"{tmpdir}/optimization_results.yaml")
    assert isinstance(returns, DictConfig)
    assert returns.name == "optuna"
    assert (
        abs(returns["best_params"]["x"] - returns["best_params"]["z"])
        <= max_z_difference_from_x
    )
    w = returns["best_params"]["+w"]
    assert 0 <= w <= 1


@mark.parametrize(
    "search_space,params,raise_warning,msg",
    [
        (None, None, False, None),
        (
            {},
            {},
            True,
            r"Both hydra.sweeper.params and hydra.sweeper.search_space are configured.*",
        ),
        (
            {},
            None,
            True,
            r"`hydra.sweeper.search_space` is deprecated and will be removed in the next major release.*",
        ),
        (None, {}, False, None),
    ],
)
def test_warnings(
    tmpdir: Path,
    search_space: Optional[DictConfig],
    params: Optional[DictConfig],
    raise_warning: bool,
    msg: Optional[str],
) -> None:
    partial_sweeper = partial(
        OptunaSweeperImpl,
        sampler=RandomSampler(),
        direction=Direction.minimize,
        storage=None,
        study_name="test",
        n_trials=1,
        n_jobs=1,
        max_failure_rate=0.0,
        custom_search_space=None,
    )
    if search_space is not None:
        search_space = OmegaConf.create(search_space)
    if params is not None:
        params = OmegaConf.create(params)
    sweeper = partial_sweeper(search_space=search_space, params=params)
    if raise_warning:
        with warns(
            UserWarning,
            match=msg,
        ):
            sweeper._process_searchspace_config()
    else:
        sweeper._process_searchspace_config()


@mark.parametrize("max_failure_rate", (0.5, 1.0))
def test_failure_rate(max_failure_rate: float, tmpdir: Path) -> None:
    cmd = [
        sys.executable,
        "example/sphere.py",
        "--multirun",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.sweeper.n_trials=20",
        "hydra.sweeper.n_jobs=2",
        "hydra/sweeper/sampler=random",
        "hydra.sweeper.sampler.seed=123",
        f"hydra.sweeper.max_failure_rate={max_failure_rate}",
        "error=true",
    ]
    out, err = run_process(cmd, print_error=False, raise_exception=False)
    error_string = "RuntimeError: cfg.error is True"
    if max_failure_rate < 1.0:
        assert error_string in err
    else:
        assert error_string not in err


def test_example_with_deprecated_search_space(
    tmpdir: Path,
) -> None:
    cmd = [
        "-W ignore::UserWarning",
        "example/sphere.py",
        "--multirun",
        "--config-dir=tests/conf",
        "--config-name=test_deprecated_search_space",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.sweeper.n_trials=20",
        "hydra.sweeper.n_jobs=1",
    ]

    run_python_script(cmd)
    returns = OmegaConf.load(f"{tmpdir}/optimization_results.yaml")
    assert isinstance(returns, DictConfig)
    assert returns.name == "optuna"
    assert abs(returns["best_params"]["x"]) <= 5.5
