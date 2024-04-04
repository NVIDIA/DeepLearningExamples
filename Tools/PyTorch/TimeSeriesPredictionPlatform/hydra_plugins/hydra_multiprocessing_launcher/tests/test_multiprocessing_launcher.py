# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
from typing import Any

from hydra.core.plugins import Plugins
from hydra.plugins.launcher import Launcher
from hydra.test_utils.launcher_common_tests import (
    IntegrationTestSuite,
    LauncherTestSuite,
)
from hydra.test_utils.test_utils import TSweepRunner, chdir_plugin_root
from pytest import mark

from hydra_plugins.hydra_multiprocessing_launcher.multiprocessing_launcher import MultiprocessingLauncher

chdir_plugin_root()


def test_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking for Launchers
    assert MultiprocessingLauncher.__name__ in [
        x.__name__ for x in Plugins.instance().discover(Launcher)
    ]


@mark.parametrize("launcher_name, overrides", [("multiprocessing", [])])
class TestMultiprocessingLauncher(LauncherTestSuite):
    """
    Run the Launcher test suite on this launcher.
    """

    pass


@mark.parametrize(
    "task_launcher_cfg, extra_flags",
    [
        # multiprocessing with process-based backend (default)
        (
            {},
            [
                "-m",
                "hydra/job_logging=hydra_debug",
                "hydra/job_logging=disabled",
                "hydra/launcher=multiprocessing",
            ],
        )
    ],
)
class TestMultiprocessingLauncherIntegration(IntegrationTestSuite):
    """
    Run this launcher through the integration test suite.
    """

    pass


def test_example_app(hydra_sweep_runner: TSweepRunner, tmpdir: Any) -> None:
    with hydra_sweep_runner(
        calling_file="example/my_app.py",
        calling_module=None,
        task_function=None,
        config_path=".",
        config_name="config",
        overrides=["task=1,2,3,4", f"hydra.sweep.dir={tmpdir}"],
    ) as sweep:
        overrides = {("task=1",), ("task=2",), ("task=3",), ("task=4",)}

        assert sweep.returns is not None and len(sweep.returns[0]) == 4
        for ret in sweep.returns[0]:
            assert tuple(ret.overrides) in overrides


@mark.parametrize(
    "overrides",
    [
        "hydra.launcher.processes=1",
        "hydra.launcher.maxtasksperchild=1"
    ],
)
def test_example_app_launcher_overrides(
    hydra_sweep_runner: TSweepRunner, overrides: str
) -> None:
    with hydra_sweep_runner(
        calling_file="example/my_app.py",
        calling_module=None,
        task_function=None,
        config_path=".",
        config_name="config",
        overrides=[overrides],
    ) as sweep:
        assert sweep.returns is not None and len(sweep.returns[0]) == 1
