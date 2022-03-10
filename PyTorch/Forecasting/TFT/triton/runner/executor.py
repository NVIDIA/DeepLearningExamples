# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import pathlib
import shutil
import traceback
from typing import Dict, List, Optional

from colorama import Fore

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from ..deployment_toolkit.core import Accelerator, Precision
from .core import Paths
from .exceptions import RunnerException
from .experiment import ExperimentResult, ExperimentStatus, Status
from .exporter import CommandsExporter
from .logger import LOGGER
from .maintainer import Container, Maintainer
from .pipeline import Pipeline
from .stages import Stage
from .task import Experiment, Task
from .triton import Triton
from .utils import clean_directory, exec_command, format_env_key, format_env_value, get_result_path


class Executor:
    """
    Experiments executor
    """

    def __init__(
        self,
        workspace: pathlib.Path,
        maintainer: Maintainer,
        pipeline: Pipeline,
        devices: List[str] = None,
    ):
        """
        Initialize experiments executor

        Args:
            workspace: Path to workspace to store artifacts
            maintainer: maintainer for running commands
            pipeline: pipeline definition

            devices: List of devices on which Triton Inference Server will be executed
        """
        self._maintainer = maintainer
        self._pipeline = pipeline
        self._devices = devices or ["0"]

        self._workspace = workspace
        self._executor_workspace = workspace / "executor"
        self._shared_dir = self._executor_workspace / "shared"
        self._triton_models_repository_dir = self._executor_workspace / "triton_models"
        self._scripts_dir = self._executor_workspace / "scripts"
        self._libraries_dir = self._executor_workspace / "libs"

        self._exporter = CommandsExporter(self._scripts_dir)
        self._triton_container: Optional[Container] = None

    def start(self, task: Task):
        """
        Process the task and execute experiments.
        """
        self._create_dirs()
        total_experiment = len(task.experiments)
        LOGGER.info(f"Total experiments to verify: {total_experiment}")
        for idx, experiment in enumerate(task.experiments, start=1):
            LOGGER.info(
                f"{Fore.CYAN}================ Experiment: {idx}/{total_experiment} Started ================{Fore.RESET}"
            )
            results = {}
            environment = self._prepare_environment(task, experiment.parameters)
            LOGGER.info(f"Experiment details")
            LOGGER.info(json.dumps(environment, indent=4))

            self._clean_experiment_artifacts(idx, total_experiment)
            self._create_experiment_results_dir(task, experiment)

            experiment.start()

            LOGGER.info("Running Triton Servers:")
            log_file = self._workspace / task.logs_dir / f"triton-server-experiment-{idx}.log"
            self._triton_container = self._triton_server_container(
                triton_container_image=task.triton_container_image,
                framework=task.framework,
                accelerator=experiment.parameters["accelerator"],
                precision=experiment.parameters["precision"],
                custom_library=bool(task.triton_custom_operations is not None),
                load_model_method=task.triton_load_model_method,
                log_file=log_file,
            )

            try:
                self._triton_container.start()

                for stage in self._pipeline.stages():
                    LOGGER.info(
                        f"{Fore.GREEN}[Experiment: {idx}/{total_experiment}] ================ Stage {stage.label} Started ================{Fore.RESET}"
                    )
                    experiment_stage = experiment.stages[stage.label]
                    experiment_stage.start()

                    is_ok = self._run_stage(stage=stage)
                    if not is_ok:
                        LOGGER.error(f"Stage {stage.label} failed.")
                        break

                    self._save_results(task, experiment, stage.label, results)
                    experiment_stage.end()

                    LOGGER.info(
                        f"{Fore.GREEN}[Experiment: {idx}/{total_experiment}] ================ Stage {stage.label} Finished ================{Fore.RESET}"
                    )
            except Exception:
                message = traceback.format_exc()
                LOGGER.error(f"Error running experiment: {message}")
                yield ExperimentResult(
                    status=Status(state=ExperimentStatus.FAILED, message=message),
                    experiment=experiment,
                    results=results,
                )
            finally:
                self._triton_container.stop()

            experiment.end()
            LOGGER.info(
                f"{Fore.CYAN}================ Experiment: {idx}/{total_experiment} Finished ================{Fore.RESET}"
            )
            yield ExperimentResult(
                status=Status(state=ExperimentStatus.SUCCEED, message="Experiment Succeed"),
                experiment=experiment,
                results=results,
            )

    def stop(self) -> None:
        """
        Stop executor

        Returns:
            None
        """
        if self._triton_container:
            self._triton_container.stop()

    def _prepare_environment(self, task: Task, parameters: Dict) -> Dict:
        """
        Prepare environment data and export it

        Args:
            parameters: Key and values which should be exported to environment

        Returns:
            Dictionary with environment data
        """
        environment = {
            "MODEL_NAME": task.model_name,
            "FRAMEWORK": task.framework,
            "SHARED_DIR": self._shared_dir.as_posix(),
            "MODEL_REPOSITORY_PATH": self._triton_models_repository_dir.as_posix(),
            "TRITON_SERVER_URL": "localhost",
            "TRITON_INSTANCES": "1",
            "TRITON_LOAD_MODEL_METHOD": task.triton_load_model_method,
        }

        checkpoint_variant = parameters.get("checkpoint_variant")
        if checkpoint_variant:
            del parameters["checkpoint_variant"]
            environment["CHECKPOINT_DIR"] = task.checkpoints[checkpoint_variant].path.as_posix()

        if task.datasets_dir:
            environment["DATASETS_DIR"] = task.datasets_dir.as_posix()

        for key, value in parameters.items():
            key = format_env_key(key)
            value = format_env_value(value)
            environment[key] = value

        for key, value in environment.items():
            os.environ[key] = value

        return environment

    def _triton_server_container(
        self,
        triton_container_image: str,
        framework: str,
        load_model_method: str,
        accelerator: str,
        precision: str,
        log_file: pathlib.Path,
        custom_library: bool,
    ) -> Container:
        """
        Create Triton Inference Server container for experiment

        Args:
            triton_container_image: Triton Inference Server container image
            framework: Framework used to run model
            accelerator: Accelerator used for experiment
            precision: Precision used for experiment
            load_model_method: Configure how Triton will load model
            log_file: File where Triton logs are stored

        Returns:
            Container object
        """
        volumes = {
            self._triton_models_repository_dir: {"bind": Paths.MODEL_REPOSITORY_PATH, "mode": "rw"},
            self._libraries_dir: {"bind": Paths.LIBRARIES_PATH, "mode": "rw"},
        }

        environment = {
            "MODEL_REPOSITORY_PATH": Paths.MODEL_REPOSITORY_PATH,
            "LIBRARIES_PATH": Paths.LIBRARIES_PATH,
            "TRITON_LOAD_MODEL_METHOD": load_model_method,
        }

        if custom_library:
            library_path = Triton.library_path(framework=framework)
            environment["LD_LIBRARY_PATH"] = f"{library_path}:${{LD_LIBRARY_PATH}}"
            environment["LD_PRELOAD"] = Triton.custom_library_path_remote()

        if accelerator == Accelerator.TRT.value and precision == Precision.FP16.value:
            environment["ORT_TENSORRT_FP16_ENABLE"] = 1

        strict_mode = False
        command = Triton.command(
            framework=framework,
            repository_path=Paths.MODEL_REPOSITORY_PATH,
            strict_mode=strict_mode,
        )
        command = f' bash -c "{command}"'

        container = self._maintainer.triton_container(
            command=command,
            image=triton_container_image,
            devices=self._devices,
            volumes=volumes,
            environment=environment,
            log_file=log_file,
        )

        return container

    def _save_results(self, task: Task, experiment: Experiment, stage_name: str, results: Dict) -> None:
        """
        Update results for stage

        Args:
            task: Task object
            experiment: Experiment for which stage has to be updated
            stage_name: Name of stage
            results: Results path mapping

        Returns:
            None
        """
        stage = experiment.stages[stage_name]

        if not stage.result_path:
            LOGGER.debug(f"No results file to copy for {stage.name}")
            return

        if not stage.result_type:
            LOGGER.debug(f"No results type provided for {stage.name}")
            return

        os.environ["SHARED_DIR"] = self._shared_dir.as_posix()
        result_path = get_result_path(result_path=stage.result_path)
        result_path = pathlib.Path(result_path)

        if not result_path.is_file() and not result_path.is_dir():
            raise RunnerException(f"Results file {result_path} not found.")

        experiment_dir = self._workspace / task.results_dir / experiment.results_dir

        LOGGER.info(f"Saving {stage.result_type} to {experiment_dir}")

        if result_path.is_dir():
            dst_path = experiment_dir / stage.result_type
            shutil.copytree(result_path, dst_path)
        elif result_path.is_file():
            suffix = result_path.suffix
            dst_path = experiment_dir / f"{stage.result_type}{suffix}"
            shutil.copy(result_path, dst_path)
        else:
            raise RunnerException(f"Result not found {result_path}")
        LOGGER.info("Done")

        results[stage.result_type] = dst_path

    def _create_dirs(self) -> None:
        """
        Create directories used to store artifacts and final results

        Returns:
            None
        """
        LOGGER.info(f"{Fore.GREEN}================ Creating Artifacts Directories Started ================{Fore.RESET}")

        if self._executor_workspace.is_dir():
            LOGGER.info(f"Removing previous executor workspace: {self._executor_workspace}")
            shutil.rmtree(self._executor_workspace)

        for directory in [
            self._libraries_dir,
            self._shared_dir,
            self._scripts_dir,
            self._triton_models_repository_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"Directory {directory.name} created.")
        LOGGER.info(
            f"{Fore.GREEN}================ Creating Artifacts Directories Finished ================{Fore.RESET}"
        )

    def _clean_experiment_artifacts(self, idx: int, total: int) -> None:
        """
        Clean artifacts stored between experiments

        Returns:
            None
        """
        LOGGER.info(
            f"{Fore.GREEN}[Experiment: {idx}/{total}] ================ Cleanup Experiment Data Started ================{Fore.RESET}"
        )
        for directory in [
            self._shared_dir,
            self._scripts_dir,
            self._triton_models_repository_dir,
        ]:
            clean_directory(directory)
            LOGGER.info(f"Location {directory} cleaned.")
        LOGGER.info(
            f"{Fore.GREEN}[Experiment: {idx}/{total}] ================ Cleanup Experiment Data Finished ================{Fore.RESET}"
        )

    def _create_experiment_results_dir(self, task: Task, experiment: Experiment):
        """
        Create result directory for experiment

        Returns:

        """
        experiment_dir = self._workspace / task.results_dir / experiment.results_dir
        experiment_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_triton_custom_operations(self, task: Task) -> None:
        """
        Prepare Triton Server custom operations library

        Returns:
            None
        """
        if task.triton_custom_operations:
            target_library_path = Triton.custom_library_path_local(self._libraries_dir)
            target_library_path_dir = target_library_path.parent
            target_library_path_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(task.triton_custom_operations, target_library_path)

    def _run_stage(self, stage: Stage) -> bool:
        """
        Run single stage commands

        Args:
            stage: Stage object with defined commands

        Returns:
            True on success, False otherwise
        """
        try:
            command = self._exporter.export(stage=stage)
            exec_command(command)
        except RunnerException:
            return False

        return True
