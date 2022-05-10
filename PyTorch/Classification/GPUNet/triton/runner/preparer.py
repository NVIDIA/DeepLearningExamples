# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import abc
import pathlib
from datetime import datetime
from typing import Dict, List

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .config import Config
from .configuration import Configuration
from .downloader import download
from .experiment import Experiment, Stage
from .logger import LOGGER
from .maintainer import Maintainer
from .pipeline import Pipeline
from .stages import ResultsType, TritonPerformanceOfflineStage, TritonPerformanceOnlineStage
from .task import Checkpoint, Dataset, SystemInfo, Task
from .triton import Triton
from .utils import clean_directory


class Preparer(abc.ABC):
    """
    Runner preparer object.
    """

    @abc.abstractmethod
    def exec(
        self,
        workspace: pathlib.Path,
        config: Config,
        pipeline: Pipeline,
        maintainer: Maintainer,
        triton: Triton,
        logs_dir: pathlib.Path,
    ):
        pass


class ExperimentPreparer(Preparer):
    """
    Experiment runner preparer object.
    """

    def exec(
        self,
        workspace: pathlib.Path,
        config: Config,
        pipeline: Pipeline,
        maintainer: Maintainer,
        triton: Triton,
        logs_dir: pathlib.Path,
    ):
        LOGGER.info("Preparing Triton container image")
        triton_container_image = self._prepare_triton_container_image(config, maintainer, triton)

        LOGGER.info("Initialize task")
        task = self._initialize_task(
            workspace=workspace,
            config=config,
            pipeline=pipeline,
            triton_container_image=triton_container_image,
            logs_dir=logs_dir,
        )

        LOGGER.info("Preparing directories")
        self._create_dirs(workspace, task)

        LOGGER.info("Clean previous run artifacts directories")
        self._clean_previous_run_artifacts(workspace, task)

        LOGGER.info("Downloading checkpoints")
        self._download_checkpoints(task)

        return task

    def _create_dirs(self, workspace: pathlib.Path, task: Task) -> None:
        """
        Create directories used to store artifacts and final results

        Returns:
            None
        """
        for directory in [task.results_dir, task.logs_dir, task.checkpoints_dir]:
            directory_path = workspace / directory
            directory_path.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"Directory {directory} created.")

    def _clean_previous_run_artifacts(self, workspace: pathlib.Path, task: Task) -> None:
        """
        Clean logs from previous run

        Returns:
            None
        """

        for directory in [
            task.logs_dir,
            task.results_dir,
        ]:
            directory_path = workspace / directory
            clean_directory(directory_path)
            LOGGER.info(f"Location {directory} cleaned.")

    def _prepare_triton_container_image(self, config: Config, maintainer: Maintainer, triton: Triton) -> str:
        """
        Prepare Triton Container Image based on provided configuration

        Returns:
            Name of container image to use in process
        """
        if not config.triton_dockerfile:
            image_name = triton.container_image(config.container_version)
            LOGGER.info(f"Using official Triton container image: {image_name}.")
            return image_name

        if config.triton_container_image:
            LOGGER.info(f"Using provided Triton Container Image: {config.triton_container_image}")
            return config.triton_container_image

        normalized_model_name = config.model_name.lower().replace("_", "-")
        image_name = f"tritonserver-{normalized_model_name}:latest"
        LOGGER.info(f"Building Triton Container Image: {image_name}")

        maintainer.build_image(
            image_name=image_name,
            image_file_path=pathlib.Path(config.triton_dockerfile),
            build_args={"FROM_IMAGE": triton.container_image(container_version=config.container_version)},
        )
        return image_name

    def _download_checkpoints(self, task: Task) -> None:
        """
        Download checkpoints
        """
        for variant, checkpoint in task.checkpoints.items():
            checkpoint_url = checkpoint.url
            download_path = checkpoint.path

            if download_path.is_dir():
                LOGGER.info(f"Checkpoint {download_path.name} already downloaded.")
                continue

            if not checkpoint_url:
                LOGGER.warning(
                    f"Checkpoint {variant} url is not provided."
                    "\nIf you want to use that checkpoint please train the model locally"
                    f"\nand copy to {download_path} directory"
                )
                continue

            download(checkpoint_url, download_path)

    def _initialize_task(
        self,
        workspace: pathlib.Path,
        config: Config,
        pipeline: Pipeline,
        triton_container_image: str,
        logs_dir: pathlib.Path,
    ) -> Task:
        """
        Initialize task object

        Args:
            workspace: Path to workspace where artifacts are stored
            config: Config object
            pipeline: Pipeline object
            triton_container_image: Triton Inference Server container image used for tests

        Returns:
            Task object
        """
        datasets = {}
        for dataset in config.datasets:
            datasets[dataset.name] = Dataset(name=dataset.name)

        checkpoints = {}
        for checkpoint in config.checkpoints:
            download_path = workspace / Task.checkpoints_dir / checkpoint.name
            checkpoints[checkpoint.name] = Checkpoint(name=checkpoint.name, url=checkpoint.url, path=download_path)

        results_types = self._task_results_types(pipeline=pipeline)

        stages = {}
        for stage in pipeline.stages():
            stages[stage.label] = {"result_path": stage.result_path, "result_type": stage.result_type}

        experiments = []
        for idx, configuration in enumerate(config.configurations, start=1):
            experiment = self._prepare_experiment(
                idx=idx,
                configuration=configuration,
                results_types=results_types,
                stages=stages,
            )
            experiments.append(experiment)

        system_info = SystemInfo.from_host()

        task = Task(
            model_name=config.model_name,
            ensemble_model_name=config.ensemble_model_name,
            framework=config.framework,
            checkpoints=checkpoints,
            datasets=datasets,
            datasets_dir=config.datasets_dir,
            experiments=experiments,
            container_version=config.container_version,
            system_info=system_info,
            triton_container_image=triton_container_image,
            triton_custom_operations=config.triton_custom_operations,
            triton_load_model_method=config.triton_load_model_method,
            started_at=int(datetime.utcnow().timestamp()),
            batching=config.batching,
            measurement_steps_offline=config.measurement_steps_offline,
            measurement_steps_online=config.measurement_steps_online,
        )
        return task

    def _task_results_types(self, pipeline: Pipeline) -> List[str]:
        """
        Types of results generated as part of task

        Returns:
            List of result types
        """
        results = []
        for stage in pipeline.stages():
            if TritonPerformanceOfflineStage.label == stage.label:
                results.append(ResultsType.TRITON_PERFORMANCE_OFFLINE)
                continue

            if TritonPerformanceOnlineStage.label == stage.label:
                results.append(ResultsType.TRITON_PERFORMANCE_ONLINE)
                continue

        return results

    def _prepare_experiment(
        self,
        idx: int,
        configuration: Configuration,
        results_types: List[str],
        stages: Dict,
    ) -> Experiment:
        """
        Prepare experiments data

        Args:
            idx: Experiment index
            configuration: Configuration object
            results_types: Results types stored in experiment
            stages: Stages executed as part of experiment

        Returns:
            Experiment object
        """
        results_mapped = {}
        for result_type in results_types:
            results_mapped[result_type] = result_type

        stages_mapped = {}
        for name, stage_data in stages.items():
            stages_mapped[name] = Stage(name=name, **stage_data)

        experiment = Experiment(
            experiment_id=idx,
            parameters=configuration.parameters,
            stages=stages_mapped,
            results=results_mapped,
            checkpoint=configuration.checkpoint,
        )

        return experiment
