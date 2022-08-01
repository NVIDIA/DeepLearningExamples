# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import pathlib
from typing import Dict, List, Optional, Union

import yaml

if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .configuration import Configuration
from .core import DataObject
from .triton import Triton


class Checkpoint(DataObject):
    """
    Checkpoint data placeholder
    """

    name: str
    url: str

    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url


class Dataset(DataObject):
    """
    Dataset data placeholder
    """

    name: str

    def __init__(self, name: str):
        self.name = name


class Config(DataObject):
    """
    Configuration object for runner experiments
    """

    def __init__(
        self,
        model_name: str,
        framework: str,
        container_version: str,
        configurations: List[Configuration],
        datasets_dir: str = "datasets",
        datasets: List[Dataset] = None,
        checkpoints: List[Checkpoint] = None,
        triton_dockerfile: Optional[str] = None,
        triton_container_image: Optional[str] = None,
        triton_custom_operations: Optional[str] = None,
        triton_load_model_method: Optional[str] = Triton.LOAD_MODE.EXPLICIT,
    ):
        """

        Args:
            model_name: Name of model
            framework: Framework used to create model
            container_version: Version of Triton Inference Server container used for evaluation
            configurations: List of experiments configurations
            datasets_dir: Directory where datasets are stored
            datasets: Datasets used for conversion/export
            checkpoints: Checkpoints with trained model
            triton_load_model_method: Triton Inference Server model loading mode
            triton_dockerfile: Dockerfile for Triton to build custom image
            triton_container_image: Custom image used for Triton Server - leave empty to use default or built from Dockerfile
            triton_custom_operations: Path where custom operation library is stored
        """
        self.model_name = model_name
        self.framework = framework
        self.container_version = container_version
        self.configurations = configurations
        self.datasets_dir = datasets_dir
        self.datasets = datasets
        self.checkpoints = checkpoints
        self.triton_load_model_method = triton_load_model_method
        self.triton_dockerfile = triton_dockerfile
        self.triton_container_image = triton_container_image
        self.triton_custom_operations = triton_custom_operations

    def to_file(self, file_path: Union[pathlib.Path, str]) -> None:
        """
        Save config data to file
        Args:
            file_path: path to file where config data is should be stored

        Returns:
            None
        """
        data = self.to_dict()
        with open(file_path, "w") as f:
            yaml.safe_dump(data, f)

    @staticmethod
    def from_dict(config_data: Dict):
        """
        Create configuration object from data stored in dictionary

        Args:
            config_data: dictionary with config data

        Returns:
            Config object
        """
        configurations = []
        for configuration_data in config_data["configurations"]:
            configuration = Configuration(**configuration_data)
            configurations.append(configuration)

        checkpoints = []
        for checkpoint_data in config_data.get("checkpoints", []):
            checkpoint = Checkpoint(
                name=checkpoint_data["name"],
                url=checkpoint_data["url"],
            )
            checkpoints.append(checkpoint)

        datasets = []
        for dataset_data in config_data.get("datasets", []):
            dataset = Dataset(name=dataset_data["name"])
            datasets.append(dataset)

        return Config(
            model_name=config_data["model_name"],
            framework=config_data["framework"],
            container_version=config_data["container_version"],
            configurations=configurations,
            checkpoints=checkpoints,
            datasets=datasets,
            datasets_dir=config_data.get("datasets_dir"),
            triton_load_model_method=config_data["triton_load_model_method"],
            triton_dockerfile=config_data.get("triton_dockerfile"),
            triton_container_image=config_data.get("triton_container_image"),
            triton_custom_operations=config_data.get("triton_custom_operations"),
        )

    @staticmethod
    def from_file(file_path: Union[pathlib.Path, str]):
        """
        Load experiment data from file
        Args:
            file_path: path to file where experiment data is stored

        Returns:
            Experiment object
        """
        with open(file_path, "r") as f:
            config_data = yaml.safe_load(f)

        return Config.from_dict(config_data)
