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

import pathlib
import platform
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Union

import cpuinfo
import psutil
import yaml

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from ..deployment_toolkit.core import PerformanceTool
from .core import CustomDumper, DataObject
from .experiment import Experiment
from .triton import Triton


class GPU(DataObject):
    """
    GPU information data object
    """

    name: str
    driver_version: str
    cuda_version: str
    memory: str
    tdp: str

    def __init__(self, name: str, driver_version: str, cuda_version: str, memory: str, tdp: str):
        """
        Args:
            name: name of GPU
            driver_version: version of driver
            cuda_version: version of CUDA
            memory: size of memory available on GPU [MB]
            tdp: Max TDP of GPU unit
        """
        self.name = name
        self.driver_version = driver_version
        self.cuda_version = cuda_version
        self.memory = memory
        self.tdp = tdp

    @staticmethod
    def from_dict(data: Dict):
        """
        Create GPU object from dictionary

        Args:
            data: dictionary with GPU data

        Returns:
            GPU object
        """
        return GPU(
            name=data["name"],
            driver_version=data["driver_version"],
            cuda_version=data["cuda_version"],
            memory=data["memory"],
            tdp=data["tdp"],
        )

    @staticmethod
    def from_host():
        """
        Create GPU object from host data

        Returns:
            GPU object
        """
        data = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.max_limit", "--format=csv"]
        ).decode()

        lines = data.split(sep="\n")
        device_details = lines[1].split(",")
        name = device_details[0].strip()
        driver_version = device_details[1].strip()
        memory = device_details[2].strip()
        tdp = device_details[3].strip()
        cuda_version = None

        data = subprocess.check_output(["nvidia-smi", "--query"]).decode()
        lines = data.split(sep="\n")
        for line in lines:
            if line.startswith("CUDA Version"):
                cuda_version = line.split(":")[1].strip()
                break

        return GPU(
            name=name,
            driver_version=driver_version,
            cuda_version=cuda_version,
            memory=memory,
            tdp=tdp,
        )


class CPU(DataObject):
    """
    CPU details
    """

    name: str
    physical_cores: int
    logical_cores: int
    min_frequency: float
    max_frequency: float

    def __init__(self, name: str, physical_cores: int, logical_cores: int, min_frequency: float, max_frequency: float):
        """
        Args:
            name: name of CPU unit
            physical_cores: number of physical cores available on CPU
            logical_cores: number of logical cores available on CPU
            min_frequency: minimal clock frequency
            max_frequency: maximal clock frequency
        """
        self.name = name
        self.physical_cores = physical_cores
        self.logical_cores = logical_cores
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

    @staticmethod
    def from_host():
        """
        Create CPU object from host data

        Returns:
            CPU object
        """
        return CPU(
            name=cpuinfo.get_cpu_info()["brand_raw"],
            physical_cores=psutil.cpu_count(logical=False),
            logical_cores=psutil.cpu_count(logical=True),
            min_frequency=psutil.cpu_freq().min,
            max_frequency=psutil.cpu_freq().max,
        )


class Memory(DataObject):
    """
    Memory data object
    """

    size: float

    def __init__(self, size: float):
        """
        Args:
            size: RAM memory size in MB
        """
        self.size = size

    @staticmethod
    def from_host():
        """
        Create Memory object from host data

        Returns:
            Memory object
        """
        svm = psutil.virtual_memory()
        return Memory(size=svm.total)


class SystemInfo(DataObject):
    """
    System Information data object
    """

    system: str
    cpu: CPU
    memory: Memory
    gpu: GPU

    def __init__(self, system: str, cpu: CPU, memory: Memory, gpu: GPU):
        """
        Args:
            system: name of operating system
            cpu: CPU info
            memory: Memory info
            gpu: GPU info
        """
        self.system = system
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu

    @staticmethod
    def from_host():
        """
        Create SystemInfo object from host data

        Returns:
            SystemInfo object
        """
        system = platform.platform()
        gpu = GPU.from_host()
        memory = Memory.from_host()
        cpu = CPU.from_host()

        return SystemInfo(system=system, cpu=cpu, gpu=gpu, memory=memory)


class Checkpoint(DataObject):
    """
    Checkpoint data object
    """

    def __init__(self, name: str, url: str, path: Union[str, pathlib.Path]):
        """
        Args:
            name: Name of checkpoint
            path: Location of checkpoint on local hardware
        """
        self.name = name
        self.url = url
        self.path = pathlib.Path(path)


class Dataset(DataObject):
    """
    Dataset data object
    """

    def __init__(self, name: str):
        """
        Args:
            name: Name of dataset
        """
        self.name = name


class Task(DataObject):
    """
    Task data object to store build information
    """

    model_name: str
    framework: str
    batching: str
    started_at: int
    ended_at: Optional[int]
    container_version: str
    checkpoints: Dict[str, Checkpoint]
    datasets: Dict[str, Dataset]
    datasets_dir: Optional[Union[str, pathlib.Path]]
    experiments: List[Experiment]
    system_info: SystemInfo
    triton_container_image: Optional[str]
    triton_custom_operations: Optional[str]
    performance_tool: PerformanceTool

    filename: str = "task.yaml"
    results_dir: str = "results"
    checkpoints_dir: str = "checkpoints"

    def __init__(
        self,
        model_name: str,
        ensemble_model_name: Optional[str],
        framework: str,
        batching: str,
        container_version: str,
        checkpoints: Dict,
        datasets: Dict,
        experiments: List,
        system_info: SystemInfo,
        started_at: int,
        datasets_dir: Optional[Union[str, pathlib.Path]] = None,
        ended_at: Optional[int] = None,
        triton_container_image: Optional[str] = None,
        triton_custom_operations: Optional[str] = None,
        triton_load_model_method: str = Triton.LOAD_MODE.EXPLICIT,
        measurement_steps_offline: int = 8,
        measurement_steps_online: int = 32,
        performance_tool: PerformanceTool = PerformanceTool.MODEL_ANALYZER,
    ):
        """

        Args:
            model_name: Name of model
            framework: Model framework
            container_version: Container version used in task
            checkpoints: List of checkpoints
            datasets: List of datasets
            datasets_dir: Directory where datasests are stored
            experiments: List of experiments run as part of task
            system_info: information about node on which experiment was executed
            started_at: Time when task has started
            ended_at: Time when task has ended
            triton_container_image: Custom Triton Container Image used for task
            triton_custom_operations: Custom operation library path
            triton_load_model_method: Method how models are loaded on Triton
            measurement_steps_offline: Number of measurement steps in offline performance stage
            measurement_steps_online: Number of measurement steps in online performance stage
            performance_tool: Performance Tool used for generating results
        """
        self.started_at = started_at
        self.ended_at = ended_at

        self.model_name = model_name
        self.ensemble_model_name = ensemble_model_name
        self.framework = framework
        self.container_version = container_version
        self.checkpoints = checkpoints
        self.datasets = datasets
        self.datasets_dir = pathlib.Path(datasets_dir)
        self.experiments = experiments
        self.system_info = system_info

        self.triton_container_image = triton_container_image
        self.triton_custom_operations = triton_custom_operations
        self.triton_load_model_method = triton_load_model_method

        self.measurement_steps_offline = measurement_steps_offline
        self.measurement_steps_online = measurement_steps_online

        self.logs_dir = pathlib.Path("/var/logs")

        self.batching = batching
        self.performance_tool = performance_tool

    def start(self) -> None:
        """
        Update stage execution info at start

        Returns:
            None
        """
        self.started_at = int(datetime.utcnow().timestamp())

    def end(self) -> None:
        """
        Update stage execution info at end

        Returns:
            None
        """
        self.ended_at = int(datetime.utcnow().timestamp())

    def to_file(self, file_path: Union[pathlib.Path, str]):
        """
        Store task data to YAML file

        Args:
            file_path: path to file where task data has to be saved

        Returns:
            None
        """
        task_data = self.to_dict()
        with open(file_path, "w") as f:
            yaml.dump(task_data, f, Dumper=CustomDumper, width=240, sort_keys=False)
