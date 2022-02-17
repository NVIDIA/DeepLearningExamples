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
import pathlib
from threading import Thread
from typing import Dict, Generator, Union

from docker.models.containers import ExecResult
from docker.types import DeviceRequest, Ulimit

if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from ....logger import LOGGER
from ...exceptions import ContainerNotStarted
from ..container import DockerContainer


class TritonServerContainer(DockerContainer):
    def __init__(
        self,
        name: str,
        command: str,
        image: str,
        volumes: Dict,
        devices: Union[list, int],
        environment: Dict,
        log_file: Union[pathlib.Path, str],
        network: str = "host",
        shm_size: str = "1G",
    ):
        """
        Initialize Triton Server Container
        Args:
            name: Container name
            command: Triton Server command to exec on container start
            image: Docker Image
            volumes: Volumes to mount inside container
            devices: Devices which has to be visible in container
            environment: Environment variables
            log_file: Path where logs should be saved
            network: Network mode
            shm_size: Shared memory size
        """
        super().__init__(name)
        self._image = image
        self._command = command
        self._volumes = volumes
        self._devices = devices
        self._environment = environment
        self._network = network
        self._shm_size = shm_size
        self._triton_exec = None
        self._logging_thread = None
        self._log_file_path = pathlib.Path(log_file)

    def start(self) -> None:
        """
        Start Triton Server Container
        """
        devices = [
            DeviceRequest(capabilities=[["gpu"]], device_ids=self._devices),
        ]

        LOGGER.info(f"Triton environment: {json.dumps(self._environment, indent=4)}")

        LOGGER.info(f"Starting Triton container {self.name}.")
        self._container = self._docker_client.containers.run(
            image=self._image,
            name=self.name,
            device_requests=devices,
            detach=True,
            tty=True,
            shm_size=self._shm_size,
            ulimits=[
                Ulimit(name="memlock", soft=-1, hard=-1),
                Ulimit(name="stack", soft=67108864, hard=67108864),
            ],
            volumes=self._volumes,
            environment=self._environment,
            network_mode=self._network,
            auto_remove=True,
            ipc_mode="host",
        )
        LOGGER.info(f"Triton command:")
        LOGGER.info(f"  {self._command}")
        LOGGER.info(f"Starting Triton Server {self.name}.")
        self._triton_exec = self._docker_api_client.exec_create(
            container=self._container.id,
            cmd=self._command,
        )
        stream_generator = self._docker_api_client.exec_start(exec_id=self._triton_exec["Id"], stream=True)

        self._logging_thread = Thread(target=TritonServerContainer._logging, args=(self, stream_generator), daemon=True)
        self._logging_thread.start()

    def stop(self) -> None:
        """
        Stop Triton Server Container and save logs to file
        """
        if self._container is not None:
            triton_result = self._docker_api_client.exec_inspect(self._triton_exec["Id"])
            if triton_result.get("ExitCode") not in (0, None):
                LOGGER.info(
                    f"Triton Inference Server instance {self.name} failed. Exit code: {triton_result.get('ExitCode')}"
                )

            LOGGER.info(f"Stopping triton server {self.name}.")
            self._container.stop()

            self._container = None
            self._docker_client.close()
            self._docker_api_client.close()

    def run(self, command: str) -> ExecResult:
        """
        Run command in container
        Args:
            command: Command to execute

        Returns:
            ExecResult
        """
        if not self._container:
            raise ContainerNotStarted("Triton Server Container is not running. Use .start() first.")

        return self._container.exec_run(command)

    def _logging(self, generator: Generator) -> None:
        """Triton logging thread for Triton Inference Server

        Args:
            generator (string generator): Triton log stream.
        """
        with open(self._log_file_path, mode="w") as file:
            try:
                while True:
                    log = next(generator)
                    txt = log.decode("utf-8")
                    file.write(txt)
            except StopIteration:
                LOGGER.info(f"Saving Triton Inference Server {self.name} logs in {self._log_file_path}.")
