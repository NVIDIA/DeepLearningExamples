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

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .core import Framework, Paths


class Triton:
    """
    Triton Inference Server helper class
    """

    image = "nvcr.io/nvidia/tritonserver"
    tag = "py3"

    class LOAD_MODE:
        """
        Loading mode available in Triton
        """

        POLL = "poll"
        EXPLICIT = "explicit"

    @staticmethod
    def container_image(container_version: str):
        """
        Container image based on version

        Args:
            container_version: Version of container to be used

        Returns:
            Image name with tag
        """
        return f"{Triton.image}:{container_version}-{Triton.tag}"

    @staticmethod
    def command(
        framework: str,
        repository_path: str,
        strict_mode: bool = False,
        poll_model: bool = False,
        metrics: bool = False,
        verbose: bool = False,
    ):
        """
        Command to run Triton Inference Server inside container
        Args:
            framework: Framework used for model
            repository_path: Path to model repository
            strict_mode: Flag to use strict model config
            poll_model: Poll model
            metrics: Enable GPU metrics (disable for MIG)
            verbose: Use verbose mode logging

        Returns:

        """
        triton_command = f"tritonserver --model-store={repository_path}"
        if poll_model:
            triton_command += " --model-control-mode=poll --repository-poll-secs 5"
        else:
            triton_command += " --model-control-mode=explicit"

        if not strict_mode:
            triton_command += " --strict-model-config=false"

        if not metrics:
            triton_command += " --allow-metrics=false --allow-gpu-metrics=false"

        if verbose:
            triton_command += " --log-verbose 1"

        if framework in (Framework.TensorFlow1, Framework.TensorFlow2):
            version = 1 if framework == Framework.TensorFlow1 else 2
            triton_command += f" --backend-config=tensorflow,version={version}"

        return triton_command

    @staticmethod
    def library_path(framework: str):
        """
        Obtain custom library path for framework

        Args:
            framework: Framework used for model

        Returns:
            Path to additional libraries needed by framework
        """
        paths = {
            Framework.PyTorch.name: "/opt/tritonserver/backends/pytorch",
            Framework.TensorFlow1.name: "/opt/tritonserver/backends/tensorflow1",
            Framework.TensorFlow2.name: "/opt/tritonserver/backends/tensorflow2",
        }

        return paths[framework]

    @staticmethod
    def custom_library_path_remote() -> str:
        """
        Path to custom library mounted in Triton container

        Returns:
            Path to shared library with custom operations
        """
        return f"{Paths.LIBRARIES_PATH}/libcustomops.so"

    @staticmethod
    def custom_library_path_local(libs_dir: pathlib.Path) -> pathlib.Path:
        """
        Path to custom library in local path

        Args:
            libs_dir: path to libraries directory

        Returns:
            Path to shared library with custom operations
        """
        return libs_dir / "libcustomops.so"
