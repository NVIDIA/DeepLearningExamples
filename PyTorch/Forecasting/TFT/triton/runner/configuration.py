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
import pathlib
from typing import Any, Dict, List, Union

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .task import DataObject


class Configuration(DataObject):
    """
    Configuration object - handle single experiment data
    """

    def __init__(
        self,
        precision: str,
        format: str,
        batch_size: Union[str, List],
        accelerator: str,
        triton_gpu_engine_count: int,
        triton_max_queue_delay: int,
        capture_cuda_graph: int,
        checkpoint_variant: str,
        triton_preferred_batch_sizes: Union[str, List],
        **kwargs: Any,
    ):
        """

        Args:
            precision: Target model precision
            format: Target conversion format
            batch_size: Batch sizes to evaluate
            accelerator: Triton Backend Accelerator
            triton_gpu_engine_count: Number of model instances
            triton_max_queue_delay: Maximal queue delay
            capture_cuda_graph: Triton Capture CUDA Graph optimization for tensorrt
            checkpoint_variant: Checkpoint used for configuration
            triton_preferred_batch_sizes: Preferred batch sizes
            **kwargs: Additional model arguments
        """
        if isinstance(batch_size, str):
            batch_size = map(lambda item: int(item), batch_size.split(","))

        if isinstance(triton_preferred_batch_sizes, str):
            triton_preferred_batch_sizes = map(lambda item: int(item), triton_preferred_batch_sizes.split(" "))

        self.precision = precision
        self.format = format
        self.batch_size = sorted(batch_size)
        self.accelerator = accelerator
        self.triton_gpu_engine_count = triton_gpu_engine_count
        self.triton_max_queue_delay = triton_max_queue_delay
        self.capture_cuda_graph = capture_cuda_graph
        self.max_batch_size = max(self.batch_size)
        self.checkpoint_variant = checkpoint_variant
        self.triton_preferred_batch_sizes = " ".join(map(lambda i: str(i), sorted(triton_preferred_batch_sizes)))

        for key, value in kwargs.items():
            self.__setattr__(key, value)

    @property
    def parameters(self) -> Dict:
        """
        Return values stored in configuration

        Returns:
            Dictionary with configuration parameters
        """
        return self.__dict__
