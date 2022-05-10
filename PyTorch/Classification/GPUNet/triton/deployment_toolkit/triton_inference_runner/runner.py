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
from pathlib import Path
from typing import Optional

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = Path(__file__).parent.name

from ..utils import TritonClientProtocol, parse_server_url
from .grpc import AsyncInferenceRunner as AsyncGRPCRunner
from .grpc import SyncInferenceRunner as SyncGRPCRunner
from .http import AsyncInferenceRunner as AsyncHTPPRunner
from .http import SyncInferenceRunner as SyncHTTPRunner


class TritonInferenceRunner:

    async_runners = {
        TritonClientProtocol.GRPC: AsyncGRPCRunner,
        TritonClientProtocol.HTTP: AsyncHTPPRunner,
    }

    sync_runners = {
        TritonClientProtocol.GRPC: SyncGRPCRunner,
        TritonClientProtocol.HTTP: SyncHTTPRunner,
    }

    def __init__(
        self,
        server_url: str,
        model_name: str,
        model_version: str,
        dataloader_fn,
        verbose: bool = False,
        response_wait_time: Optional[float] = None,
        max_unresponded_requests: int = 128,
        synchronous: bool = False,
    ):

        protocol, host, port = parse_server_url(server_url)
        server_url = f"{host}:{port}"

        if synchronous:
            sync_runner_cls = TritonInferenceRunner.sync_runners[protocol]
            self._runner = sync_runner_cls(
                server_url,
                model_name,
                model_version,
                dataloader=dataloader_fn(),
                verbose=verbose,
                response_wait_time=response_wait_time,
            )
        else:
            async_runner_cls = TritonInferenceRunner.async_runners[protocol]
            self._runner = async_runner_cls(
                server_url,
                model_name,
                model_version,
                dataloader=dataloader_fn(),
                verbose=verbose,
                response_wait_time=response_wait_time,
                max_unresponded_requests=max_unresponded_requests,
            )

    def __iter__(self):
        return self._runner.__iter__()
