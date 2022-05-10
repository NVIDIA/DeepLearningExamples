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
import logging
from typing import Optional

LOGGER = logging.getLogger("triton_inference_runner.base")


class BaseRunner:
    DEFAULT_MAX_RESP_WAIT_S = 120
    DEFAULT_MAX_FINISH_WAIT_S = 900  # 15min

    def __init__(
        self,
        server_url: str,
        model_name: str,
        model_version: str,
        *,
        dataloader,
        verbose=False,
        response_wait_time: Optional[float] = None,
    ):
        self._model_name = model_name
        self._model_version = model_version
        self._dataloader = dataloader
        self._verbose = verbose
        self._response_wait_t = int(self.DEFAULT_MAX_RESP_WAIT_S if response_wait_time is None else response_wait_time)
        self._response_wait_t_ms = self._response_wait_t * 1000 * 1000
        self._max_wait_time = max(self._response_wait_t, self.DEFAULT_MAX_FINISH_WAIT_S)
        self._server_url = server_url

    def _verify_triton_state(self, triton_client):
        errors = []
        if not triton_client.is_server_live():
            errors.append(f"Triton server {self._server_url} is not live")
        elif not triton_client.is_server_ready():
            errors.append(f"Triton server {self._server_url} is not ready")
        elif not triton_client.is_model_ready(self._model_name, self._model_version):
            errors.append(f"Model {self._model_name}:{self._model_version} is not ready")
        return errors
