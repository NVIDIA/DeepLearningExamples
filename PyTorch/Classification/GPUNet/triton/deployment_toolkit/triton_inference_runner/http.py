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

import json
import logging
from pathlib import Path
from typing import Optional

# pytype: disable=import-error
try:
    from tritonclient import utils as client_utils  # noqa: F401
except ImportError:
    import tritonclientutils as client_utils  # noqa: F401

try:
    import tritonclient.http as http_client
except (ImportError, RuntimeError):
    import tritonhttpclient as http_client
# pytype: enable=import-error

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = Path(__file__).parent.name

from .base import BaseRunner

LOGGER = logging.getLogger("triton_inference_runner.http")


class HTTPInferenceRunner(BaseRunner):
    def _parse_content(self, response):
        return json.dumps(response, indent=4)


class SyncInferenceRunner(HTTPInferenceRunner):
    def __iter__(self):
        LOGGER.debug(f"Connecting to {self._server_url}")
        client = http_client.InferenceServerClient(
            url=self._server_url,
            verbose=self._verbose,
            connection_timeout=self._response_wait_t,
            network_timeout=self._response_wait_t,
        )

        error = self._verify_triton_state(client)
        if error:
            raise RuntimeError(f"Could not communicate to Triton Server: {error}")

        LOGGER.debug(
            f"Triton server {self._server_url} and model {self._model_name}:{self._model_version} " f"are up and ready!"
        )

        model_config = client.get_model_config(self._model_name, self._model_version)
        model_metadata = client.get_model_metadata(self._model_name, self._model_version)
        LOGGER.info(f"Model config {self._parse_content(model_config)}")
        LOGGER.info(f"Model metadata {self._parse_content(model_metadata)}")

        inputs = {tm["name"]: tm for tm in model_metadata["inputs"]}
        outputs = {tm["name"]: tm for tm in model_metadata["outputs"]}
        output_names = list(outputs)
        outputs_req = [http_client.InferRequestedOutput(name) for name in outputs]

        for ids, x, y_real in self._dataloader:
            infer_inputs = []
            for name in inputs:
                data = x[name]
                datatype = inputs[name]["datatype"]
                infer_input = http_client.InferInput(name, data.shape, datatype)

                target_np_dtype = client_utils.triton_to_np_dtype(datatype)
                data = data.astype(target_np_dtype)

                infer_input.set_data_from_numpy(data)
                infer_inputs.append(infer_input)

            results = client.infer(
                model_name=self._model_name,
                model_version=self._model_version,
                inputs=infer_inputs,
                outputs=outputs_req,
                timeout=self._response_wait_t_ms,
            )
            y_pred = {name: results.as_numpy(name) for name in output_names}
            yield ids, x, y_pred, y_real


class AsyncInferenceRunner(HTTPInferenceRunner):
    DEFAULT_MAX_UNRESP_REQS = 128

    def __init__(
        self,
        server_url: str,
        model_name: str,
        model_version: str,
        *,
        dataloader,
        verbose=False,
        response_wait_time: Optional[float] = None,
        max_unresponded_requests: Optional[int] = None,
    ):
        super().__init__(
            server_url,
            model_name,
            model_version,
            dataloader=dataloader,
            verbose=verbose,
            response_wait_time=response_wait_time,
        )
        self._max_unresp_reqs = (
            self.DEFAULT_MAX_UNRESP_REQS if max_unresponded_requests is None else max_unresponded_requests
        )

    def __iter__(self):
        client = http_client.InferenceServerClient(
            url=self._server_url,
            verbose=self._verbose,
            concurrency=self._max_unresp_reqs,
            connection_timeout=self._response_wait_t,
            network_timeout=self._response_wait_t,
        )

        self._errors = self._verify_triton_state(client)
        if self._errors:
            return

        LOGGER.debug(
            f"Triton server {self._server_url} and model {self._model_name}:{self._model_version} " f"are up and ready!"
        )

        model_config = client.get_model_config(self._model_name, self._model_version)
        model_metadata = client.get_model_metadata(self._model_name, self._model_version)
        LOGGER.info(f"Model config {self._parse_content(model_config)}")
        LOGGER.info(f"Model metadata {self._parse_content(model_metadata)}")

        inputs = {tm["name"]: tm for tm in model_metadata["inputs"]}
        outputs = {tm["name"]: tm for tm in model_metadata["outputs"]}
        output_names = list(outputs)

        async_requests = []
        for ids, x, y_real in self._dataloader:
            infer_inputs = []
            for name in inputs:
                data = x[name]
                datatype = inputs[name]["datatype"]
                infer_input = http_client.InferInput(name, data.shape, datatype)

                target_np_dtype = client_utils.triton_to_np_dtype(datatype)
                data = data.astype(target_np_dtype)

                infer_input.set_data_from_numpy(data)
                infer_inputs.append(infer_input)

            outputs_req = [http_client.InferRequestedOutput(name) for name in outputs]

            request_id = str(ids[0])
            async_request = client.async_infer(
                model_name=self._model_name,
                model_version=self._model_version,
                inputs=infer_inputs,
                outputs=outputs_req,
                request_id=request_id,
                timeout=self._response_wait_t_ms,
            )
            async_requests.append((ids, x, y_real, async_request))

            if len(async_requests) > self._max_unresp_reqs:
                yield from self._yield_response(async_requests, output_names)
                async_requests = []

        yield from self._yield_response(async_requests, output_names)

        LOGGER.debug("Finished request thread")

    def _yield_response(self, async_requests, output_names):
        for ids, x, y_real, async_response in async_requests:
            result = async_response.get_result()
            y_pred = {name: result.as_numpy(name) for name in output_names}

            yield ids, x, y_pred, y_real
