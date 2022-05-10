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

import functools
import logging
import queue
import threading
from pathlib import Path
from typing import Optional

# pytype: disable=import-error
try:
    from tritonclient import utils as client_utils  # noqa: F401
except ImportError:
    import tritonclientutils as client_utils  # noqa: F401

try:
    import tritonclient.grpc as grpc_client
except ImportError:
    import tritongrpcclient as grpc_client
# pytype: enable=import-error

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = Path(__file__).parent.name

from .base import BaseRunner

LOGGER = logging.getLogger("triton_inference_runner.grpc")


class SyncInferenceRunner(BaseRunner):
    def __iter__(self):
        LOGGER.debug(f"Connecting to {self._server_url}")
        client = grpc_client.InferenceServerClient(url=self._server_url, verbose=self._verbose)

        error = self._verify_triton_state(client)
        if error:
            raise RuntimeError(f"Could not communicate to Triton Server: {error}")

        LOGGER.debug(
            f"Triton server {self._server_url} and model {self._model_name}:{self._model_version} " f"are up and ready!"
        )

        model_config = client.get_model_config(self._model_name, self._model_version)
        model_metadata = client.get_model_metadata(self._model_name, self._model_version)
        LOGGER.info(f"Model config {model_config}")
        LOGGER.info(f"Model metadata {model_metadata}")

        inputs = {tm.name: tm for tm in model_metadata.inputs}
        outputs = {tm.name: tm for tm in model_metadata.outputs}
        output_names = list(outputs)
        outputs_req = [grpc_client.InferRequestedOutput(name) for name in outputs]

        for ids, x, y_real in self._dataloader:
            infer_inputs = []
            for name in inputs:
                data = x[name]
                datatype = inputs[name].datatype
                infer_input = grpc_client.InferInput(name, data.shape, datatype)

                target_np_dtype = client_utils.triton_to_np_dtype(datatype)
                data = data.astype(target_np_dtype)

                infer_input.set_data_from_numpy(data)
                infer_inputs.append(infer_input)

            results = client.infer(
                model_name=self._model_name,
                model_version=self._model_version,
                inputs=infer_inputs,
                outputs=outputs_req,
                timeout=self._response_wait_t,
            )
            y_pred = {name: results.as_numpy(name) for name in output_names}
            yield ids, x, y_pred, y_real


class AsyncInferenceRunner(BaseRunner):
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

        self._results = queue.Queue()
        self._processed_all = False
        self._errors = []
        self._num_waiting_for = 0
        self._sync = threading.Condition()
        self._req_thread = threading.Thread(target=self.req_loop, daemon=True)

    def __iter__(self):
        self._req_thread.start()
        timeout_s = 0.050  # check flags processed_all and error flags every 50ms
        while True:
            try:
                ids, x, y_pred, y_real = self._results.get(timeout=timeout_s)
                yield ids, x, y_pred, y_real
            except queue.Empty:
                shall_stop = self._processed_all or self._errors
                if shall_stop:
                    break

        LOGGER.debug("Waiting for request thread to stop")
        self._req_thread.join()
        if self._errors:
            error_msg = "\n".join(map(str, self._errors))
            raise RuntimeError(error_msg)

    def _on_result(self, ids, x, y_real, output_names, result, error):
        with self._sync:
            request_id = str(ids[0])
            NOT_MATCHING_REQUEST_ID_MSG = (
                "Error during processing result - request_id doesn't match. This shouldn't have happened."
            )
            if error:
                response_id = error.get_response().id
                if response_id != request_id:
                    raise RuntimeError(NOT_MATCHING_REQUEST_ID_MSG)
                self._errors.append(error)
            else:
                response_id = result.get_response().id
                if response_id != request_id:
                    raise RuntimeError(NOT_MATCHING_REQUEST_ID_MSG)
                y_pred = {name: result.as_numpy(name) for name in output_names}
                self._results.put((ids, x, y_pred, y_real))
            self._num_waiting_for -= 1
            self._sync.notify_all()

    def req_loop(self):
        LOGGER.debug(f"Connecting to {self._server_url}")
        client = grpc_client.InferenceServerClient(url=self._server_url, verbose=self._verbose)

        self._errors = self._verify_triton_state(client)
        if self._errors:
            return

        LOGGER.debug(
            f"Triton server {self._server_url} and model {self._model_name}:{self._model_version} " f"are up and ready!"
        )

        model_config = client.get_model_config(self._model_name, self._model_version)
        model_metadata = client.get_model_metadata(self._model_name, self._model_version)
        LOGGER.info(f"Model config {model_config}")
        LOGGER.info(f"Model metadata {model_metadata}")

        inputs = {tm.name: tm for tm in model_metadata.inputs}
        outputs = {tm.name: tm for tm in model_metadata.outputs}
        output_names = list(outputs)

        self._num_waiting_for = 0

        for ids, x, y_real in self._dataloader:
            infer_inputs = []
            for name in inputs:
                data = x[name]
                datatype = inputs[name].datatype
                infer_input = grpc_client.InferInput(name, data.shape, datatype)

                target_np_dtype = client_utils.triton_to_np_dtype(datatype)
                data = data.astype(target_np_dtype)

                infer_input.set_data_from_numpy(data)
                infer_inputs.append(infer_input)

            outputs_req = [grpc_client.InferRequestedOutput(name) for name in outputs]

            with self._sync:

                def _check_can_send():
                    return self._num_waiting_for < self._max_unresp_reqs

                can_send = self._sync.wait_for(_check_can_send, timeout=self._response_wait_t)
                if not can_send:
                    error_msg = f"Runner could not send new requests for {self._response_wait_t}s"
                    self._errors.append(error_msg)
                    self._sync.notify_all()
                    break

                request_id = str(ids[0])
                callback = functools.partial(AsyncInferenceRunner._on_result, self, ids, x, y_real, output_names)
                client.async_infer(
                    model_name=self._model_name,
                    model_version=self._model_version,
                    inputs=infer_inputs,
                    outputs=outputs_req,
                    callback=callback,
                    request_id=request_id,
                )
                self._num_waiting_for += 1
                self._sync.notify_all()

        # wait till receive all requested data
        with self._sync:

            def _all_processed():
                LOGGER.debug(f"wait for {self._num_waiting_for} unprocessed jobs")
                return self._num_waiting_for == 0

            self._processed_all = self._sync.wait_for(_all_processed, self._max_wait_time)
            if not self._processed_all:
                error_msg = f"Runner {self._response_wait_t}s timeout received while waiting for results from server"
                self._errors.append(error_msg)

            self._sync.notify_all()

        LOGGER.debug("Finished request thread")
