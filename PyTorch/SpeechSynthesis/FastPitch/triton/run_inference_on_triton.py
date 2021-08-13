#!/usr/bin/env python3

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

r"""
To infer the model deployed on Triton, you can use `run_inference_on_triton.py` script.
It sends a request with data obtained from pointed data loader and dumps received data into npz files.
Those files are stored in directory pointed by `--output-dir` argument.

Currently, the client communicates with the Triton server asynchronously using GRPC protocol.

Example call:

```shell script
python ./triton/run_inference_on_triton.py \
    --server-url localhost:8001 \
    --model-name ResNet50 \
    --model-version 1 \
    --dump-labels \
    --output-dir /results/dump_triton
```
"""

import argparse
import functools
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# pytype: disable=import-error
try:
    from tritonclient import utils as client_utils  # noqa: F401
    from tritonclient.grpc import (
        InferenceServerClient,
        InferInput,
        InferRequestedOutput,
    )
except ImportError:
    import tritongrpcclient as grpc_client
    from tritongrpcclient import (
        InferenceServerClient,
        InferInput,
        InferRequestedOutput,
    )
# pytype: enable=import-error

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = Path(__file__).parent.name

from .deployment_toolkit.args import ArgParserGenerator
from .deployment_toolkit.core import DATALOADER_FN_NAME, load_from_file
from .deployment_toolkit.dump import NpzWriter

LOGGER = logging.getLogger("run_inference_on_triton")


class AsyncGRPCTritonRunner:
    DEFAULT_MAX_RESP_WAIT_S = 120
    DEFAULT_MAX_UNRESP_REQS = 128
    DEFAULT_MAX_FINISH_WAIT_S = 900  # 15min

    def __init__(
        self,
        server_url: str,
        model_name: str,
        model_version: str,
        *,
        dataloader,
        verbose=False,
        resp_wait_s: Optional[float] = None,
        max_unresponded_reqs: Optional[int] = None,
    ):
        self._server_url = server_url
        self._model_name = model_name
        self._model_version = model_version
        self._dataloader = dataloader
        self._verbose = verbose
        self._response_wait_t = self.DEFAULT_MAX_RESP_WAIT_S if resp_wait_s is None else resp_wait_s
        self._max_unresp_reqs = self.DEFAULT_MAX_UNRESP_REQS if max_unresponded_reqs is None else max_unresponded_reqs

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
            if error:
                self._errors.append(error)
            else:
                y_pred = {name: result.as_numpy(name) for name in output_names}
                self._results.put((ids, x, y_pred, y_real))
            self._num_waiting_for -= 1
            self._sync.notify_all()

    def req_loop(self):
        client = InferenceServerClient(self._server_url, verbose=self._verbose)
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
        outputs_req = [InferRequestedOutput(name) for name in outputs]

        self._num_waiting_for = 0

        for ids, x, y_real in self._dataloader:
            infer_inputs = []
            for name in inputs:
                data = x[name]
                infer_input = InferInput(name, data.shape, inputs[name].datatype)

                target_np_dtype = client_utils.triton_to_np_dtype(inputs[name].datatype)
                data = data.astype(target_np_dtype)

                infer_input.set_data_from_numpy(data)
                infer_inputs.append(infer_input)

            with self._sync:

                def _check_can_send():
                    return self._num_waiting_for < self._max_unresp_reqs

                can_send = self._sync.wait_for(_check_can_send, timeout=self._response_wait_t)
                if not can_send:
                    error_msg = f"Runner could not send new requests for {self._response_wait_t}s"
                    self._errors.append(error_msg)
                    break

                callback = functools.partial(AsyncGRPCTritonRunner._on_result, self, ids, x, y_real, output_names)
                client.async_infer(
                    model_name=self._model_name,
                    model_version=self._model_version,
                    inputs=infer_inputs,
                    outputs=outputs_req,
                    callback=callback,
                )
                self._num_waiting_for += 1

        # wait till receive all requested data
        with self._sync:

            def _all_processed():
                LOGGER.debug(f"wait for {self._num_waiting_for} unprocessed jobs")
                return self._num_waiting_for == 0

            self._processed_all = self._sync.wait_for(_all_processed, self.DEFAULT_MAX_FINISH_WAIT_S)
            if not self._processed_all:
                error_msg = f"Runner {self._response_wait_t}s timeout received while waiting for results from server"
                self._errors.append(error_msg)
        LOGGER.debug("Finished request thread")

    def _verify_triton_state(self, triton_client):
        errors = []
        if not triton_client.is_server_live():
            errors.append(f"Triton server {self._server_url} is not live")
        elif not triton_client.is_server_ready():
            errors.append(f"Triton server {self._server_url} is not ready")
        elif not triton_client.is_model_ready(self._model_name, self._model_version):
            errors.append(f"Model {self._model_name}:{self._model_version} is not ready")
        return errors


def _parse_args():
    parser = argparse.ArgumentParser(description="Infer model on Triton server", allow_abbrev=False)
    parser.add_argument(
        "--server-url", type=str, default="localhost:8001", help="Inference server URL (default localhost:8001)"
    )
    parser.add_argument("--model-name", help="The name of the model used for inference.", required=True)
    parser.add_argument("--model-version", help="The version of the model used for inference.", required=True)
    parser.add_argument("--dataloader", help="Path to python file containing dataloader.", required=True)
    parser.add_argument("--dump-labels", help="Dump labels to output dir", action="store_true", default=False)
    parser.add_argument("--dump-inputs", help="Dump inputs to output dir", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", help="Verbose logs", action="store_true", default=False)
    parser.add_argument("--output-dir", required=True, help="Path to directory where outputs will be saved")
    parser.add_argument("--response-wait-time", required=False, help="Maximal time to wait for response", type=int, default=120)
    parser.add_argument(
        "--max-unresponded-requests", required=False, help="Maximal number of unresponded requests", type=int, default=128
    )

    args, *_ = parser.parse_known_args()

    get_dataloader_fn = load_from_file(args.dataloader, label="dataloader", target=DATALOADER_FN_NAME)
    ArgParserGenerator(get_dataloader_fn).update_argparser(parser)
    args = parser.parse_args()

    return args


def main():
    args = _parse_args()

    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    log_level = logging.INFO if not args.verbose else logging.DEBUG
    logging.basicConfig(level=log_level, format=log_format)

    LOGGER.info(f"args:")
    for key, value in vars(args).items():
        LOGGER.info(f"    {key} = {value}")

    get_dataloader_fn = load_from_file(args.dataloader, label="dataloader", target=DATALOADER_FN_NAME)
    dataloader_fn = ArgParserGenerator(get_dataloader_fn).from_args(args)

    runner = AsyncGRPCTritonRunner(
        args.server_url,
        args.model_name,
        args.model_version,
        dataloader=dataloader_fn(),
        verbose=False,
        resp_wait_s=args.response_wait_time,
        max_unresponded_reqs=args.max_unresponded_requests,
    )

    with NpzWriter(output_dir=args.output_dir) as writer:
        start = time.time()
        for ids, x, y_pred, y_real in tqdm(runner, unit="batch", mininterval=10):
            data = _verify_and_format_dump(args, ids, x, y_pred, y_real)
            writer.write(**data)
        stop = time.time()

    LOGGER.info(f"\nThe inference took {stop - start:0.3f}s")


def _verify_and_format_dump(args, ids, x, y_pred, y_real):
    data = {"outputs": y_pred, "ids": {"ids": ids}}
    if args.dump_inputs:
        data["inputs"] = x
    if args.dump_labels:
        if not y_real:
            raise ValueError(
                "Found empty label values. Please provide labels in dataloader_fn or do not use --dump-labels argument"
            )
        data["labels"] = y_real
    return data


if __name__ == "__main__":
    main()
