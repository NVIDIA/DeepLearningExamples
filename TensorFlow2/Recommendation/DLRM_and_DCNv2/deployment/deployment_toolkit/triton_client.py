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
import logging
import time
from enum import Enum
from typing import Any, Dict, Optional

# pytype: disable=import-error
from .utils import parse_server_url

try:
    import tritonclient.grpc as grpc_client
    from tritonclient import utils as client_utils  # noqa: F401
except ImportError:
    try:
        import tritonclientutils as client_utils  # noqa: F401
        import tritongrpcclient as grpc_client
    except ImportError:
        client_utils = None
        grpc_client = None

try:
    import tritonclient.http as http_client
except (ImportError, RuntimeError):
    try:
        import tritonhttpclient as http_client
    except (ImportError, RuntimeError):
        http_client = None

# pytype: enable=import-error

LOGGER = logging.getLogger(__name__)


class TritonServerNotReadyException(Exception):
    pass


# TODO: in which state "native" warm-up takes place?
class ModelState(Enum):
    """Describe model state in Triton.

    Attributes:
        LOADING: Loading of model
        UNLOADING: Unloading of model
        UNAVAILABLE: Model is missing or could not be loaded
        READY: Model is ready for inference
    """

    LOADING = "LOADING"
    UNLOADING = "UNLOADING"
    UNAVAILABLE = "UNAVAILABLE"
    READY = "READY"


class TritonClientProtocol(Enum):
    """Describe protocol with which client communicates with Triton"""

    GRPC = "grpc"
    HTTP = "http"


# TODO: How to obtain models that are available but not loaded yet?
# TODO: encode model_name and model_version as for ex. model_name/model_version (here and in many other places)
# TODO: How to obtain server model loading mode
class TritonClient:
    """Provide high-level API for communicating with Triton.

    Usage:

    >>> client = TritonClient("grpc://127.0.0.1:8001")
    >>> client.load_model("ResNet50")

    Above sample loads model on Triton and run inference iterating over provided dataloader.

    Args:
        server_url: url where Triton is binded in format `<protocol>://<address/hostname>:<port>`
        verbose: provide verbose logs from tritonclient library

    Attributes:
        client: handle to low-level API client obtained from tritonclient python package

    Raises:
        RuntimeError: in case of missing tritonclient library for selected protocol
                      or problems with connecting to Triton or its not in ready state yet.
        ValueError: in case of errors in parsing provided server_url. Example source of errors are: missing protocol unknown protocol was requested.
        InferenceServerClient: in case of error in processing initial requests on server side
    """

    def __init__(self, server_url: str, *, verbose: bool = False):
        self.server_url = server_url
        self._verbose = verbose

        self.client = self._create_client(server_url=server_url, verbose=verbose)

    def wait_for_server_ready(self, timeout: int):
        """
        Parameters
        ----------
        timeout : int
            timeout in seconds to send a ready status
            request to the server before raising
            an exception
        Raises
        ------
        TritonModelAnalyzerException
            If server readiness could not be
            determined in given num_retries
        """

        retries = timeout
        while retries > 0:
            try:
                if self.client.is_server_ready() and self.client.is_server_live():
                    return
                else:
                    time.sleep(1)
                    retries -= 1
            except Exception as e:
                time.sleep(1)
                retries -= 1
                if retries == 0:
                    return TritonServerNotReadyException(e)
        raise TritonServerNotReadyException(
            "Could not determine server readiness. " "Number of retries exceeded."
        )

    def get_server_metadata(self):
        """Returns `server metadata <https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md#server-metadata-response-json-object>`_.

        >>> client.get_server_metadata()
        {name: "triton", version: "2.5.0", extensions: ["classification", "sequence", "model_repository", "schedule_policy", "model_configuration", "system_shared_memory", "cuda_shared_memory", "binary_tensor_data", "statistics"}

        Returns:
            Dictionary with server metadata.

        Raises:
            InferenceServerClient: in case of error in processing request on server side
        """
        server_metadata = self.client.get_server_metadata()
        server_metadata = self._format_response(server_metadata)
        return server_metadata

    def get_model_metadata(self, model_name: str, model_version: Optional[str] = None):
        """Returns `model metadata <https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md#model-metadata>`_.

        Args:
            model_name: name of the model which metadata is requested to obtain.
            model_version: version of the model which metadata is requested to obtain.

        Returns:
            Dictionary with model metadata.

        Raises:
            InferenceServerClient: in case of error in processing request on server side.
        """
        model_metadata = self.client.get_model_metadata(model_name, model_version)
        model_metadata = self._format_response(model_metadata)
        return model_metadata

    def load_model(self, model_name: str) -> None:
        """Requests that a model be loaded into Triton, or reloaded if the model is already loaded.

        Args:
            model_name: name of the model to load

        Raises:
            InferenceServerException: in case of error in processing request on server side.
        """
        self.client.load_model(model_name)

    def wait_for_model(
        self,
        *,
        model_name: str,
        model_version: str,
        timeout_s: int = 120,
        check_interval_s: int = 5,
    ) -> Dict[str, Any]:
        """Iteratively check for model state until model is ready or unavailable.

        Args:
            model_name: name of the model to wait for
            model_version: version of the model to wait for
            timeout_s: how long in seconds to wait till model is in ready or in unavailable state
            check_interval_s: time intervals in seconds at which state of model is should be checked

        Returns:
            Dictionary with model metadata.

        Raises:
            RuntimeError: in case model is not ready yet (is marked unavailable or timeout has been reached)
            InferenceServerException: in case of error in processing request on server side.
        """

        def _shall_wait(model_state: ModelState) -> bool:
            return model_state not in [ModelState.UNAVAILABLE, ModelState.READY]

        elapsed_time_s = 0
        start_time_s = time.time()
        state = self.get_model_state(model_name, model_version)
        while elapsed_time_s < timeout_s and _shall_wait(state):
            LOGGER.info(
                f"waiting for model... {elapsed_time_s:.0f}/{timeout_s} state={state}"
            )
            time.sleep(check_interval_s)
            state = self.get_model_state(model_name, model_version)
            elapsed_time_s = time.time() - start_time_s

        if not self.client.is_model_ready(model_name):
            raise RuntimeError(
                f"Model {model_name} requested to be loaded, but is not ready"
            )

        model_metadata = self.client.get_model_metadata(model_name)
        model_metadata = self._format_response(model_metadata)
        return model_metadata

    def get_model_state(self, model_name: str, model_version: str) -> ModelState:
        """Obtains the state of a model on Triton.

        Args:
            model_name: name of the model which state is requested to obtain.
            model_version: version of the model which state is requested to obtain.

        Returns:
            Requested model state.

        Raises:
            InferenceServerException: in case of error in processing request on server side.
        """

        def handle_http_response(models):
            models_states = {}
            for model in models:
                if not model.get("version"):
                    continue

                model_state = (
                    ModelState(model["state"])
                    if model.get("state")
                    else ModelState.LOADING
                )
                models_states[(model["name"], model["version"])] = model_state

            return models_states

        def handle_grpc_response(models):
            models_states = {}
            for model in models:
                if not model.version:
                    continue

                model_state = (
                    ModelState(model.state) if model.state else ModelState.LOADING
                )
                models_states[(model.name, model.version)] = model_state

            return models_states

        repository_index = self.client.get_model_repository_index()
        if isinstance(repository_index, list):
            models_states = handle_http_response(models=repository_index)
        else:
            models_states = handle_grpc_response(models=repository_index.models)

        return models_states.get((model_name, model_version), ModelState.UNAVAILABLE)

    def _format_response(self, response):
        if not isinstance(response, dict):
            response = json.loads(
                grpc_client.MessageToJson(response, preserving_proto_field_name=True)
            )
        return response

    def _create_client(self, server_url: str, verbose: bool):
        protocol, host, port = parse_server_url(server_url)
        if protocol == TritonClientProtocol.HTTP and http_client is None:
            raise RuntimeError(
                "Could not obtain Triton HTTP client. Install extras while installing tritonclient wheel. "
                "Example installation call: "
                "find /workspace/install/python/ -iname triton*manylinux*.whl -exec pip install {}[all] \\;"
            )

        LOGGER.debug(f"Connecting to {server_url}")

        client_lib = {
            TritonClientProtocol.HTTP.value: http_client,
            TritonClientProtocol.GRPC.value: grpc_client,
        }[protocol.value]
        server_url = f"{host}:{port}"

        # pytype: disable=attribute-error
        client = client_lib.InferenceServerClient(url=server_url, verbose=verbose)
        # pytype: enable=attribute-error

        return client
