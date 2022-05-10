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
from enum import Enum
from typing import Any, Dict, Tuple

LOGGER = logging.getLogger(__name__)


class TritonClientProtocol(Enum):
    """Describe protocol with which client communicates with Triton"""

    GRPC = "grpc"
    HTTP = "http"


def parse_server_url(server_url: str) -> Tuple[TritonClientProtocol, str, int]:
    DEFAULT_PORTS = {
        TritonClientProtocol.HTTP: 8000,
        TritonClientProtocol.GRPC: 8001,
    }

    # extract protocol
    server_url_items = server_url.split("://")
    if len(server_url_items) != 2:
        raise ValueError("Prefix server_url with protocol ex.: grpc://127.0.0.1:8001")
    requested_protocol, server_url = server_url_items
    requested_protocol = TritonClientProtocol(requested_protocol.lower())

    if requested_protocol not in DEFAULT_PORTS:
        raise ValueError(f"Unsupported protocol: {requested_protocol}")

    # extract host and port
    default_port = DEFAULT_PORTS[requested_protocol]
    server_url_items = server_url.split(":")
    if len(server_url_items) == 1:
        host, port = server_url, default_port
    elif len(server_url_items) == 2:
        host, port = server_url_items
        port = int(port)
        if port != default_port:
            LOGGER.warning(
                f"Current server URL is {server_url} while default {requested_protocol} port is {default_port}"
            )
    else:
        raise ValueError(f"Could not parse {server_url}. Example of correct server URL: grpc://127.0.0.1:8001")
    return requested_protocol, host, port


def log_dict(title: str, dict_: Dict[str, Any]):
    LOGGER.info(title)
    for key, value in dict_.items():
        LOGGER.info(f"\t{key} = {value}")
