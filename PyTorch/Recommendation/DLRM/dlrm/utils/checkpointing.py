# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
from os.path import join
from typing import Dict, Any, Optional, Sequence


class DlrmCheckpointNavigator:

    @property
    def bottom_mlp_path(self) -> str:
        return "bottom_model.mlp.pt"

    @property
    def top_model_path(self) -> str:
        return "top_model.pt"

    @property
    def metadata_path(self) -> str:
        return "metadata.pt"

    def embedding_path(self, embedding_index: int) -> str:
        return f"bottom_model.embeddings.{embedding_index}.pt"


class DistributedCheckpointWriter:

    def __init__(
        self,
        device_mapping: Dict[str, Any],
        config: Dict[str, Any],
        rank: int,
        main_process: bool
    ):
        self._device_mapping = device_mapping
        self._config = config
        self._main_process = main_process
        self._has_bottom_mlp = rank == device_mapping["bottom_mlp"]
        self._embedding_indices = device_mapping["embedding"][rank]
        self._navigator = DlrmCheckpointNavigator()

    def save_checkpoint(
        self,
        model,
        checkpoint_path: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None
    ):
        os.makedirs(checkpoint_path, exist_ok=True)

        self._save_embeddings_weights(checkpoint_path, model)

        if self._has_bottom_mlp:
            torch.save(model.bottom_model.mlp.state_dict(), join(checkpoint_path, self._navigator.bottom_mlp_path))

        if self._main_process:
            torch.save(model.top_model.state_dict(), join(checkpoint_path, self._navigator.top_model_path))
            self._save_metadata(checkpoint_path, epoch, step)

        torch.distributed.barrier()

    def _save_embeddings_weights(self, checkpoint_path: str, model):
        for embedding_index, weight in zip(self._embedding_indices, model.bottom_model.embeddings.weights):
            torch.save({"weight": weight}, join(checkpoint_path, self._navigator.embedding_path(embedding_index)))

    def _save_metadata(self, checkpoint_path, epoch, step):
        torch.save({
            "config": self._config,
            "device_mapping": self._device_mapping,
            "epoch": epoch,
            "step": step
        }, join(checkpoint_path, self._navigator.metadata_path))


class DistributedCheckpointLoader:

    def __init__(self, device_mapping: Dict[str, Any], rank: int):
        self._device_mapping = device_mapping
        self._has_bottom_mlp = rank == device_mapping["bottom_mlp"]
        self._embedding_indices = device_mapping["embedding"][rank]
        self._navigator = DlrmCheckpointNavigator()

    def load_checkpoint(self, model, checkpoint_path: str):
        top_model_state = self._load(checkpoint_path, self._navigator.top_model_path)
        model.top_model.load_state_dict(top_model_state)

        if self._has_bottom_mlp:
            bottom_mlp_state = self._load(checkpoint_path, self._navigator.bottom_mlp_path)
            model.bottom_model.mlp.load_state_dict(bottom_mlp_state)

        embedding_weights = (self._load(checkpoint_path, self._navigator.embedding_path(index))["weight"]
                             for index in self._embedding_indices)
        model.bottom_model.embeddings.load_weights(embedding_weights)

        torch.distributed.barrier()

    def _load(self, checkpoint_path: str, state_path: str):
        return torch.load(join(checkpoint_path, state_path), map_location="cpu")  # loading to CUDA causes OOM errors


class CpuCheckpointLoader:

    def __init__(self, embedding_indices: Sequence[int]):
        self._embedding_indices = embedding_indices
        self._navigator = DlrmCheckpointNavigator()

    def load_checkpoint(self, model, checkpoint_path: str):
        top_model_state = self._load(checkpoint_path, self._navigator.top_model_path)
        model.top_model.load_state_dict(top_model_state)

        bottom_mlp_state = self._load(checkpoint_path, self._navigator.bottom_mlp_path)
        model.bottom_model.mlp.load_state_dict(bottom_mlp_state)

        embedding_weights = (self._load(checkpoint_path, self._navigator.embedding_path(index))["weight"]
                             for index in self._embedding_indices)
        model.bottom_model.embeddings.load_weights(embedding_weights)

    def _load(self, checkpoint_path: str, state_path: str):
        data = torch.load(join(checkpoint_path, state_path), map_location="cpu")
        return {self._strip_key(key): value for key, value in data.items()}

    def _strip_key(self, key: str):
        prefix = "module."
        if key.startswith(prefix):
            return key[len(prefix):]
        return key
