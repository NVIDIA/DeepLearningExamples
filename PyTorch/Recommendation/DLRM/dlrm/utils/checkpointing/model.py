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
import numpy as np
from os.path import join
from typing import Sequence, Any, Dict

import torch

_BOTTOM_MLP_FILE = "bottom_model.mlp.pt"
_TOP_MLP_FILE = "top_model.mlp.pt"
_TOP_OUT_FILE = "top_model.out.pt"
_EMBEDDING_METADATA_FILE = "embeddings.metadata.pt"
_METADATA_FILE = "metadata.pt"


def _get_embedding_file(embedding_index: int) -> str:
    return f"bottom_model.embeddings.{embedding_index}.bin"


def _get_embedding_meta_file(embedding_index: int) -> str:
    return f"embeddings.{embedding_index}.meta.pt"


class DlrmCheckpointWriter:
    """
    Class responsible for saving checkpoints of DLRM model parts.

    Depends on `dlrm.nn.embeddings.Embeddings` and `dlrm.nn.mlps.AbstractMlp` interfaces
    (for handling multiple model configurations)
    """
    def __init__(self, embedding_indices: Sequence[int], config: Dict[str, Any]):
        self._embedding_indices = embedding_indices
        self._config = config

    def save_embeddings(self, checkpoint_path: str, model):
        self._ensure_directory(checkpoint_path)
        for embedding_index, weight in zip(self._embedding_indices, model.bottom_model.embeddings.weights):
            self._save_as_bytes(weight.data, join(checkpoint_path, _get_embedding_file(embedding_index)))
            torch.save({"shape": weight.shape}, join(checkpoint_path, _get_embedding_meta_file(embedding_index)))

    def save_bottom_mlp(self, checkpoint_path: str, model):
        self._ensure_directory(checkpoint_path)
        torch.save(self._mlp_state(model.bottom_model.mlp), join(checkpoint_path, _BOTTOM_MLP_FILE))

    def save_top_model(self, checkpoint_path: str, model):
        self._ensure_directory(checkpoint_path)
        # DistributedDataParallel wraps top_model under "module" attribute
        top_model = model.top_model.module if hasattr(model.top_model, 'module') else model.top_model

        torch.save(self._mlp_state(top_model.mlp), join(checkpoint_path, _TOP_MLP_FILE))
        torch.save(top_model.out.state_dict(), join(checkpoint_path, _TOP_OUT_FILE))

    def save_metadata(self, checkpoint_path: str, data: Dict[str, Any]):
        self._ensure_directory(checkpoint_path)
        torch.save({"data": data, "config": self._config}, join(checkpoint_path, _METADATA_FILE))

    def _ensure_directory(self, checkpoint_path: str):
        os.makedirs(checkpoint_path, exist_ok=True)

    def _mlp_state(self, mlp):
        return {
            "weights": [x.to(torch.float32) for x in mlp.weights],
            "biases": [x.to(torch.float32) for x in mlp.biases]
        }

    def _save_as_bytes(self, tensor: torch.Tensor, path: str):
        with open(path, "wb+") as file:
            file.write(tensor.cpu().numpy().astype(np.float32).tobytes())


class DlrmCheckpointLoader:
    """
    Class responsible for loading checkpoints of DLRM model parts.

    Depends on `dlrm.nn.embeddings.Embeddings` and `dlrm.nn.mlps.AbstractMlp` interfaces
    (for handling multiple model configurations)
    """
    def __init__(self, embedding_indices: Sequence[int], device: str = "cpu"):
        self._embedding_indices = embedding_indices
        self._device = device

    def load_embeddings(self, checkpoint_path: str, model):
        embedding_weights = (self._load_from_bytes(join(checkpoint_path, _get_embedding_file(index)),
                                                   self._get_embedding_shape(checkpoint_path, index))
                             for index in self._embedding_indices)
        model.bottom_model.embeddings.load_weights(embedding_weights)

    def load_bottom_mlp(self, checkpoint_path: str, model):
        bottom_mlp_state = self._load(checkpoint_path, _BOTTOM_MLP_FILE)
        model.bottom_model.mlp.load_state(bottom_mlp_state["weights"], bottom_mlp_state["biases"])

    def load_top_model(self, checkpoint_path: str, model):
        # DistributedDataParallel wraps top_model under "module" attribute
        top_model = model.top_model.module if hasattr(model.top_model, 'module') else model.top_model
        top_mlp_state = self._load(checkpoint_path, _TOP_MLP_FILE)
        top_model.mlp.load_state(top_mlp_state["weights"], top_mlp_state["biases"])

        top_out_state = self._load(checkpoint_path, _TOP_OUT_FILE)
        top_model.out.load_state_dict(top_out_state)

    def _load(self, checkpoint_path: str, state_path: str):
        data = torch.load(join(checkpoint_path, state_path), map_location=self._device)
        return {self._strip_key(key): value for key, value in data.items()}

    def _strip_key(self, key: str):
        # DistributedDataParallel wraps top_model under "module" attribute
        prefix = "module."
        if key.startswith(prefix):
            return key[len(prefix):]
        return key

    def _load_from_bytes(self, path: str, shape) -> torch.Tensor:
        with open(path, "rb") as file:
            array = np.frombuffer(file.read(), dtype=np.float32).reshape(*shape)
            return torch.from_numpy(array).to(self._device)

    def _get_embedding_shape(self, checkpoint_path: str, index: int) -> tuple:
        embedding_meta = torch.load(join(checkpoint_path, _get_embedding_meta_file(index)))
        return embedding_meta["shape"]
