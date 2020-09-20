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

from typing import Optional, Sequence, Dict, Any

from dlrm.utils.checkpointing.model import DlrmCheckpointWriter, DlrmCheckpointLoader


class SerialCheckpointWriter:

    def __init__(self, writer: DlrmCheckpointWriter):
        self._writer = writer

    def save_checkpoint(
        self,
        model,
        checkpoint_path: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None
    ):
        self._writer.save_embeddings(checkpoint_path, model)
        self._writer.save_bottom_mlp(checkpoint_path, model)
        self._writer.save_top_model(checkpoint_path, model)
        self._writer.save_metadata(checkpoint_path, {
            "epoch": epoch,
            "step": step
        })


class SerialCheckpointLoader:

    def __init__(self, loader: DlrmCheckpointLoader):
        self._loader = loader

    def load_checkpoint(self, model, checkpoint_path: str):
        self._loader.load_top_model(checkpoint_path, model)
        self._loader.load_bottom_mlp(checkpoint_path, model)
        self._loader.load_embeddings(checkpoint_path, model)


def make_serial_checkpoint_loader(embedding_indices: Sequence[int], device: str) -> SerialCheckpointLoader:
    return SerialCheckpointLoader(DlrmCheckpointLoader(
        embedding_indices=embedding_indices,
        device=device,
    ))


def make_serial_checkpoint_writer(
        embedding_indices: Sequence[int],
        config: Dict[str, Any],
) -> SerialCheckpointWriter:
    return SerialCheckpointWriter(DlrmCheckpointWriter(
        embedding_indices=embedding_indices,
        config=config
    ))
