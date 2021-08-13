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

from pathlib import Path
from typing import Dict, Iterable

import numpy as np

MB2B = 2 ** 20
B2MB = 1 / MB2B
FLUSH_THRESHOLD_B = 256 * MB2B


def pad_except_batch_axis(data: np.ndarray, target_shape_with_batch_axis: Iterable[int]):
    assert all(
        [current_size <= target_size for target_size, current_size in zip(target_shape_with_batch_axis, data.shape)]
    ), "target_shape should have equal or greater all dimensions comparing to data.shape"
    padding = [(0, 0)] + [  # (0, 0) - do not pad on batch_axis (with index 0)
        (0, target_size - current_size)
        for target_size, current_size in zip(target_shape_with_batch_axis[1:], data.shape[1:])
    ]
    return np.pad(data, padding, "constant", constant_values=np.nan)


class NpzWriter:
    """
    Dumps dicts of numpy arrays into npz files

    It can/shall be used as context manager:
    ```
    with OutputWriter('mydir') as writer:
        writer.write(outputs={'classes': np.zeros(8), 'probs': np.zeros((8, 4))},
                     labels={'classes': np.zeros(8)},
                     inputs={'input': np.zeros((8, 240, 240, 3)})
    ```

    ## Variable size data

    Only dynamic of last axis is handled. Data is padded with np.nan value.
    Also each generated file may have different size of dynamic axis.
    """

    def __init__(self, output_dir, compress=False):
        self._output_dir = Path(output_dir)
        self._items_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._items_counters: Dict[str, int] = {}
        self._flush_threshold_b = FLUSH_THRESHOLD_B
        self._compress = compress

    @property
    def cache_size(self):
        return {name: sum([a.nbytes for a in data.values()]) for name, data in self._items_cache.items()}

    def _append_to_cache(self, prefix, data):
        if data is None:
            return

        if not isinstance(data, dict):
            raise ValueError(f"{prefix} data to store shall be dict")

        cached_data = self._items_cache.get(prefix, {})
        for name, value in data.items():
            assert isinstance(
                value, (list, np.ndarray)
            ), f"Values shall be lists or np.ndarrays; current type {type(value)}"
            if not isinstance(value, np.ndarray):
                value = np.array(value)

            assert value.dtype.kind in ["S", "U"] or not np.any(
                np.isnan(value)
            ), f"Values with np.nan is not supported; {name}={value}"
            cached_value = cached_data.get(name, None)
            if cached_value is not None:
                target_shape = np.max([cached_value.shape, value.shape], axis=0)
                cached_value = pad_except_batch_axis(cached_value, target_shape)
                value = pad_except_batch_axis(value, target_shape)
                value = np.concatenate((cached_value, value))
            cached_data[name] = value
        self._items_cache[prefix] = cached_data

    def write(self, **kwargs):
        """
        Writes named list of dictionaries of np.ndarrays.
        Finally keyword names will be later prefixes of npz files where those dictionaries will be stored.

        ex. writer.write(inputs={'input': np.zeros((2, 10))},
                         outputs={'classes': np.zeros((2,)), 'probabilities': np.zeros((2, 32))},
                         labels={'classes': np.zeros((2,))})
        Args:
            **kwargs: named list of dictionaries of np.ndarrays to store
        """

        for prefix, data in kwargs.items():
            self._append_to_cache(prefix, data)

        biggest_item_size = max(self.cache_size.values())
        if biggest_item_size > self._flush_threshold_b:
            self.flush()

    def flush(self):
        for prefix, data in self._items_cache.items():
            self._dump(prefix, data)
        self._items_cache = {}

    def _dump(self, prefix, data):
        idx = self._items_counters.setdefault(prefix, 0)
        filename = f"{prefix}-{idx:012d}.npz"
        output_path = self._output_dir / filename
        if self._compress:
            np.savez_compressed(output_path, **data)
        else:
            np.savez(output_path, **data)

        nitems = len(list(data.values())[0])

        msg_for_labels = (
            "If these are correct shapes - consider moving loading of them into metrics.py."
            if prefix == "labels"
            else ""
        )
        shapes = {name: value.shape if isinstance(value, np.ndarray) else (len(value),) for name, value in data.items()}

        assert all(len(v) == nitems for v in data.values()), (
            f'All items in "{prefix}" shall have same size on 0 axis equal to batch size. {msg_for_labels}'
            f'{", ".join(f"{name}: {shape}" for name, shape in shapes.items())}'
        )
        self._items_counters[prefix] += nitems

    def __enter__(self):
        if self._output_dir.exists() and len(list(self._output_dir.iterdir())):
            raise ValueError(f"{self._output_dir.as_posix()} is not empty")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
