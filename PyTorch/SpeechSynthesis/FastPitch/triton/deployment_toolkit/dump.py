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
import abc
import json
import pickle
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Union

import numpy as np

MB2B = 2 ** 20
B2MB = 1 / MB2B
FLUSH_THRESHOLD_B = 256 * MB2B


def _validate_batch(name: str, value: Union[list, np.ndarray]):
    if not isinstance(value, (list, np.ndarray)):
        raise ValueError(f"Values shall be lists or np.ndarrays; current type {type(value)}")


def _validate_prefix_data(prefix_data: Dict[str, List[np.ndarray]]):
    batch_sizes_per_io_name = {name: [len(batch) for batch in batches] for name, batches in prefix_data.items()}
    names = list(batch_sizes_per_io_name)
    for io_name in names:
        for batch_idx, batch_size in enumerate(batch_sizes_per_io_name[io_name]):
            if not all([batch_sizes_per_io_name[other_name][batch_idx] == batch_size for other_name in names]):
                non_equal_batch_sizes = {
                    other_name: batch_sizes_per_io_name[other_name][batch_idx] for other_name in names
                }
                non_equal_batch_sizes_str = ", ".join(
                    [f"{name}={batch_size}" for name, batch_size in non_equal_batch_sizes.items()]
                )
                raise ValueError(
                    "All inputs/outputs should have same number of batches with equal batch_size. "
                    f"At batch_idx={batch_idx} there are batch_sizes: {non_equal_batch_sizes_str}"
                )
        # ensure if each io has same number of batches with equal size


def _get_nitems_and_batches(prefix_data: Dict[str, List[np.ndarray]]):
    nitems = 0
    nbatches = 0

    if prefix_data:
        nitems_per_io_name = {name: sum(len(batch) for batch in batches) for name, batches in prefix_data.items()}
        nbatches_per_io_name = {name: len(batches) for name, batches in prefix_data.items()}
        nitems = list(nitems_per_io_name.values())[0]
        nbatches = list(nbatches_per_io_name.values())[0]
    return nitems, nbatches


class BaseDumpWriter(abc.ABC):
    FILE_SUFFIX = ".abstract"

    def __init__(self, output_dir: Union[str, Path]):
        self._output_dir = Path(output_dir)
        # outer dict key is prefix (i.e. input/output/labels/...), inner dict key is input/output name
        # list is list of batches
        self._items_cache: Dict[str, Dict[str, List[np.ndarray]]] = {}
        # key is prefix
        self._items_counters: Dict[str, int] = {}
        self._cache_lock = threading.RLock()
        self._flush_threshold_b = FLUSH_THRESHOLD_B

    @property
    def cache_size(self):
        def _get_bytes_size(name, batch):
            _validate_batch(name, batch)
            if not isinstance(batch, np.ndarray):
                batch = np.narray(batch)

            return batch.nbytes

        with self._cache_lock:
            return {
                prefix: sum(_get_bytes_size(name, batch) for name, batches in data.items() for batch in batches)
                for prefix, data in self._items_cache.items()
            }

    def _append_to_cache(self, prefix, prefix_data):
        if prefix_data is None:
            return

        if not isinstance(prefix_data, dict):
            raise ValueError(f"{prefix} data to store shall be dict")

        with self._cache_lock:
            cached_prefix_data = self._items_cache.setdefault(prefix, {})
            for name, batch in prefix_data.items():
                _validate_batch(name, batch)
                if not isinstance(batch, np.ndarray):
                    batch = np.array(batch)

                cached_batches = cached_prefix_data.setdefault(name, [])
                cached_batches += [batch]

    def write(self, **kwargs):
        with self._cache_lock:
            for prefix, prefix_data in kwargs.items():
                self._append_to_cache(prefix, prefix_data)

            biggest_prefix_data_size = max(self.cache_size.values())
            if biggest_prefix_data_size > self._flush_threshold_b:
                self.flush()

    def flush(self):
        with self._cache_lock:
            for prefix, prefix_data in self._items_cache.items():
                _validate_prefix_data(prefix_data)

                output_path = self._output_dir / self._get_filename(prefix)
                self._dump(prefix_data, output_path)

                nitems, nbatches = _get_nitems_and_batches(prefix_data)
                self._items_counters[prefix] += nitems
            self._items_cache = {}

    def _get_filename(self, prefix):
        idx = self._items_counters.setdefault(prefix, 0)
        return f"{prefix}-{idx:012d}{self.FILE_SUFFIX}"

    @abc.abstractmethod
    def _dump(self, prefix_data: Dict[str, List[np.ndarray]], output_path: Path):
        pass

    def __enter__(self):
        if self._output_dir.exists() and len(list(self._output_dir.iterdir())):
            raise ValueError(f"{self._output_dir.as_posix()} is not empty")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


class PickleDumpWriter(BaseDumpWriter):
    FILE_SUFFIX = ".pkl"

    def _dump(self, prefix_data: Dict[str, List[np.ndarray]], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as pickle_file:
            pickle.dump(prefix_data, pickle_file)


class JsonDumpWriter(BaseDumpWriter):
    FILE_SUFFIX = ".json"

    def _dump(self, prefix_data: Dict[str, List[np.ndarray]], output_path: Path):
        repacked_prefix_data = self._format_data(prefix_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as json_file:
            json.dump(repacked_prefix_data, json_file)

    def _format_data(self, prefix_data: Dict[str, List[np.ndarray]]) -> Dict:
        def _format_batch_for_perf_analyzer_json_format(batch: np.ndarray):
            return {
                "content": batch.flatten().tolist(),
                "shape": list(batch.shape),
                "dtype": str(batch.dtype),
            }

        _, nbatches = _get_nitems_and_batches(prefix_data)
        batches = [{} for _ in range(nbatches)]
        for io_name, batches_per_io in prefix_data.items():
            for batch_idx, batch in enumerate(batches_per_io):
                batches[batch_idx][io_name] = _format_batch_for_perf_analyzer_json_format(batch)

        return {"data": batches}


class BaseDumpReader(abc.ABC):
    FILE_SUFFIX = ".abstract"

    def __init__(self, dump_dir: Union[Path, str]):
        self._dump_dir = Path(dump_dir)

    def get(self, prefix: str) -> Iterator[Dict[str, np.ndarray]]:
        dump_files_paths = sorted(self._dump_dir.glob(f"{prefix}*{self.FILE_SUFFIX}"))
        for dump_file_path in dump_files_paths:
            prefix_data = self._load_file(dump_file_path)
            nitems, nbatches = _get_nitems_and_batches(prefix_data)
            for batch_idx in range(nbatches):
                yield {io_name: prefix_data[io_name][batch_idx] for io_name in prefix_data}

    @abc.abstractmethod
    def _load_file(self, dump_file_path: Path) -> Dict[str, List[np.ndarray]]:
        pass

    def iterate_over(self, prefix_list: List[str]) -> Iterator:
        iterators = [self.get(prefix) for prefix in prefix_list]
        empty_iterators = [False] * len(iterators)
        while not all(empty_iterators):
            values = [None] * len(iterators)
            for idx, iterator in enumerate(iterators):
                if empty_iterators[idx]:
                    continue
                try:
                    values[idx] = next(iterator)
                except StopIteration:
                    empty_iterators[idx] = True
                    if all(empty_iterators):
                        break

            if not all(empty_iterators):
                yield values

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class PickleDumpReader(BaseDumpReader):
    FILE_SUFFIX = ".pkl"

    def _load_file(self, dump_file_path: Path) -> Dict[str, List[np.ndarray]]:
        with dump_file_path.open("rb") as pickle_file:
            return pickle.load(pickle_file)


class JsonDumpReader(BaseDumpReader):
    FILE_SUFFIX = ".json"

    def _load_file(self, dump_file_path: Path) -> Dict[str, List[np.ndarray]]:
        with dump_file_path.open("rb") as json_file:
            data = json.load(json_file)
            return self._repack_data(data)

    def _repack_data(self, data: Dict) -> Dict[str, List[np.ndarray]]:
        result: Dict[str, List[np.ndarray]] = {}
        batches = data["data"]
        for batch in batches:
            for io_name, batch_as_dict in batch.items():
                io_batches = result.setdefault(io_name, [])
                flat_array = batch_as_dict["content"]
                shape = batch_as_dict["shape"]
                dtype = batch_as_dict["dtype"]
                batch_as_array = np.array(flat_array).reshape(shape).astype(dtype)
                io_batches.append(batch_as_array)
        return result
