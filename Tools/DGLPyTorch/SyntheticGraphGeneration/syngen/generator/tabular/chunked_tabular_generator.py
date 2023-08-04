# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import os.path
from abc import ABC

import tqdm
import cupy as cp
import numpy as np

import multiprocessing
from functools import partial

from syngen.utils.io_utils import dump_dataframe
from syngen.utils.types.dataframe_type import DataFrameType
from syngen.utils.memory_manager import MemoryManager
from syngen.generator.tabular import BaseTabularGenerator


class ChunkedBaseTabularGenerator(BaseTabularGenerator, ABC):

    """ A Chunked Base Tabular Generator contains the base functionality of the multiprocess (Multi-GPU) data generation.

    """
    def chunked_sampling(self, n_samples: int, save_path: str, fname: str, n_workers: int = 0, gpus: int = -1,
                         use_memmap=False, memory_threshold=0.8, verbose=True):
        memory_manager = MemoryManager()

        if gpus < 0:
            gpus = memory_manager.get_available_gpus()

        emp_n = 1000
        est_samples = self.sample(emp_n, gpu=False)
        mem_usage = est_samples.memory_usage(index=True, deep=True).sum()
        est_sample_mem = int(np.ceil(mem_usage / emp_n * self._space_complexity_factor()))
        est_mem = est_sample_mem * n_samples

        memmap_kwargs = None
        chunk_save_path = None

        if use_memmap:
            assert fname.endswith(".npy")
            memmap_shape = list(est_samples.shape)
            memmap_shape[0] = n_samples
            memmap_shape = tuple(memmap_shape)
            memmap_dtype = est_samples.dtypes.iloc[0]
            memmap_filename = os.path.join(save_path, fname)
            memmap_kwargs = dict(
                filename=memmap_filename,
            )
            memmap_outfile = np.lib.format.open_memmap(memmap_filename, dtype=memmap_dtype, shape=memmap_shape, mode='w+')
        else:
            chunk_format = '{chunk_id}'
            chunk_save_path = os.path.join(save_path, f'{fname}_{chunk_format}')

        if gpus > 0:
            mem_avail = memory_manager.get_min_available_across_gpus_memory(gpus=gpus)
            n_workers = gpus
            chunk_partial = partial(self._generate_chunk,
                                    chunk_save_path=chunk_save_path, gpu=True, gpus=gpus, memmap_kwargs=memmap_kwargs)
        else:
            mem_avail = memory_manager.get_available_virtual_memory()
            chunk_partial = partial(self._generate_chunk,
                                    chunk_save_path=chunk_save_path, gpu=False, memmap_kwargs=memmap_kwargs)

        if mem_avail * memory_threshold > est_mem:
            df = self.sample(n_samples, gpu=True, memmap_kwargs=memmap_kwargs, start_idx=0, end_idx=n_samples)
            if chunk_save_path:
                chunk_save_path = chunk_save_path.format(chunk_id=0)
                dump_dataframe(df, save_path=chunk_save_path, format='parquet')
            res = [chunk_save_path]
        else:
            mem_avail = int(mem_avail * memory_threshold)  # to avoid OOM
            max_samples_per_chunk = int(mem_avail // est_sample_mem)
            n_chunks = n_samples//max_samples_per_chunk + (1 if n_samples % max_samples_per_chunk > 0 else 0)

            samples_per_chunk = n_samples // n_chunks
            chunk_sizes = [samples_per_chunk] * n_chunks

            if n_samples % n_chunks > 0:
                chunk_sizes.append(n_samples % n_chunks)

            multiprocessing.set_start_method('spawn', force=True)
            with multiprocessing.Pool(processes=n_workers) as pool:
                res = pool.imap_unordered(chunk_partial, enumerate(zip(chunk_sizes, np.cumsum(chunk_sizes))),
                                          chunksize=(len(chunk_sizes)+n_workers-1)//n_workers)

                if verbose:
                    res = tqdm.tqdm(res, total=len(chunk_sizes))

                res = list(res)

        return res

    def _generate_chunk(self, chunk_info, chunk_save_path, gpu, memmap_kwargs, gpus=0):
        chunk_id, (chunk_size, chunk_end) = chunk_info

        if gpu:
            gpu_id = int(multiprocessing.current_process()._identity[0]) % gpus
            with cp.cuda.Device(gpu_id):
                df = self.sample(chunk_size, gpu=True, memmap_kwargs=memmap_kwargs,
                                 start_idx=chunk_end-chunk_size, end_idx=chunk_end)
        else:
            df = self.sample(chunk_size, gpu=False, memmap_kwargs=memmap_kwargs,
                             start_idx=chunk_end-chunk_size, end_idx=chunk_end)

        if chunk_save_path:
            chunk_save_path = chunk_save_path.format(chunk_id=chunk_id)
            dump_dataframe(df, save_path=chunk_save_path, format='parquet')

        return chunk_save_path

    def _space_complexity_factor(self):
        return 2.0  # we support float16 but it requires intermediate float32

    @property
    def supports_memmap(self) -> bool:
        return True

    def sample(self, num_samples, *args, gpu=False, **kwargs) -> DataFrameType:
        """generate `num_samples` from generator

        Args:
            num_samples (int): number of samples to generate
            gpu (bool): whether to use cpu or gpu implementation (default: False)
            *args: optional positional args
            **kwargs: optional key-word arguments
        """
        raise NotImplementedError()
