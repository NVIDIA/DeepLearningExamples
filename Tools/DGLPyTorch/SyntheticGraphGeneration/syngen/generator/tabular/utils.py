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

import os
import gc
import multiprocessing
from functools import partial
from typing import Callable, List

from tqdm import tqdm

from syngen.generator.tabular.chunked_tabular_generator import ChunkedBaseTabularGenerator
from syngen.utils.io_utils import dump_dataframe
from syngen.utils.memory_manager import MemoryManager


def _generate_samples(
        gen,
        n_samples: int,
        fname: str,
        save_path: str,
        post_gen_fn: Callable = None,
        i: int = 0,
):
    """
        MP sample generation fn
    """
    fp = os.path.join(save_path, f"{fname}_{i}")
    samples = gen.sample(n_samples)
    if post_gen_fn is not None:
        samples = post_gen_fn(samples)
    dump_dataframe(samples, fp, format='parquet')
    return fp


def pass_through(x):
    return x


def tabular_chunk_sample_generation(
        gen,
        n_samples: int,
        save_path: str,
        fname: str,
        post_gen_fn: Callable = pass_through,
        num_workers: int = 1,
        use_memmap=False,
        verbose=True,
) -> List[str]:
    """
    Chunk large sample generation into parts,
    and dump csv files into save_path to avoid memory issues.

    Args:
        gen: generator to sample new synthetic data from,
        must implement `sample`
        n_samples (int): number of samples to generate
        save_path: directory to dump generated samples
        fname (str): file name for saving csv's
        post_gen_fn (Callable): will be called on generated samples
        num_workers (int): number of workers to speed up generation using multiprocessing
    Returns:
        None
    """

    if isinstance(gen, ChunkedBaseTabularGenerator):
        return gen.chunked_sampling(int(n_samples),
                                    save_path=save_path,
                                    fname=fname,
                                    gpus=-1,
                                    use_memmap=use_memmap,
                                    verbose=verbose,
                                    )

    n_samples = int(n_samples)
    # - check if mem available
    gc.collect()
    mem_avail = MemoryManager().get_available_virtual_memory()
    emp_n = 1000
    est_samples = gen.sample(emp_n)
    mem_usage = est_samples.memory_usage(index=True, deep=True).sum()
    est_mem = (mem_usage / emp_n) * n_samples

    # - path
    file_paths = []

    # - gen samples
    if n_samples <= 1e6 and mem_avail > est_mem:
        file_paths.append(
            _generate_samples(
                gen=gen,
                n_samples=n_samples,
                fname=fname,
                save_path=save_path,
                post_gen_fn=post_gen_fn,
                i=n_samples,
            )
        )
    else:
        r = (est_mem // mem_avail) + 10
        inc = int(min(n_samples // r, 5e6))
        num_iters = n_samples / inc
        if num_iters - n_samples // inc > 0.0:
            num_iters += 1
        num_iters = int(num_iters)

        generate_samples_p = partial(
            _generate_samples, gen, inc, fname, save_path, post_gen_fn
        )
        if num_workers > 1:
            multiprocessing.set_start_method("spawn", force=True)
            with multiprocessing.Pool(processes=num_workers) as pool:
                tasks = pool.imap_unordered(generate_samples_p, range(0, num_iters))

                if verbose:
                    tasks = tqdm(tasks, total=num_iters)

                file_paths = list(tasks)
        else:
            itr = range(0, n_samples, inc)
            if verbose:
                itr = tqdm(itr)
            for i in itr:
                file_paths.append(generate_samples_p(i))

    return file_paths
