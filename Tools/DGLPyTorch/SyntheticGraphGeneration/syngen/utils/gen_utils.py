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

import gc
import inspect
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Callable, List

import cudf
import numpy as np
import pandas as pd
import psutil
import torch
import cugraph
from tqdm import tqdm
import torch.multiprocessing as mp
import cugraph.dask as dask_cugraph
from torch.multiprocessing import Pool

from syngen.utils.df_reader import DFReader

logger = logging.getLogger(__name__)
log = logger


def read_edge_list(
    input_data_path: str,
    delimiter: str = "\t",
    col_names: List = ["src", "dst"],
    dtype: List = ["int32", "int32"],
    reader: str = "cudf",
):
    """Read edge list stored in txt/csv file into dask/dask_cudf for large graphs
    or pandas/cudf for smaller graphs as specified by `reader`.
    Assumes graph is stored as edge list where each row is a (src, dst) tuple
    in csv form.

    Args:
        delimiter (str): delimiter value
        col_names (List[str]): column names
        dtypes (List): dtypes of columns
        dask_cudf (bool): flag to either read edge list using `dask_cudf` or `cudf`.
        default is `dask_cudf`

    Returns:
        `DataFrame` containing edge list
    """

    if reader == "dask_cudf":
        df_reader = DFReader.get_dask_reader()
        try:
            chunksize = dask_cugraph.get_chunksize(input_data_path)
        except:
            chunksize = int(1e6)  # may cause dask error
        e_list = df_reader.read_csv(
            input_data_path,
            chunksize=chunksize,
            delimiter=delimiter,
            names=col_names,
            dtype=dtype,
        )
    elif reader == "cudf" or reader == "pandas":
        df_reader = DFReader.get_df_reader()
        e_list = df_reader.read_csv(
            input_data_path, delimiter=delimiter, names=col_names, dtype=dtype
        )
    else:
        raise ValueError(
            f"{reader} is not supported, must be one of \ {READERS}"
        )

    return e_list


def write_csv(data: pd.DataFrame, save_path: str) -> None:
    log.info(f"writing to file {save_path}")
    data = cudf.DataFrame(data)
    data.to_csv(save_path, index=False)


def dump_generated_graph_to_txt(path, graph):
    f = open(path, "w")
    for e in graph:
        line = "\t".join(str(v) for v in e)
        f.write(line + "\n")
    f.close()


def _generate_samples(
    gen,
    n_samples: int,
    fname: str,
    save_path: str,
    post_gen_fn: Callable = None,
    queue=None,
    i: int = 0,
):
    """
        MP sample generation fn
    """
    ext = str(i)
    fp = save_path / f"{fname}_{ext}.csv"
    samples = gen.sample(n_samples)
    if post_gen_fn is not None:
        samples = post_gen_fn(samples)
    if queue is not None:
        queue.put(samples)
    else:
        write_csv(samples, fp)
    gc.collect()
    return fp


def pass_through(x):
    return x


def chunk_sample_generation(
    gen,
    n_samples: int,
    save_path: str,
    fname: str,
    post_gen_fn: Callable = pass_through,
    num_workers: int = 1,
) -> List[Path]:
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
        num_workers (int): number of workers to speed up generation
        using multiprocessing
    Returns:
        None
    """

    n_samples = int(n_samples)
    # - check if mem available
    gc.collect()
    mem_avail = psutil.virtual_memory().available
    emp_n = 1000
    est_samples = gen.sample(emp_n)
    mem_usage = est_samples.memory_usage(index=True, deep=True).sum()
    est_mem = (mem_usage / emp_n) * n_samples

    # - path
    save_path = Path(save_path)
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
        inc = int(min(n_samples // r, 2e6))
        num_iters = n_samples / inc
        if num_iters - n_samples // inc > 0.0:
            num_iters += 1
        num_iters = int(num_iters)

        queue = None
        manager = None

        generate_samples_p = partial(
            _generate_samples, gen, inc, fname, save_path, post_gen_fn, queue
        )
        if num_workers > 1:

            try:
                torch.multiprocessing.set_start_method("spawn", force=True)
            except RuntimeError:
                import pdb

                pdb.set_trace()

            with Pool(processes=num_workers) as pool:
                file_paths = list(
                    tqdm(
                        pool.imap(generate_samples_p, range(0, num_iters)),
                        total=num_iters,
                    )
                )
                # queue.put('kill')
                pool.close()
                pool.join()
        else:
            for i in tqdm(
                range(0, n_samples, inc), desc="Generating features..."
            ):
                file_paths.append(generate_samples_p(i))

    return file_paths


def write_csv_file_listener(save_path: str, save_name: str, queue):
    KILL_SIG = "kill"
    save_path = Path(save_path) / f"{save_name}.csv"
    first_file = True
    while True:
        # - keep listening until `kill` signal
        m = queue.get()
        if m == KILL_SIG:
            break
        elif type(m) == pd.DataFrame:
            if first_file:
                m.to_csv(save_path, index=False, header=True)
                first_file = False
            else:
                m.to_csv(save_path, mode="append", index=False, header=False)
        else:
            raise Exception(f"{m} is not supported")


def merge_csv_files(
    file_paths: list,
    save_path: str,
    save_name: str = "samples",
    header: bool = True,
    remove_original_files=True,
) -> None:

    """
    Merges CSV files into a single large CSV file

    Args:
        file_paths (str): a list of paths to individual csv files
        save_path (str): a path to directory to save merged csv file
        save_name (str): file name of merged csv file
        Returns:
        None
    """

    save_path = Path(save_path)
    record_header = False

    if header:
        record_header = True

    with open(save_path / f"{save_name}", "w") as out_file:
        for i, fp in enumerate(tqdm(file_paths)):
            with open(fp, "r") as csv:
                for i, l in enumerate(csv):
                    if i == 0 and record_header:
                        out_file.write(l + "\n")
                        record_header = False
                        continue
                    elif i == 0:
                        continue
                    else:
                        out_file.write(l + "\n")

    if remove_original_files:
        for f in file_paths:
            os.remove(f)


def get_generator_timing(
    gen,
    low: int = 0,
    high: int = 1_000_000,
    n_steps: int = 10,
    num_repeat: int = 1,
) -> dict:
    """Runs generator `gen` with different number of samples as
    defined by [`low`, `high`] using `n_steps` linear space in that range.

    Args:
        gen: generator to sample from
        low (int): lowest number of samples to sample,
        must be greater than 0
        high (int): highest number of samples to sample
        n_steps (int): number of steps to interpolate
        between the range [`low`, `high`]
        num_repeat (int): number of times to repeat experiment

     Returns:
         output: dict[n_samples] = <execution time> ms
    """
    assert hasattr(gen, "sample"), "generator must implement `sample` function"
    assert num_repeat >= 1, "`num_repeat` must be greater than"
    assert low < high, "`low` must be less than `high`"

    n_samples = np.linspace(low, high, n_steps)
    n_samples = list(map(int, n_samples))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    output = dict()
    for n_sample in n_samples:
        time = list()
        for i in range(num_repeat):
            try:
                start.record()
                gen.sample(n_sample)
                end.record()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"could not generate {n_sample} samples, exception: {e}")
                output[n_sample] = float("inf")
                break
            time.append(start.elapsed_time(end))
        avg_time = np.mean(time)
        std_time = np.std(time)
        output[n_sample] = {"avg": avg_time, "std": std_time}
    return output
