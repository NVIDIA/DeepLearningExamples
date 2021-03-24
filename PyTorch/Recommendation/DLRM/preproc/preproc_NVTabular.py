# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
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

"""Preprocess Criteo 1TB Click Logs dataset with frequency thresholding and filling missing values.

This script accepts input in either tsv or parquet format.
"""

import argparse
from collections import OrderedDict
import json
import os
import subprocess
from time import time
from typing import List, Optional

import numpy as np
import nvtabular as nvt
import rmm
import cudf
from dask.base import tokenize
from dask.dataframe.io.parquet.utils import _analyze_paths
from dask.delayed import Delayed
from dask.distributed import Client
from dask.highlevelgraph import HighLevelGraph
from dask.utils import natural_sort_key
from dask_cuda import LocalCUDACluster
from fsspec.core import get_fs_token_paths
from nvtabular import Workflow
from nvtabular.io import Dataset, Shuffle
from nvtabular.utils import device_mem_size
from nvtabular.ops import Normalize, Categorify, LogOp, FillMissing, Clip, get_embedding_sizes, \
    LambdaOp
from cudf.io.parquet import ParquetWriter

CRITEO_CONTINUOUS_COLUMNS = [f'_c{x}' for x in range(1, 14)]
CRITEO_CATEGORICAL_COLUMNS = [f'_c{x}' for x in range(14, 40)]
CRITEO_CLICK_COLUMNS = ['_c0']
COLUMNS = CRITEO_CONTINUOUS_COLUMNS + CRITEO_CATEGORICAL_COLUMNS + CRITEO_CLICK_COLUMNS
CRITEO_TRAIN_DAYS = list(range(0, 23))

ALL_DS_MEM_FRAC = 0.04
TRAIN_DS_MEM_FRAC = 0.045
TEST_DS_MEM_FRAC = 0.3
VALID_DS_MEM_FRAC = 0.3

def _pool(frac=0.8):
    initial_pool_size = frac * device_mem_size()
    if initial_pool_size % 256 != 0:
        new_initial_pool_size = initial_pool_size // 256 * 256
        print(
            f"Initial pool size for rmm has to be a multiply of 256. Got {initial_pool_size}, reducing to {new_initial_pool_size}")
        initial_pool_size = new_initial_pool_size

    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=initial_pool_size,
    )


def _convert_file(path, name, out_dir, gpu_mem_frac, fs, cols, dtypes):
    fn = f"{name}.parquet"
    out_path = fs.sep.join([out_dir, f"{name}.parquet"])
    writer = ParquetWriter(out_path, compression=None)
    for gdf in nvt.Dataset(
        path,
        engine="csv",
        names=cols,
        part_memory_fraction=gpu_mem_frac,
        sep='\t',
        dtypes=dtypes,
    ).to_iter():
        writer.write_table(gdf)
        del gdf
    md = writer.close(metadata_file_path=fn)
    return md


def _write_metadata(md_list, fs, path):
    if md_list:
        metadata_path = fs.sep.join([path, "_metadata"])
        _meta = (
            cudf.io.merge_parquet_filemetadata(md_list)
            if len(md_list) > 1
            else md_list[0]
        )
        with fs.open(metadata_path, "wb") as f:
            _meta.tofile(f)
    return True


def convert_criteo_to_parquet(
    input_path: str,
    output_path: str,
    client,
    gpu_mem_frac: float = 0.05,
):
    print("Converting tsv to parquet files")
    if not output_path:
        raise RuntimeError("Intermediate directory must be defined, if the dataset is tsv.")
    os.makedirs(output_path, exist_ok=True)

    # split last day into two parts
    number_of_lines = int(
        subprocess.check_output((f'wc -l {os.path.join(input_path, "day_23")}').split()).split()[0])
    valid_set_size = number_of_lines // 2
    test_set_size = number_of_lines - valid_set_size

    with open(os.path.join(input_path, "day_23.part1"), "w") as f:
        subprocess.run(['head', '-n', str(test_set_size), str(os.path.join(input_path, "day_23"))], stdout=f)

    with open(os.path.join(input_path, "day_23.part2"), "w") as f:
        subprocess.run(['tail', '-n', str(valid_set_size), str(os.path.join(input_path, "day_23"))], stdout=f)

    fs = get_fs_token_paths(input_path, mode="rb")[0]
    file_list = [
        x for x in fs.glob(fs.sep.join([input_path, "day_*"]))
        if not x.endswith("parquet")
    ]
    file_list = sorted(file_list, key=natural_sort_key)
    name_list = _analyze_paths(file_list, fs)[1]

    cols = CRITEO_CLICK_COLUMNS + CRITEO_CONTINUOUS_COLUMNS + CRITEO_CATEGORICAL_COLUMNS

    dtypes = {}
    dtypes[CRITEO_CLICK_COLUMNS[0]] = np.int64
    for x in CRITEO_CONTINUOUS_COLUMNS:
        dtypes[x] = np.int64
    for x in CRITEO_CATEGORICAL_COLUMNS:
        dtypes[x] = "hex"

    dsk = {}
    token = tokenize(file_list, name_list, output_path, gpu_mem_frac, fs, cols, dtypes)
    convert_file_name = "convert_file-" + token
    for i, (path, name) in enumerate(zip(file_list, name_list)):
        key = (convert_file_name, i)
        dsk[key] = (_convert_file, path, name, output_path, gpu_mem_frac, fs, cols, dtypes)

    write_meta_name = "write-metadata-" + token
    dsk[write_meta_name] = (
        _write_metadata,
        [(convert_file_name, i) for i in range(len(file_list))],
        fs,
        output_path,
    )
    graph = HighLevelGraph.from_collections(write_meta_name, dsk, dependencies=[])
    conversion_delayed = Delayed(write_meta_name, graph)

    if client:
        conversion_delayed.compute()
    else:
        conversion_delayed.compute(scheduler="synchronous")

    print("Converted")


def save_model_size_config(workflow: Workflow, output_path: str):
    embeddings = {}
    for k, v in get_embedding_sizes(workflow).items():
        embeddings[k] = v[0] - 1  # we have to subtract one, as the model expects to get a maximal id for each category

    ordered_dict = OrderedDict()
    for k, v in sorted(list(embeddings.items()), key=lambda x: x[0]):
        ordered_dict[k] = v
    with open(os.path.join(output_path, "model_size.json"), 'w') as file:
        file.write(json.dumps(ordered_dict))


def preprocess_criteo_parquet(
    input_path: str,
    output_path: str,
    client,
    frequency_threshold: int,
):
    train_days = [str(x) for x in CRITEO_TRAIN_DAYS]
    train_files = [
        os.path.join(input_path, x)
        for x in os.listdir(input_path)
        if x.startswith("day") and x.split(".")[0].split("_")[-1] in train_days
    ]
    valid_file = os.path.join(input_path, "day_23.part2.parquet")
    test_file = os.path.join(input_path, "day_23.part1.parquet")

    all_set = train_files + [valid_file] + [test_file]

    print(all_set, train_files, valid_file, test_file)
    print("Creating Workflow Object")

    workflow = Workflow(
        cat_names=CRITEO_CATEGORICAL_COLUMNS,
        cont_names=CRITEO_CONTINUOUS_COLUMNS,
        label_name=CRITEO_CLICK_COLUMNS
    )

    # We want to assign 0 to all missing values, and calculate log(x+3) for present values
    # so if we set missing values to -2, then the result of log(1+2+(-2)) would be 0
    workflow.add_cont_feature([
        FillMissing(fill_val=-2.0),
        LambdaOp(op_name='Add3ButMinusOneCauseLogAddsOne', f=lambda col, _: col.add(2.0)),
        LogOp(),  # Log(1+x)
    ])

    workflow.add_cat_preprocess(
        Categorify(freq_threshold=frequency_threshold, out_path=output_path)
    )

    workflow.finalize()

    print("Creating Dataset Iterator")
    all_ds = Dataset(all_set, engine="parquet", part_mem_fraction=ALL_DS_MEM_FRAC)
    trains_ds = Dataset(train_files, engine="parquet", part_mem_fraction=TRAIN_DS_MEM_FRAC)
    valid_ds = Dataset(valid_file, engine="parquet", part_mem_fraction=TEST_DS_MEM_FRAC)
    test_ds = Dataset(test_file, engine="parquet", part_mem_fraction=VALID_DS_MEM_FRAC)

    print("Running apply")
    out_train = os.path.join(output_path, "train")
    out_valid = os.path.join(output_path, "validation")
    out_test = os.path.join(output_path, "test")

    start = time()
    workflow.update_stats(all_ds)
    print(f"Gathering statistics time: {time() - start}")

    start = time()
    workflow.apply(
        trains_ds,
        record_stats=False,
        output_path=out_train
    )
    print(f"train preprocess time: {time() - start}")

    start = time()
    workflow.apply(
        valid_ds,
        record_stats=False,
        output_path=out_valid
    )
    print(f"valid preprocess time: {time() - start}")

    start = time()
    workflow.apply(
        test_ds,
        record_stats=False,
        output_path=out_test
    )
    print(f"test preprocess time: {time() - start}")

    save_model_size_config(workflow, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "input_dir",
        help="directory with either csv or parquet dataset files inside"
    )
    parser.add_argument(
        "output_dir",
        help="directory to save preprocessed dataset files"
    )
    parser.add_argument(
        "--intermediate_dir",
        required=False,
        default=None,
        help="directory for converted to parquet dataset files inside"
    )
    parser.add_argument(
        "--devices",
        required=True,
        help="available gpus, separated with commas; e.g 0,1,2,3"
    )
    parser.add_argument(
        "--freq_threshold",
        required=False,
        default=15,
        help="frequency threshold for categorical can be int or dict {column_name: threshold}"
    )
    parser.add_argument(
        "--pool",
        required=False,
        default=False,
        help="bool value to use a RMM pooled allocator"
    )

    args = parser.parse_args()

    args.devices = args.devices.split(",")

    return args


def is_input_parquet(input_dir: str):
    for f in os.listdir(input_dir):
        if 'parquet' in f:
            return True
    return False


def start_local_CUDA_cluster(devices, pool):
    if len(devices) > 1:
        cluster = LocalCUDACluster(
            n_workers=len(devices),
            CUDA_VISIBLE_DEVICES=",".join(str(x) for x in devices),
        )
        client = Client(cluster)
        if pool:
            client.run(_pool)
    elif pool:
        _pool()
    return client


def main():
    args = parse_args()

    client = start_local_CUDA_cluster(args.devices, args.pool)

    if not is_input_parquet(args.input_dir):
        convert_criteo_to_parquet(
            input_path=args.input_dir,
            output_path=args.intermediate_dir,
            client=client,
        )
        args.input_dir = args.intermediate_dir

    print("Preprocessing data")
    preprocess_criteo_parquet(
        input_path=args.input_dir,
        output_path=args.output_dir,
        client=client,
        frequency_threshold=int(args.freq_threshold),
    )
    print("Done")


if __name__ == '__main__':
    main()
