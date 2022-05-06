# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
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

import ast
import json
import multiprocessing
from typing import Dict

import cudf
import dask.dataframe

JSON_READ_BLOCKSIZE = 100_000_000


def _read_metadata_line(line: str) -> Dict[str, str]:
    dict_line = ast.literal_eval(line)
    return {"item": dict_line["asin"], "cat": dict_line["categories"][0][-1]}


def load_metadata(
    path: str,
    n_proc: int,
) -> cudf.DataFrame:
    metadata = []
    with open(path) as fp:
        metadata = fp.readlines()

    with multiprocessing.Pool(n_proc) as pool:
        metadata = pool.map(_read_metadata_line, metadata)

    df = cudf.DataFrame(metadata)
    return df


def _read_json(*args, **kwargs):
    df = cudf.read_json(*args, **kwargs)
    df = df.rename(columns={"reviewerID": "user", "asin": "item"})
    df = df[["user", "item", "unixReviewTime"]]
    return df


def load_review_data(
    path: str,
    num_workers: int,
    dask_scheduler="processes",
) -> cudf.DataFrame:
    ddf = dask.dataframe.read_json(
        path,
        lines=True,
        blocksize=JSON_READ_BLOCKSIZE,
        engine=_read_json,
    )
    df = ddf.compute(scheduler=dask_scheduler, num_workers=num_workers)
    return df


def save_metadata(
    number_of_items: int,
    number_of_categories: int,
    number_of_users: int,
    output_path: str,
):
    data = {
        "cardinalities": [
            {"name": "uid", "value": number_of_users},
            {"name": "item", "value": number_of_items},
            {"name": "cat", "value": number_of_categories},
        ],
    }
    with open(output_path, "w") as fp:
        json.dump(data, fp)
