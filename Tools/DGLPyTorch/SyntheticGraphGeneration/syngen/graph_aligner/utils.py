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
from pathlib import Path, PosixPath
from typing import List, Union

import cudf
import cupy
import pandas as pd
import torch
from tqdm import tqdm

from syngen.utils.types import ColumnType
from syngen.utils.cugraph import import_cugraph


def get_graph(df: cudf.DataFrame, src="src", dst="dst"):
    """Construct directed graph

    Args:
        df (DataFrameType): dataframe containing edge info
        src (str): source node column name
        dst (str): destination node column name

    Returns:
        `cugraph.DiGraph`
    """
    cugraph = import_cugraph()
    graph = cugraph.DiGraph()
    graph.from_cudf_edgelist(df, source=src, destination=dst)
    return graph


def merge_dfs(dfs, **kwargs):
    """merge a list of dataframes on a particular column

    Args:
        dfs (DataFrame): list of dataframes to merge on
        kwargs (dict): key-word arguments to pass to DataFrame `merge` function
    """
    if "on" not in kwargs:
        kwargs["on"] = "vertex"
    if "how" not in kwargs:
        kwargs["how"] = "outer"

    df = dfs[0]
    for i in range(1, len(dfs)):
        df = df.merge(dfs[i], **kwargs)
    return df


def get_features(
    df,
    G,
    src: str = "src",
    dst: str = "dst",
    pagerank_kwargs: dict = {"tol": 1e-4},
):
    """Extract structural features from graph `G`
    features extracted: katz_centrality, out degree, pagerank

    Args:
        df (cudf.DataFrame): data containg edge list informatoin
        G (cugraph.DiGraph): cuGraph graph descriptor containing connectivity information
        from df.
        src (str): source node column name.
        dst (dst): destination node column name.
        pagerank_kwargs (dict): page rank function arguments to pass.
    """
    # - pagerank feat
    cugraph = import_cugraph()

    pr_df = cugraph.pagerank(G, **pagerank_kwargs)
    # - out-degree feat
    degree_src_df = df.groupby(src).count()
    degree_src_df = degree_src_df.reset_index().rename(
        columns={src: "vertex", dst: "out_degree"}
    )

    # - in-degree feat
    degree_dst_df = df.groupby(dst).count()
    degree_dst_df = degree_dst_df.reset_index().rename(
        columns={dst: "vertex", src: "in_degree"}
    )

    # - katz feat
    katz_df = cugraph.katz_centrality(G, tol=1e-2, alpha=1e-3)

    return [pr_df, degree_src_df, degree_dst_df, katz_df]


def merge_graph_vertex_feat(old, new):
    if old is None:
        return new
    merged_df = old.merge(new, on=['vertex'], how='outer')
    merged_df = merged_df.fillna(0)
    return merged_df


def chunk_pd_save(
    df: pd.DataFrame,
    save_path: Union[str, PosixPath],
    chunk_size: Union[int, float],
):
    """Chunks a large dataframe and casts to a cudf for faster save

    Args:
        df (pdDataFrame): dataframe object to dump data
        save_path (str): data path to dump chunks
        chunk_size (int): size of the chunks
    """

    save_path = Path(save_path)
    num_rows = len(df)

    if not save_path.exists():
        os.makedirs(save_path)

    if chunk_size > 0.0 <= 1.0:
        chunk_size = int(num_rows * chunk_size)
    else:
        chunk_size = int(chunk_size)

    for i in tqdm(range(num_rows // chunk_size - 1)):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        chunk_cudf = cudf.from_pandas(chunk_df)
        chunk_cudf.to_parquet(save_path / f"{i}_chunk.parquet", index=False)


def z_norm(series, meta=None, compute=False):
    """applies z-normalization (x - mu) / std"""
    if meta:
        mean = meta["mean"]
        std = meta["std"]
    else:
        mean = series.mean()
        std = series.std()

    out = (series - mean) / std
    return out, {"mean": mean, "std": std}


def categorify(series, meta=None, compute=False):
    """Converts categorical to ordinal"""
    cat_codes = series.astype("category").cat.codes
    return cat_codes, {}


def get_preproc_fn(name: str):
    """Preprocessing map function"""
    PREPROC_FN_MAP = {"z_norm": z_norm, "categorify": categorify}
    return PREPROC_FN_MAP[name]


def get_preproc_dict(feature_types: dict):
    """Apply preprocessing functions to each column type specified in `feature_types` """
    preproc_dict = {}
    for feat, type_ in feature_types.items():
        if type_ == ColumnType.CONTINUOUS:
            preproc_dict[feat] = {"type": type_, "preproc": "z_norm"}
        elif type_ == ColumnType.CATEGORICAL:
            preproc_dict[feat] = {"type": type_, "preproc": "categorify"}
    return preproc_dict


def spread_ranks(ranks):
    vals = cupy.unique(ranks)
    rr = 0
    for v in vals:
        m = ranks == v
        num_v = cupy.sum(m)
        idx_range = cupy.arange(0, cupy.sum(m))
        ranks[m] = ranks[m] + idx_range + rr
        rr += num_v
    return ranks
