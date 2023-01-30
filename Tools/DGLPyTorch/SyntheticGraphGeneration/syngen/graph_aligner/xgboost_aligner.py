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

import pickle
import logging
import os
import warnings
from collections import defaultdict
from pathlib import PosixPath
from typing import Dict, Union

import cudf
import cupy
import dask_cudf
import numpy as np
import pandas as pd
import xgboost
from tqdm import tqdm

try:
    from cuml.preprocessing import LabelEncoder
    from pylibraft.random import rmat  # rmat needs to be imported before cuml
except ImportError:
    from sklearn.preprocessing import OrdinalEncoder as LabelEncoder

from syngen.graph_aligner.base_graph_aligner import BaseGraphAligner
from syngen.graph_aligner.utils import (
    get_features,
    get_graph,
    get_preproc_dict,
    get_preproc_fn,
    merge_dfs,
    spread_ranks,
)
from syngen.utils.types import ColumnType, DataFrameType, MetaData
from syngen.utils.utils import df_to_cudf, df_to_dask_cudf, df_to_pandas

# - suppress numba in debug mode
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

warnings.filterwarnings('ignore')


class XGBoostAligner(BaseGraphAligner):
    """Aligns two graphs via correlating structural graph features
    and tabular features using a xgboost predictor.


    Args:
        features_to_correlate_edge (`Dict[str, ColumnType]`): features to correlate
        with graph structure used for ranking edges. The dict should contain feature
        features_to_correlate_node (`Dict[str, ColumnType]`): features to correlate
    with graph structure used for ranking edges. The dict should contain feature
        generated_graph_path: `Union[PosixPath, str]`
            path to the generated graph data stored as edge list.
            Assumes data is stored as .txt
        generated_table_data_path: `Union[PosixPath, str]`
            path to the generated tabular data
        xgboost_params: `dict`
            key-value parameters to pass to `xgboost.train`. To use
            different parameters for each feature pass a
            `dict` of `dict` corresponding to each feature,
            with keys as the feature name and values as the xgboost_params.
        num_boost_round: `dict` or int
            number of boosting rounds for xgboost. The same `num_boost_round`
            is used for all features unless a `dict` with keys as feature name
            and values as `num_boost_round` is passed.
    """

    def __init__(
        self,
        features_to_correlate_edge: Dict[str, ColumnType] = None,
        features_to_correlate_node: Dict[str, ColumnType] = None,
        xgboost_params: Union[Dict[str, dict], dict] = {
            "learning_rate": 0.1,
            "colsample_bytree": 0.3,
            "max_depth": 5,
            "n_estimators": 10,
            "alpha": 10,
            "silent": 0,
            "verbose_eval": False,
            "tree_method": "gpu_hist",
        },
        num_boost_round: Union[Dict[str, int], int] = 10,
        device: int = 0,
        batch_size: int = 1000,
        topk: int = 4,
        use_dask=False,
        **kwargs,
    ):
        self.features_to_correlate_node = features_to_correlate_node
        self.features_to_correlate_edge = features_to_correlate_edge
        self.xgboost_params = xgboost_params
        self.num_boost_round = num_boost_round
        self.device = device
        self.batch_size = batch_size
        self.topk = topk
        self.use_dask = use_dask and cupy.cuda.runtime.getDeviceCount() > 1

        self.src_col = None
        self.dst_col = None
        self.col_maps_edge = None
        self.col_maps_node = None
        self.edge_columns = None
        self.node_columns = None

    def fit(
        self,
        data: Dict[str, DataFrameType],
        src_col: str,
        dst_col: str,
        node_id_col: str = "id",
        **kwargs,
    ):

        """
        `fit` function for aligner. There are three modes: ["edge", "node", "edge+node"],
        corresponding to graphs with edge features, node features or both respectively.

        Args:
            data (Dict[str, DataFrameType]): dictionary containing `DataFrame` associated with
            `MetaData.EDGE_DATA`, `MetaData.NODE_DATA`,
                - "edge" mode contains `DataFrame` corresponding to
                both graph and feature information,
                -"node" mode it contains a <"nodes", `DataFrame`>
                with node features and <"edges", `DataFrame`>
                with the corresponding graph [`src_col`, `dst_col`]
                - "both" mode <"edges", `DataFrame`> contains the edges + features
                <"nodes", `DataFrame`> contains the node id's + features
            src_col (str): the column name for source nodes to index
                into the `DataFrames`'s
            dst_col (str): the column name for destination nodes to index
                into the `DataFrames`'s
        """

        self.src_col = src_col
        self.dst_col = dst_col
        self.node_id_col = node_id_col
        edge_features = None
        edge_data = data[MetaData.EDGE_DATA]
        edge_list = edge_data[[self.src_col, self.dst_col]]
        feature_cols = list(
            set(edge_data.columns) - {self.src_col, self.dst_col}
        )
        if len(feature_cols) > 0:
            edge_features = edge_data[feature_cols]
            self.fit_edge(
                edge_list,
                edge_features,
                src_col=self.src_col,
                dst_col=self.dst_col,
            )
        else:
            warnings.warn(
                "Edge data does not contain any features for graph aligner, skipping ..."
            )

        if MetaData.NODE_DATA in data:
            node_data = data[MetaData.NODE_DATA]
            if len(node_data.columns) > 1:
                self.fit_node(
                    edge_list,
                    node_data,
                    src_col=self.src_col,
                    dst_col=self.dst_col,
                    node_id_col=node_id_col,
                )
            else:
                warnings.warn(
                    "Node data does not contain any features for graph aligner, skipping ..."
                )

    def fit_edge(
        self,
        table_el_df: DataFrameType,
        edge_df: DataFrameType,
        src_col: str,
        dst_col: str,
    ):

        # - store params for align function
        self.edge_columns = list(edge_df.columns)
        
        if self.features_to_correlate_edge is None:
            print("`features_to_correlate_edge` was not provided... aligning on all features and treating them as continuous")
            self.features_to_correlate_edge = {col: ColumnType.CONTINUOUS for col in self.edge_columns}

        # - create graph
        table_el_df = df_to_cudf(table_el_df)
        if self.use_dask:
            table_el_df = dask_cudf.from_cudf(
                table_el_df,
                chunksize=min(
                    int(1e6),
                    len(table_el_df) // cupy.cuda.runtime.getDeviceCount(),
                ),
            )
        table_G = get_graph(
            table_el_df, src=src_col, dst=dst_col, from_dask=self.use_dask
        )

        # - extract structural features
        graph_feat_dfs = get_features(
            table_el_df,
            table_G,
            src=src_col,
            dst=dst_col,
            use_dask=self.use_dask,
        )
        graph_feat_df = merge_dfs(graph_feat_dfs, on="vertex")
        graph_feat_df = graph_feat_df.fillna(0)

        graph_feat_df = df_to_pandas(graph_feat_df)
        graph_feat_df = cudf.DataFrame.from_pandas(graph_feat_df)

        graph_feat_df["vertex"] = graph_feat_df["vertex"].values.astype(int)
        graph_feat_df = graph_feat_df.set_index("vertex")

        # - load smaller graph into cpu df
        if self.use_dask:
            table_el_df = pd.DataFrame(
                cupy.asnumpy(table_el_df.values.compute()),
                columns=table_el_df.columns,
            )
        else:
            table_el_df = table_el_df.to_pandas()

        # - preprocess data
        preproc_dict = {}
        preproc_dict = get_preproc_dict(self.features_to_correlate_edge)
        self.meta_dict_edge = defaultdict(None)

        # original tabular data
        for feat, v in preproc_dict.items():
            preproc_fn = get_preproc_fn(v["preproc"])
            edge_df[feat], meta = preproc_fn(edge_df[feat])
            self.meta_dict_edge[feat] = meta

        # - setup train dataset
        src = table_el_df[src_col].values.astype(int)
        dst = table_el_df[dst_col].values.astype(int)

        src_feat = graph_feat_df.loc[src].values
        dst_feat = graph_feat_df.loc[dst].values

        X_train = np.concatenate([src_feat, dst_feat], axis=1).astype(float)

        self.edge_trained_models = {}

        self.col_maps_edge = {}
        for col_name, col_type in self.features_to_correlate_edge.items():
            print(f"Fitting {col_name} ...")
            if col_name in self.xgboost_params:
                xgboost_params = dict(self.xgboost_params[col_name])
            else:
                xgboost_params = dict(self.xgboost_params)
            y_train = edge_df[col_name]
            # - default objective
            if "objective" not in xgboost_params:
                if col_type == ColumnType.CONTINUOUS:
                    xgboost_params["objective"] = "reg:squarederror"
                elif col_type == ColumnType.CATEGORICAL:
                    xgboost_params["objective"] = "multi:softmax"
                    vals = edge_df[col_name]
                    encoder = LabelEncoder()
                    encoder.fit(vals)
                    self.col_maps_edge[col_name] = encoder
                    num_classes = len(encoder.classes_)
                    xgboost_params["num_class"] = num_classes
                    y_train = encoder.transform(y_train)

            y_train = y_train.values
            # - XGBoost
            dtrain = xgboost.DMatrix(X_train, y_train)
            # - train the model
            trained_model = xgboost.train(
                xgboost_params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtrain, "train")],
            )

            self.edge_trained_models[col_name] = trained_model

    def fit_node(
        self,
        table_el_df: DataFrameType,
        node_df: DataFrameType,
        src_col: str,
        dst_col: str,
        node_id_col: str = "id",
    ):
        # - store params for align function
        self.node_columns = list(node_df.columns)
        if self.features_to_correlate_node is None:
            print("`features_to_correlate_node` was not provided... aligning on all features and treating them as continuous")
            self.features_to_correlate_node = {col: ColumnType.CONTINUOUS for col in self.node_columns}
        # - create graph
        table_el_df = df_to_cudf(table_el_df)
        if self.use_dask:
            table_el_df = dask_cudf.from_cudf(
                table_el_df,
                chunksize=min(
                    int(1e6),
                    len(table_el_df) // cupy.cuda.runtime.getDeviceCount(),
                ),
            )
        table_G = get_graph(
            table_el_df, src=src_col, dst=dst_col, from_dask=self.use_dask
        )

        # - extract structural features
        graph_feat_dfs = get_features(
            table_el_df,
            table_G,
            src=src_col,
            dst=dst_col,
            use_dask=self.use_dask,
        )
        graph_feat_df = merge_dfs(graph_feat_dfs, on="vertex")
        graph_feat_df = graph_feat_df.fillna(0)

        graph_feat_df = df_to_pandas(graph_feat_df)
        graph_feat_df = cudf.DataFrame.from_pandas(graph_feat_df)

        graph_feat_df["vertex"] = graph_feat_df["vertex"].values.astype(int)
        graph_feat_df = graph_feat_df.set_index("vertex")

        # - preprocess data
        preproc_dict = {}
        preproc_dict = get_preproc_dict(self.features_to_correlate_node)
        self.meta_dict_node = defaultdict(None)

        # original tabular data
        for feat, v in preproc_dict.items():
            preproc_fn = get_preproc_fn(v["preproc"])
            node_df[feat], meta = preproc_fn(node_df[feat])
            self.meta_dict_node[feat] = meta

        # - setup train dataset
        nodes = node_df[node_id_col].values.astype(int)
        node_feat = graph_feat_df.loc[nodes].values
        X_train = node_feat.astype(float)
        self.node_trained_models = {}
        self.col_maps_node = {}
        for col_name, col_type in self.features_to_correlate_node.items():
            print(f"Fitting {col_name} ...")
            if col_name in self.xgboost_params:
                xgboost_params = dict(self.xgboost_params[col_name])
            else:
                xgboost_params = dict(self.xgboost_params)
            y_train = node_df[col_name]

            # - default objective
            if "objective" not in xgboost_params:
                if col_type == ColumnType.CONTINUOUS:
                    xgboost_params["objective"] = "reg:squarederror"
                elif col_type == ColumnType.CATEGORICAL:
                    xgboost_params["objective"] = "multi:softmax"
                    vals = node_df[col_name]
                    encoder = LabelEncoder()
                    encoder.fit(vals)
                    self.col_maps_node[col_name] = encoder
                    num_classes = len(encoder.classes_)
                    xgboost_params["num_class"] = num_classes
                    y_train = encoder.transform(y_train)

            y_train = y_train.values
            # - XGBoost
            dtrain = xgboost.DMatrix(X_train, y_train)
            trained_model = xgboost.train(
                xgboost_params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtrain, "train")],
            )
            self.node_trained_models[col_name] = trained_model

    def align(
        self, data: Dict[str, DataFrameType], src_col: str, dst_col: str, **kwargs
    ) -> pd.DataFrame:
        """ Align given features onto graph defined in `data[MetaData.EDGE_LIST]`

        Args:
            data (Dict[str, DataFrameType]): dictionary containing graph edge list and edge/node
            features to align. Each stored in `MetaData.EDGE_LIST`, `MetaData.EDGE_DATA`,
            `MetaData.NODE_DATA` correspondingly.
            src_col (str): source column in `MetaData.EDGE_LIST`
            dst_col (str): destination column in `MetaData.EDGE_LIST`

        """
        edge_list = data[MetaData.EDGE_LIST]
        edge_data = edge_list
        node_data = None

        if data.get(MetaData.EDGE_DATA, None) is not None:
            assert hasattr(
                self, "edge_trained_models"
            ), "the `fit` function with edge features \
                must be called before `align`"
            features = data[MetaData.EDGE_DATA]
            edge_data = self._align(
                features,
                edge_list,
                self.features_to_correlate_edge,
                self.col_maps_edge,
                self.edge_trained_models,
                mode="edge",
                src_col=src_col,
                dst_col=dst_col,
            )

        if data.get(MetaData.NODE_DATA, None) is not None:
            assert hasattr(
                self, "node_trained_models"
            ), "the `fit` function with node features \
                must be called before `align`"
            features = data[MetaData.NODE_DATA]
            node_data = self._align(
                features,
                edge_list,
                self.features_to_correlate_node,
                self.col_maps_node,
                self.node_trained_models,
                mode="node",
                dst_col=dst_col,
                src_col=src_col,
            )

        return {MetaData.EDGE_DATA: edge_data, MetaData.NODE_DATA: node_data}

    def _align(
        self,
        features: DataFrameType,
        edges: DataFrameType,
        features_to_correlate: Dict[str, ColumnType],
        col_maps,
        trained_models: Dict[str, xgboost.Booster],
        mode: str,
        src_col: str = "src",
        dst_col: str = "dst",
        **kwargs,
    ) -> pd.DataFrame:
        if mode == "edge":
            assert len(features) >= len(
                edges
            ), "generated features must be greater than number of edges"
        if self.use_dask:
            edges = df_to_dask_cudf(edges)
        topk = min(len(edges), self.topk)
        # - create graph
        generated_G = get_graph(
            edges, src=src_col, dst=dst_col, from_dask=self.use_dask
        )
        graph_feat_dfs = get_features(
            edges,
            generated_G,
            src=src_col,
            dst=dst_col,
            use_dask=self.use_dask,
        )
        graph_feat_df = merge_dfs(graph_feat_dfs, on="vertex")
        graph_feat_df = graph_feat_df.fillna(0)

        preproc_dict = get_preproc_dict(features_to_correlate)

        # - convert to cudf if table can fit in gpu
        graph_feat_df = df_to_cudf(graph_feat_df)
        graph_feat_df["vertex"] = graph_feat_df["vertex"].values.astype(int)
        graph_feat_df = graph_feat_df.set_index("vertex")
        if mode == "node":
            assert len(features) >= len(
                graph_feat_df
            ), "generated features must be greater than number of nodes"

        edges = df_to_pandas(edges)
        all_preds = []

        if mode == "edge":
            split_df = edges
        elif mode == "node":
            split_df = df_to_pandas(graph_feat_df)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        batch_size = self.batch_size
        if len(split_df) // batch_size == 0:
            batch_size = len(split_df)

        chunks = np.array_split(split_df, len(split_df) // batch_size)

        # - predictions & ranking
        for idx, chunk in tqdm(
            enumerate(chunks),
            total=len(chunks),
            desc=f"Aligner - preds {mode}",
        ):
            col_preds = []
            if mode == "edge":
                src_feat = graph_feat_df.loc[chunk[src_col].values].values
                dst_feat = graph_feat_df.loc[chunk[dst_col].values].values
                X_test = np.concatenate([src_feat, dst_feat], axis=1)
                dtest = xgboost.DMatrix(X_test)
            elif mode == "node":
                node_feat = chunk.values
                X_test = node_feat.astype(float)
                dtest = xgboost.DMatrix(X_test)

            for col_name, col_type in features_to_correlate.items():
                preds = trained_models[col_name].predict(dtest)
                col_preds.append(preds.reshape(-1, 1))
            col_preds = np.concatenate(col_preds, axis=1)
            all_preds.append(col_preds)

        all_preds = np.concatenate(all_preds, axis=0)
        all_preds = cupy.asarray(all_preds)

        target_cols = list(features_to_correlate.keys())
        y_generated = []
        for col_name, col_type in features_to_correlate.items():
            preproc_fn = None
            if preproc_dict:
                try:
                    preproc_fn = get_preproc_fn(
                        preproc_dict[col_name]["preproc"]
                    )
                except:
                    pass

            y = features[col_name]
            if self.use_dask:
                y = y.compute()
            if preproc_fn is not None:
                y, _ = preproc_fn(y)
            if col_type == ColumnType.CATEGORICAL:
                y = col_maps[col_name].inverse_transform(y)
                y_generated.append(y)
            else:
                y_generated.append(y)
        y_generated = cudf.concat(y_generated, axis=1).values
        ranks = cupy.zeros((len(split_df), 1))
        if len(target_cols) == 1:
            y_generated = y_generated.reshape(-1)
            target_col = target_cols[0]
            col_type = features_to_correlate[target_col]

            if col_type == ColumnType.CATEGORICAL:
                all_preds = col_maps[target_col].inverse_transform(
                    cudf.Series(all_preds)
                )
                all_preds = all_preds.values
                unique_preds = cupy.unique(all_preds)
                unique_preds = cupy.asnumpy(unique_preds)
                unique_generated = cupy.unique(y_generated)
                present_unique = [
                    up for up in unique_preds if up in unique_generated
                ]
                idxs = cupy.arange(0, len(y_generated))
                pred_assigned = cupy.zeros(len(all_preds), dtype="bool")
                gen_assigned = cupy.zeros(len(y_generated), dtype="bool")
                unassigned_idxs_gen = []
                unassigned_idxs_pred = []

                for up in present_unique:
                    sel_idxs = idxs[y_generated == up]
                    cupy.random.shuffle(sel_idxs)
                    ups_mask = (all_preds == up).squeeze()
                    num_ups = cupy.sum(ups_mask)

                    if len(sel_idxs) > num_ups:
                        r_idxs = sel_idxs[:num_ups]
                        ranks[ups_mask] = r_idxs.reshape(-1, 1)
                        pred_assigned[ups_mask] = True
                        gen_assigned[sel_idxs[:num_ups]] = True
                    else:
                        r_idxs = cupy.where(ups_mask)[0]
                        ra_idxs = r_idxs[: len(sel_idxs)]
                        ranks[ra_idxs] = sel_idxs.reshape(-1, 1)
                        ups_mask[ra_idxs] = False
                        unassigned_idxs = ra_idxs[len(sel_idxs) :]
                        unassigned_idxs_pred.append(unassigned_idxs)
                        pred_assigned[ra_idxs] = True
                        gen_assigned[sel_idxs] = True
                ranks[~pred_assigned] = idxs[~gen_assigned][
                    : cupy.sum(~pred_assigned)
                ].reshape(-1, 1)
            elif col_type == ColumnType.CONTINUOUS:
                y_generated = cupy.ravel(y_generated)
                y_idxsort = cupy.argsort(y_generated)
                y_generated_sorted = y_generated[y_idxsort]
                ranking = cupy.searchsorted(y_generated_sorted, all_preds)
                ranks = y_idxsort[ranking]
                ranks = spread_ranks(ranks)
        elif len(target_cols) > 1:
            y_generated = y_generated / (
                cupy.linalg.norm(y_generated, ord=2, axis=1).reshape(-1, 1)
            )
            chunks = cupy.array_split(all_preds, len(all_preds) // batch_size)
            for idx, chunk in tqdm(
                enumerate(chunks), total=len(chunks), desc=f"Assigning {mode}"
            ):
                idxs = cupy.ones((len(y_generated),), dtype=cupy.bool)
                chunk = chunk / cupy.linalg.norm(chunk, ord=2, axis=1).reshape(
                    -1, 1
                )
                sim = cupy.einsum("ij,kj->ik", chunk, y_generated)
                chunk_ranks = cupy.argsort(sim, axis=1)[:, -topk:]
                rand_sel = cupy.random.randint(0, topk, len(chunk_ranks))
                chunk_ranks = chunk_ranks[
                    cupy.arange(len(chunk_ranks)), rand_sel
                ]
                cupy.put(idxs, chunk_ranks, False)
                y_generated = y_generated[idxs]
                ranks[
                    idx * batch_size : idx * batch_size + len(chunk)
                ] = chunk_ranks.reshape(-1, 1)

        ranks = cupy.asnumpy(ranks)
        ranks = ranks.squeeze()

        print("Finished ranking, overlaying features on generated graph...")
        if mode == "edge":
            if self.use_dask:
                features = features.compute()
            features = features.to_pandas()
            ranks[ranks >= len(features)] = len(features) - 1
            columns = list(self.edge_columns)
            try:
                columns.remove(self.src_col)
            except ValueError as e:
                pass
            try:
                columns.remove(self.dst_col)
            except ValueError as e:
                pass
            features = features.iloc[ranks][columns].reset_index(drop=True)
            overlayed_df = pd.concat([edges, features], axis=1)
            overlayed_df.rename(
                columns={src_col: self.src_col, dst_col: self.dst_col},
                inplace=True,
            )
        elif mode == "node":
            if self.use_dask:
                features = features.compute()
            features = features.iloc[ranks].reset_index(drop=True)
            features[self.node_id_col] = split_df.index
            overlayed_df = features

        return overlayed_df

    def save(self, save_dir: Union[PosixPath, str]):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if hasattr(self, "edge_trained_models"):
            for k, v in self.edge_trained_models.items():
                v.save_model(
                    os.path.join(save_dir, f"{k}_xgb_aligner_edge.json")
                )

        if hasattr(self, "node_trained_models"):
            for k, v in self.node_trained_models.items():
                v.save_model(
                    os.path.join(save_dir, f"{k}_xgb_aligner_node.json")
                )

        meta_data = {
            "features_to_correlate_edge": self.features_to_correlate_edge,
            "features_to_correlate_node": self.features_to_correlate_node,
            "xgboost_params": self.xgboost_params,
            "num_boost_round": self.num_boost_round,
            "device": self.device,
            "batch_size": self.batch_size,
            "topk": self.topk,
            "use_dask": self.use_dask,
            "fitted_data": {
                "src_col": self.src_col,
                "dst_col": self.dst_col,
                "col_maps_edge": self.col_maps_edge,
                "col_maps_node": self.col_maps_node,
                "edge_columns": self.edge_columns,
                "node_columns": self.node_columns,
            }
        }
        with open(os.path.join(save_dir, "xgb_aligner_meta.pkl"), "wb") as file_handler:
            pickle.dump(meta_data, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, dir_path: Union[PosixPath, str]):
        with open(os.path.join(dir_path, "xgb_aligner_meta.pkl"), "rb") as file_handler:
            meta_data = pickle.load(file_handler)

        fitted_data = meta_data['fitted_data']
        meta_data.pop('fitted_data')

        instance = cls(**meta_data)
        for k, v in fitted_data.items():
            setattr(instance, k, v)

        files = os.listdir(dir_path)
        edge_files = [f for f in files if "xgb_aligner_edge" in f]
        instance.edge_trained_models = {}
        for ef in edge_files:
            xgb_model = xgboost.Booster()
            xgb_model.load_model(os.path.join(dir_path, ef))
            k = ef.split("_xgb_aligner_edge")[0]  # - same format as `save`
            instance.edge_trained_models[k] = xgb_model

        node_files = [f for f in files if "xgb_aligner_node" in f]
        instance.node_trained_models = {}
        for nf in node_files:
            xgb_model = xgboost.Booster()
            xgb_model.load_model(os.path.join(dir_path, nf))
            k = nf.split("_xgb_aligner_node")[0]  # - same format as `save`
            instance.node_trained_models[k] = xgb_model
        return instance

    @staticmethod
    def add_args(parser):
        import json

        parser.add_argument(
            "--features-to-correlate-node",
            type=json.loads,
            default=None,
            help="Node feature columns to use to train `XGBoostAligner`. Must be provided in a dict format {<feature-column-name>: <column-type>}, where <column-type> is an enum of type `ColumnType` (see syngen/utils/types/column_type.py).",
        )
        parser.add_argument(
            "--features-to-correlate-edge",
            type=json.loads,
            default=None,
            help="Edge feature columns to use to train `XGBoostAligner`. Must be provided in a dict format {<feature-column-name>: <column-type>}, where <column-type> is an enum of type `ColumnType` (see syngen/utils/types/column_type.py).",
        )
