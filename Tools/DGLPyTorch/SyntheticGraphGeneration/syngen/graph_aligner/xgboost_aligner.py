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
from typing import Dict, Union, Literal

import cudf
import cupy
import numpy as np
import pandas as pd
import xgboost

try:
    from cuml.preprocessing import LabelEncoder
    from pylibraft.random import rmat  # rmat needs to be imported before cuml
except ImportError:
    from sklearn.preprocessing import OrdinalEncoder as LabelEncoder

from syngen.graph_aligner.base_graph_aligner import BaseGraphAligner
from syngen.graph_aligner.utils import (
    get_graph,
    get_preproc_dict,
    get_preproc_fn,
    merge_dfs,
    spread_ranks, merge_graph_vertex_feat,
)

from syngen.graph_aligner.utils import get_features as default_features


from syngen.utils.types import ColumnType, DataFrameType, MetaData
from syngen.utils.utils import df_to_cudf, df_to_pandas

# - suppress numba in debug mode
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

warnings.filterwarnings('ignore')


class XGBoostAligner(BaseGraphAligner):
    """Aligns two graphs via correlating structural graph features
    and tabular features using a xgboost predictor.

    Args:
        xgboost_params: `dict`
            key-value parameters to pass to `xgboost.train`. To use
            different parameters for each feature pass a
            `dict` of `dict` corresponding to each feature,
            with keys as the feature name and values as the xgboost_params.
        num_boost_round: `dict` or int
            number of boosting rounds for xgboost. The same `num_boost_round`
            is used for all features unless a `dict` with keys as feature name
            and values as `num_boost_round` is passed.
        batch_size: int
            the size of the chunk during the alignment process
        topk: int
            the number of candidates with the highest ranks to be chosen from during alignment
    """

    def __init__(
        self,
        xgboost_params: Union[Dict[str, dict], dict] = {
            "learning_rate": 0.1,
            "colsample_bytree": 0.3,
            "max_depth": 5,
            "n_estimators": 100,
            "alpha": 10,
            "tree_method": "gpu_hist",
        },
        num_boost_round: Union[Dict[str, int], int] = 10,
        batch_size: int = 100000,
        topk: int = 4,
        get_features=default_features,
        verbose=False,
        **kwargs,
    ):
        self.xgboost_params = xgboost_params
        self.num_boost_round = num_boost_round
        self.batch_size = batch_size
        self.topk = topk

        self.col_maps_edge = None
        self.col_maps_node = None
        self.get_features = get_features
        self.verbose = verbose
        self.xgboost_params['verbosity'] = int(xgboost_params.get('verbosity', self.verbose))
        self.xgboost_params['silent'] = int(xgboost_params.get('silent', not self.verbose))

        self.features_to_correlate_edge = None
        self.features_to_correlate_node = None
        self.col_maps_edge = None
        self.col_maps_node = None
        self.meta_dict_edge = None
        self.meta_dict_node = None

        self.edge_trained_models = None
        self.node_trained_models = None

    def _extract_structural_features(self, graphs):
        structural_features = {}

        for graph_name, graph_info in graphs.items():
            is_hetero = graph_info[MetaData.SRC_NODE_TYPE] != graph_info[MetaData.DST_NODE_TYPE]
            if is_hetero:
                offset = graph_info['src_size'] + 10
                graph_info[MetaData.STRUCTURE_DATA][:, 1] = graph_info[MetaData.STRUCTURE_DATA][:, 1] + offset

            edge_list_df = cudf.DataFrame(graph_info[MetaData.STRUCTURE_DATA], columns=["src", "dst"])
            graph = get_graph(edge_list_df, src="src", dst="dst").to_undirected()

            graph_feat_dfs = self.get_features(edge_list_df, graph, src="src", dst="dst")
            graph_feat_df = merge_dfs(graph_feat_dfs, on="vertex")
            graph_feat_df = graph_feat_df.fillna(0)

            if is_hetero:
                src_nodes = graph_feat_df['vertex'] <= graph_info['src_size']
                structural_features[graph_info[MetaData.SRC_NODE_TYPE]] = merge_graph_vertex_feat(
                    structural_features.get(graph_info[MetaData.SRC_NODE_TYPE]),
                    graph_feat_df.loc[src_nodes])

                dst_nodes = graph_feat_df['vertex'] > graph_info['src_size']
                dst_graph_feat_df = graph_feat_df.loc[dst_nodes]
                dst_graph_feat_df["vertex"] -= offset
                structural_features[graph_info[MetaData.DST_NODE_TYPE]] = merge_graph_vertex_feat(
                    structural_features.get(graph_info[MetaData.DST_NODE_TYPE]),
                    dst_graph_feat_df)
                graph_info[MetaData.STRUCTURE_DATA][:, 1] = graph_info[MetaData.STRUCTURE_DATA][:, 1] - offset
            else:
                structural_features[graph_info[MetaData.SRC_NODE_TYPE]] = merge_graph_vertex_feat(
                    structural_features.get(graph_info[MetaData.SRC_NODE_TYPE]), graph_feat_df)
        for _, df in structural_features.items():
            df['vertex'] = df['vertex'].values.astype(int)
            df.set_index('vertex', inplace=True)
        return structural_features

    def fit(
        self,
        graphs,
        node_features,
        edge_features,
        **kwargs,
    ):
        structural_features = self._extract_structural_features(graphs)
        self._fit_node(node_features, structural_features)
        self._fit_edge(edge_features, structural_features, graphs)

    def _fit_edge(
        self,
        edge_features,
        structural_features,
        graphs
    ):
        self.features_to_correlate_edge = {}
        self.edge_trained_models = {}
        self.col_maps_edge = {}
        self.meta_dict_edge = {}

        for edge_name, edge_features_data in edge_features.items():
            self.features_to_correlate_edge[edge_name] = {}
            cat_cols = edge_features_data[MetaData.CATEGORICAL_COLUMNS]
            cont_columns = list(set(edge_features_data[MetaData.FEATURES_LIST]) - set(cat_cols))
            for c in cat_cols:
                self.features_to_correlate_edge[edge_name][c] = MetaData.CATEGORICAL
            for c in cont_columns:
                self.features_to_correlate_edge[edge_name][c] = MetaData.CONTINUOUS

            self.meta_dict_edge[edge_name] = defaultdict(None)
            preproc_dict = get_preproc_dict(self.features_to_correlate_edge[edge_name])

            for feat, v in preproc_dict.items():
                preproc_fn = get_preproc_fn(v["preproc"])
                edge_features_data[MetaData.FEATURES_DATA][feat], meta = \
                    preproc_fn(edge_features_data[MetaData.FEATURES_DATA][feat])
                self.meta_dict_edge[feat] = meta

            graph_info = graphs[edge_name]

            edge_list = graph_info[MetaData.STRUCTURE_DATA]
            src_ids = edge_list[:, 0]
            dst_ids = edge_list[:, 1]

            src_struct_feat = structural_features[graph_info[MetaData.SRC_NODE_TYPE]].loc[src_ids].values
            dst_struct_feat = structural_features[graph_info[MetaData.DST_NODE_TYPE]].loc[dst_ids].values

            X_train = np.concatenate([src_struct_feat, dst_struct_feat], axis=1).astype(float)

            self.edge_trained_models[edge_name] = {}
            self.col_maps_edge[edge_name] = {}

            edge_features_df = cudf.DataFrame.from_pandas(edge_features_data[MetaData.FEATURES_DATA])

            for col_name, col_type in self.features_to_correlate_edge[edge_name].items():
                if col_name in self.xgboost_params:
                    xgboost_params = dict(self.xgboost_params[col_name])
                else:
                    xgboost_params = dict(self.xgboost_params)
                y_train = edge_features_df[col_name]

                if "objective" not in xgboost_params:
                    if col_type == ColumnType.CONTINUOUS:
                        xgboost_params["objective"] = "reg:squarederror"
                    elif col_type == ColumnType.CATEGORICAL:
                        xgboost_params["objective"] = "multi:softmax"
                        vals = edge_features_df[col_name]
                        encoder = LabelEncoder()
                        encoder.fit(vals)
                        self.col_maps_edge[edge_name][col_name] = encoder
                        num_classes = len(encoder.classes_)
                        xgboost_params["num_class"] = num_classes
                        y_train = encoder.transform(y_train)
                y_train = y_train.values
                dtrain = xgboost.DMatrix(X_train, y_train)
                # - train the model
                trained_model = xgboost.train(
                    xgboost_params,
                    dtrain,
                    num_boost_round=self.num_boost_round,
                    evals=[(dtrain, "train")],
                    verbose_eval=self.verbose,
                )
                self.edge_trained_models[edge_name][col_name] = trained_model

    def _fit_node(
        self,
        node_features,
        structural_features
    ):
        self.features_to_correlate_node = {}
        self.node_trained_models = {}
        self.col_maps_node = {}
        self.meta_dict_node = {}

        # fit nodes
        for node_name, node_features_data in node_features.items():
            self.features_to_correlate_node[node_name] = {}
            cat_cols = node_features_data[MetaData.CATEGORICAL_COLUMNS]
            cont_columns = list(set(node_features_data[MetaData.FEATURES_LIST]) - set(cat_cols))

            for c in cat_cols:
                self.features_to_correlate_node[node_name][c] = MetaData.CATEGORICAL
            for c in cont_columns:
                self.features_to_correlate_node[node_name][c] = MetaData.CONTINUOUS

            self.meta_dict_node[node_name] = defaultdict(None)

            preproc_dict = get_preproc_dict(self.features_to_correlate_node[node_name])

            for feat, v in preproc_dict.items():
                preproc_fn = get_preproc_fn(v["preproc"])
                node_features_data[MetaData.FEATURES_DATA][feat], meta = \
                    preproc_fn(node_features_data[MetaData.FEATURES_DATA][feat])
                self.meta_dict_node[feat] = meta

            nodes = structural_features[node_name].index.values.astype(int)
            node_struct_feat = structural_features[node_name].loc[nodes].values
            X_train = node_struct_feat.astype(float)

            self.node_trained_models[node_name] = {}
            self.col_maps_node[node_name] = {}

            node_features_df = cudf.DataFrame.from_pandas(node_features_data[MetaData.FEATURES_DATA])

            for col_name, col_type in self.features_to_correlate_node[node_name].items():
                if col_name in self.xgboost_params:
                    xgboost_params = dict(self.xgboost_params[col_name])
                else:
                    xgboost_params = dict(self.xgboost_params)

                y_train = node_features_df[col_name].loc[nodes]

                if "objective" not in xgboost_params:
                    if col_type == ColumnType.CONTINUOUS:
                        xgboost_params["objective"] = "reg:squarederror"
                    elif col_type == ColumnType.CATEGORICAL:
                        xgboost_params["objective"] = "multi:softmax"
                        vals = node_features_df[col_name].loc[nodes]
                        encoder = LabelEncoder()
                        encoder.fit(vals)
                        self.col_maps_node[node_name][col_name] = encoder
                        num_classes = len(encoder.classes_)
                        xgboost_params["num_class"] = num_classes
                        y_train = encoder.transform(y_train)
                y_train = y_train.values

                dtrain = xgboost.DMatrix(X_train, y_train)
                trained_model = xgboost.train(
                    xgboost_params,
                    dtrain,
                    num_boost_round=self.num_boost_round,
                    evals=[(dtrain, "train")],
                    verbose_eval=self.verbose,
                )
                self.node_trained_models[node_name][col_name] = trained_model

    def align(
        self,
        graphs,
        node_features,
        edge_features,
    ) -> pd.DataFrame:

        structural_features = self._extract_structural_features(graphs)
        for k, v in structural_features.items():
            structural_features[k] = df_to_pandas(v)

        res = {
            MetaData.NODES: {},
            MetaData.EDGES: {},
        }
        if self.features_to_correlate_node:
            res[MetaData.NODES] = self._align(
                structural_features,
                node_features,
                None,
                self.features_to_correlate_node,
                self.col_maps_node,
                self.node_trained_models,
                MetaData.NODES,
            )

        if self.features_to_correlate_edge:
            res[MetaData.EDGES] = self._align(
                structural_features,
                edge_features,
                graphs,
                self.features_to_correlate_edge,
                self.col_maps_edge,
                self.edge_trained_models,
                MetaData.EDGES,
            )

        return res

    def _align(
        self,
        structural_features,
        tab_features,
        graphs,
        features_to_correlate_part,
        col_maps,
        trained_models: Dict[str, xgboost.Booster],
        part: Literal[MetaData.NODES, MetaData.EDGES],
    ) -> Dict[str, pd.DataFrame]:
        result_dict = {}
        for part_name, features_to_correlate in features_to_correlate_part.items():
            preproc_dict = get_preproc_dict(features_to_correlate)

            if part == MetaData.NODES:
                split_df = structural_features[part_name]
            elif part == MetaData.EDGES:
                split_df = graphs[part_name][MetaData.STRUCTURE_DATA]
            else:
                raise ValueError(f"Only `{MetaData.NODES}` and `{MetaData.EDGES}` parts expected, got ({part})")

            topk = min(len(split_df), self.topk)

            batch_size = self.batch_size
            if len(split_df) // batch_size == 0:
                batch_size = len(split_df)
            chunks = np.array_split(split_df, len(split_df) // batch_size)

            all_preds = []

            for chunk in chunks:
                if part == MetaData.NODES:
                    node_feat = chunk.values
                    X_test = node_feat.astype(float)
                    dtest = xgboost.DMatrix(X_test)
                elif part == MetaData.EDGES:
                    src_ids = chunk[:, 0]
                    dst_ids = chunk[:, 1]
                    src_struct_feat = structural_features[graphs[part_name][MetaData.SRC_NODE_TYPE]].loc[src_ids].values
                    dst_struct_feat = structural_features[graphs[part_name][MetaData.DST_NODE_TYPE]].loc[dst_ids].values
                    X_test = np.concatenate([src_struct_feat, dst_struct_feat], axis=1).astype(float)
                    dtest = xgboost.DMatrix(X_test)

                col_preds = []
                for col_name, col_type in features_to_correlate.items():
                    preds = trained_models[part_name][col_name].predict(dtest)
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
                y = tab_features[part_name][col_name]
                if preproc_fn is not None:
                    y, _ = preproc_fn(y)
                if col_type == ColumnType.CATEGORICAL:
                    y = col_maps[part_name][col_name].inverse_transform(y)
                y_generated.append(cudf.Series(y))

            y_generated = cudf.concat(y_generated, axis=1).values
            ranks = cupy.zeros((len(split_df), 1))
            if len(target_cols) == 1:
                y_generated = y_generated.reshape(-1)
                target_col = target_cols[0]
                col_type = features_to_correlate[target_col]

                if col_type == ColumnType.CATEGORICAL:
                    all_preds = col_maps[part_name][target_col].inverse_transform(
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
                            unassigned_idxs = ra_idxs[len(sel_idxs):]
                            unassigned_idxs_pred.append(unassigned_idxs)
                            pred_assigned[ra_idxs] = True
                            gen_assigned[sel_idxs] = True
                    ranks[~pred_assigned] = idxs[~gen_assigned][: cupy.sum(~pred_assigned)].reshape(-1, 1)

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
                for idx, chunk in enumerate(chunks):
                    idxs = cupy.ones((len(y_generated),), dtype=bool)
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
                    idx * batch_size: idx * batch_size + len(chunk)
                    ] = chunk_ranks.reshape(-1, 1)

            ranks[ranks >= len(tab_features[part_name])] = len(tab_features[part_name]) - 1
            ranks = cupy.asnumpy(ranks)
            ranks = ranks.squeeze()

            features = tab_features[part_name].iloc[ranks].reset_index(drop=True)
            result_dict[part_name] = features

        return result_dict

    def save(self, save_dir: Union[PosixPath, str]):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.edge_trained_models:
            for edge_name, models in self.edge_trained_models.items():
                for col_name, model in models.items():
                    model.save_model(
                        os.path.join(save_dir, f"{edge_name}___{col_name}___xgb_aligner_edge.json")
                    )

        if self.node_trained_models:
            for node_name, models in self.node_trained_models.items():
                for col_name, model in models.items():
                    model.save_model(
                        os.path.join(save_dir, f"{node_name}___{col_name}___xgb_aligner_node.json")
                    )

        meta_data = {
            "xgboost_params": self.xgboost_params,
            "num_boost_round": self.num_boost_round,
            "batch_size": self.batch_size,
            "topk": self.topk,
            "get_features": self.get_features,
            "verbose": self.verbose,
            "fitted_data": {
                "features_to_correlate_edge": self.features_to_correlate_edge,
                "features_to_correlate_node": self.features_to_correlate_node,
                "col_maps_edge": self.col_maps_edge,
                "col_maps_node": self.col_maps_node,
                "meta_dict_edge": self.meta_dict_edge,
                "meta_dict_node": self.meta_dict_node,
            }
        }
        with open(os.path.join(save_dir, "xgb_aligner_meta.pkl"), "wb") as file_handler:
            pickle.dump(meta_data, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, dir_path: Union[PosixPath, str]):

        with open(os.path.join(dir_path, "xgb_aligner_meta.pkl"), "rb") as file_handler:
            meta_data = pickle.load(file_handler)

        fitted_data = meta_data.pop('fitted_data')

        instance = cls(**meta_data)
        for k, v in fitted_data.items():
            setattr(instance, k, v)

        files = os.listdir(dir_path)
        edge_files = [f for f in files if "xgb_aligner_edge" in f]

        instance.edge_trained_models = defaultdict(dict)
        for ef in edge_files:
            xgb_model = xgboost.Booster()
            xgb_model.load_model(os.path.join(dir_path, ef))
            edge_name, col_name = ef.split("___")[:2]  # - same format as `save`
            instance.edge_trained_models[edge_name][col_name] = xgb_model

        node_files = [f for f in files if "xgb_aligner_node" in f]
        instance.node_trained_models = defaultdict(dict)
        for nf in node_files:
            xgb_model = xgboost.Booster()
            xgb_model.load_model(os.path.join(dir_path, nf))
            node_name, col_name = ef.split("___")[:2]  # - same format as `save`
            instance.node_trained_models[node_name][col_name] = xgb_model
        return instance
