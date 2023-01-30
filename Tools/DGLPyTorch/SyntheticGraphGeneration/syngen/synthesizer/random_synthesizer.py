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
import logging
import os
import json
import warnings
from typing import Optional, Union, Tuple

import cudf
import dask_cudf
import numpy as np
import pandas as pd
import dask.dataframe as dd


from syngen.utils import LocalCudaClusterManager
from syngen.generator.graph import RandomGraph, RandomBipartite
from syngen.generator.tabular.random import RandomMVGenerator
from syngen.synthesizer.base_synthesizer import BaseSynthesizer
from syngen.utils.gen_utils import (
    chunk_sample_generation,
    dump_generated_graph_to_txt,
    merge_csv_files,
    read_edge_list
)
from syngen.utils.df_reader import DFReader
from syngen.utils.types import MetaData
from syngen.utils.utils import CustomTimer, df_to_pandas, get_object_path

logger = logging.getLogger(__name__)
log = logger

warnings.filterwarnings('ignore')


class RandomSynthesizer(BaseSynthesizer):
    """A random graph synthesizer. It Supports generating graphs with edge features,
    node features, or both. This synthesizer does not require data for fitting, and
    generated static graphs with arbitrary number of nodes, edges, and feature dimensions.

    Args:
        bipartite (bool): flag to specify whether the generated graph should be bipartite.
        is_directed (bool): flag to specify if the edges of the generated graph should be directed
        save_path (str): path to the directory where the results will be saved
        features_save_name (str): save file name for the features (default: "features.csv").
        edge_list_save_name (str): save file name for the edge list (default: "edge_list.txt").
        graph_save_name (str): save file name for the final graph (default: "graph.csv").
        timer_path (str): path to save the timing information of the synthesizer (default: None).
        num_workers (int): number of workers to speed up generation using multiprocessing
        gpu (bool): flag to use GPU graph generator (default: True ), if set to False CPU will be used.
        use_dask (bool): flag to use dask, useful for large tables/graphs.
    """

    def __init__(
        self,
        *,
        bipartite: bool = False,
        is_directed: bool = True,
        save_path: str = "./",
        features_save_name: str = "feature.csv",
        edge_list_save_name: str = "edge_list.txt",
        graph_save_name: str = "graph.csv",
        timer_path: Optional[str] = None,
        num_workers: int = 1,
        gpu: bool = True,
        use_dask: bool = False,
        **kwargs,
    ):
        self.bipartite = bipartite
        self.is_directed = is_directed
        if self.bipartite:
            self.graph_generator = RandomBipartite()
        else:
            self.graph_generator = RandomGraph()
        self.edge_feature_generator = None
        self.node_feature_generator = None
        self.save_path = save_path
        self.num_workers = num_workers

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.features_save_name = features_save_name
        self.edge_list_save_name = edge_list_save_name
        self.graph_save_name = graph_save_name
        self.timer = CustomTimer(timer_path)
        self.gpu = gpu
        self.use_dask = use_dask

        if self.use_dask:
            self.dask_cluster = LocalCudaClusterManager()
            self.dask_cluster.initialize_local_cluster()

        if gpu:
            self.setup_gpu()
        else:
            self.setup_cpu()

    def setup_gpu(self):
        self.graph_generator.gpu = True

    def setup_cpu(self):
        self.graph_generator.gpu = False

    def fit(
        self,
        edge_dim: Optional[int] = None,
        node_dim: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """Fit the random synthesizer. A `RandomMVGenerator` is instantiated for either
        edge features as specified by `edge_dim` and node features specified by `node_dim`.
        The generated features follow a random multivariate gaussian distribution.

        Args:
            edge_dim (int): the dimension of edge features (default: None).
            node_dim (int): the dimension of node features (default: None)
        """
        self.graph_generator.fit(is_directed=self.is_directed)
        if edge_dim is not None:
            self.edge_feature_generator = RandomMVGenerator()
            self.edge_feature_generator.fit(ndims=edge_dim)
        if node_dim is not None:
            self.node_feature_generator = RandomMVGenerator()
            self.node_feature_generator.fit(ndims=node_dim)

    def generate(
        self,
        num_nodes: Optional[int] = None,
        num_edges: Optional[int] = None,
        num_nodes_src_set: Optional[int] = None,
        num_nodes_dst_set: Optional[int] = None,
        num_edges_src_dst: Optional[int] = None,
        num_edges_dst_src: Optional[int] = None,
        batch_size_graph: int = 1_000_000,
        batch_size_tabular: int = 1_000,
        graph_noise: float = 0.5,
        has_self_loop: bool = False,
        *args,
        **kwargs,
    ):
        """ Generate a graph with `num_nodes` and `num_edges`,

        Args:
            If `bipartite` is set to false:
                num_nodes (Optional[int]): approximate number of nodes to be generated must be provided
                num_edges (Optional[int]): approximate number of edges to be generated must be provided
            If `bipartite` is set to true:
                num_nodes_src_set (Optional[int]): approximate number of nodes in the source set must be provided
                num_nodes_dst_set (Optional[int]): approximate number of nodes in the destination set must be provided
                num_edges_src_dst (Optional[int]): approximate number of edges from source to destination must be provided
                num_edges_dst_src (Optional[int]): approximate number of edges from destination to source must be provided
            batch_size_graph (int): size of the edge chunk that will be
            generated in one generation step (default: 1_000_000).
            batch_size_tabular (int): batch size for the tabular feature generator (default: 1_000).
            graph_noise (float): graph noise param for generation (default: 0.5).
            has_self_loop (bool): set to true if graph should have self loops (default: False).
        """
        log.info("Generating graph...")
        if self.bipartite:
            self.timer.start_counter("gen_s")
            graph = self.graph_generator.generate(
                num_nodes_src_set=num_nodes_src_set,
                num_nodes_dst_set=num_nodes_dst_set,
                num_edges_src_dst=num_edges_src_dst,
                num_edges_dst_src=num_edges_dst_src,
                is_directed=self.is_directed,
                batch_size=batch_size_graph,
            )
            self.timer.end_counter("gen_s", "GEN STRUCT TOOK")
        else:
            # - generate static graph
            self.timer.start_counter("gen_s")
            graph = self.graph_generator.generate(
                num_nodes=num_nodes,
                num_edges=num_edges,
                is_directed=self.is_directed,
                has_self_loop=has_self_loop,
                noise=graph_noise,
                batch_size=batch_size_graph,
            )
            self.timer.end_counter("gen_s", "GEN STRUCT TOOK")

        if not self.is_directed:
            # - rid of duplicate edges
            graph = list(set(map(frozenset, graph)))
            graph = np.asarray(list(map(list, graph)))

        # - dump edge list
        generated_graph_path = os.path.join(
            self.save_path, self.edge_list_save_name
        )
        nodes = set(list(graph[:, 0]) + list(graph[:, 1]))
        nodes_map = dict(zip(nodes, list(range(len(nodes)))))
        graph = np.asarray(
            [np.asarray([nodes_map[g[0]], nodes_map[g[1]]]) for g in graph]
        )
        dump_generated_graph_to_txt(generated_graph_path, graph)
        num_nodes = len(nodes_map)
        num_edges = len(graph)
        graph = None
        nodes_map = None
        nodes = None
        gc.collect()

        edge_features_save_name = "edge_" + self.features_save_name
        node_features_save_name = "node_" + self.features_save_name
        # - generate features
        self.timer.start_counter("gen_t")
        if self.edge_feature_generator is not None:
            generated_files = chunk_sample_generation(
                self.edge_feature_generator,
                n_samples=num_edges,
                save_path=self.save_path,
                fname="table_edge_samples",
                num_workers=self.num_workers,
            )
            merge_csv_files(
                file_paths=generated_files,
                save_path=self.save_path,
                save_name=edge_features_save_name,
            )
        if self.node_feature_generator is not None:
            generated_files = chunk_sample_generation(
                self.node_feature_generator,
                n_samples=num_nodes,
                save_path=self.save_path,
                fname="table_node_samples",
                num_workers=self.num_workers,
            )
            merge_csv_files(
                file_paths=generated_files,
                save_path=self.save_path,
                save_name=node_features_save_name,
            )
        self.timer.end_counter("gen_t", "GEN FEATURE TOOK")

        # - align graph + features
        node_generated_df = None
        edge_generated_df = None
        reader = (
            DFReader.get_dask_reader() if self.use_dask else DFReader.get_df_reader()
        )
        if os.path.exists(os.path.join(self.save_path, edge_features_save_name)):
            generated_table_path = os.path.join(
                self.save_path, edge_features_save_name
            )
            edge_generated_df = reader.read_csv(generated_table_path)

        if os.path.exists(os.path.join(self.save_path, node_features_save_name)):
            generated_table_path = os.path.join(
                self.save_path, node_features_save_name
            )
            node_generated_df = reader.read_csv(generated_table_path)

        generated_graph_el_df = read_edge_list(
            generated_graph_path,
            reader="dask_cudf" if self.use_dask else "cudf",
        )
        data = {}
        if node_generated_df is not None:
            data[MetaData.NODE_DATA] = df_to_pandas(node_generated_df)

        if edge_generated_df is not None:
            el_df = df_to_pandas(generated_graph_el_df)
            ef_df = df_to_pandas(edge_generated_df)
            edge_data = pd.concat([el_df, ef_df], axis=1)
            data[MetaData.EDGE_DATA] = edge_data
        elif generated_graph_el_df is not None:
            data[MetaData.EDGE_DATA] = df_to_pandas(generated_graph_el_df)

        # - dump generated data
        for k, v in data.items():
            if isinstance(v, (cudf.DataFrame, pd.DataFrame)):
                data_df_save_path = os.path.join(
                    self.save_path, f"{k}_{self.graph_save_name}"
                )
                v.to_csv(data_df_save_path, index=False)
            if isinstance(v, (dask_cudf.DataFrame, dd.DataFrame)):
                data_df_save_path = os.path.join(
                    self.save_path, f"{k}_*_{self.graph_save_name}"
                )
                v = v.compute()
                v.to_csv(data_df_save_path, index=False)

        return data

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--edge-dim",
            type=int,
            default=None,
            help="Edge feature dimension to generate using RandomSynthesizer.",
        )
        parser.add_argument(
            "--node-dim",
            type=int,
            default=None,
            help="Node feature dimension to generate using RandomSynthesizer.",
        )
        parser.add_argument(
            "--g-bipartite",
            default=False,
            action='store_true',
            help="Generates bipartite graph, flag used in RandomSynthesizer.",
        )
        parser.add_argument(
            "--g-directed",
            default=False,
            action='store_true',
            help="Generates directed graph, flag used in RandomSynthesizer.",
        )
        return parser

    def cleanup_session(self):
        """clean up session and free up resources
        """
        if self.use_dask:
            self.dask_cluster.destroy_local_cluster()

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "synthesizer_metadata.json"), "r") as fp:
            meta_data = json.load(fp)
        instance = cls(**meta_data)
        instance.graph_generator.fit(is_directed=instance.is_directed)
        if os.path.exists(os.path.join(path, 'node_feature_generator')):
            instance.node_feature_generator = RandomMVGenerator.load(os.path.join(path, 'node_feature_generator'))
        if os.path.exists(os.path.join(path, 'edge_feature_generator')):
            instance.edge_feature_generator = RandomMVGenerator.load(os.path.join(path, 'edge_feature_generator'))
        return instance

    def save(self, path):
        meta_data = {
            "bipartite": self.bipartite,
            "is_directed": self.is_directed,
            "save_path": self.save_path,
            "features_save_name": self.features_save_name,
            "edge_list_save_name": self.edge_list_save_name,
            "graph_save_name": self.graph_save_name,
            "timer_path": self.timer.path,
            "num_workers": self.num_workers,
            "gpu": self.gpu,
            "use_dask": self.use_dask,
        }

        if not os.path.exists(path):
            os.makedirs(path)

        if self.edge_feature_generator is not None:
            self.edge_feature_generator.save(os.path.join(path, 'edge_feature_generator'))
        if self.node_feature_generator is not None:
            self.node_feature_generator.save(os.path.join(path, 'node_feature_generator'))
        with open(os.path.join(path, "synthesizer_metadata.json"), "w") as fp:
            json.dump(meta_data, fp)
