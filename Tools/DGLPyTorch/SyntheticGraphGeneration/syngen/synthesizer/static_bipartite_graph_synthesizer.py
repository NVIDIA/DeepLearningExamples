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
from typing import Optional, Union

import cudf
import dask_cudf
import pandas as pd
import dask.dataframe as dd

from syngen.utils import LocalCudaClusterManager
from syngen.utils import dynamic_import, get_object_path
from syngen.generator.graph.base_graph_generator import BaseBipartiteGraphGenerator
from syngen.generator.tabular.base_tabular_generator import BaseTabularGenerator
from syngen.graph_aligner.base_graph_aligner import BaseGraphAligner
from syngen.synthesizer.base_synthesizer import BaseSynthesizer
from syngen.utils.gen_utils import (
    chunk_sample_generation,
    dump_generated_graph_to_txt,
    merge_csv_files,
    read_edge_list
)
from syngen.utils.df_reader import DFReader
from syngen.utils.types import DataFrameType, MetaData
from syngen.utils.utils import CustomTimer, df_to_pandas

logger = logging.getLogger(__name__)
log = logger

warnings.filterwarnings('ignore')


class StaticBipartiteGraphSynthesizer(BaseSynthesizer):
    """A static bipartite graph synthesizer. Supports generating bipartite graphs with
    edge featurs, node features or both. This synthesizer requires a dataset to be fit on
    prior to generating graphs of similar properties.

    Args:
        graph_generator (BaseGraphGenerator): generator to fit on the structural
        component of the tabular dataset
        graph_info (dict): metadata associated with graph. See examples in
        syngen/preprocessing/datasets/.
        edge_feature_generator (BaseTabularGenerator): generator to fit on
        edge features (default: None).
        node_feature_generator (BaseTabularGenerator): generator to fit on
        node features (default: None).
        graph_aligner (BaseAligner): obj for aligning generated graph structure
        with graph tabular features (default: None). This must be provided if either
        `edge_feature_generator` or `node_feature_generator` is provided.
         save_path (str): path to the directory where the results will be saved
        features_save_name (str): save file name for the features (default: "features.csv").
        edge_list_save_name (str): save file name for the edge list (default: "edge_list.txt").
        graph_save_name (str): save file name for the final graph (default: "graph.csv").
        num_workers (int): number of workers to speed up generation.
        gpu (bool): flag to use GPU graph generator (default: True ), if set to False CPU will be used.
        use_dask (bool): flag to use dask, useful for large tables/graphs.
    """

    def __init__(
        self,
        graph_generator: BaseBipartiteGraphGenerator,
        graph_info: dict,
        *,
        graph_aligner: Optional[BaseGraphAligner] = None,
        edge_feature_generator: Optional[BaseTabularGenerator] = None,
        node_feature_generator: Optional[BaseTabularGenerator] = None,
        is_directed: bool = False,
        save_path: str = "./",
        feature_save_name: str = "feature.csv",
        edge_list_save_name: str = "edge_list.txt",
        graph_save_name: str = "graph.csv",
        timer_path: str = None,
        num_workers: int = 1,
        gpu: bool = True,
        use_dask: bool = False,
        **kwargs,
    ):
        self.edge_feature_generator = edge_feature_generator
        self.node_feature_generator = node_feature_generator
        self.graph_generator = graph_generator
        self.is_directed = is_directed
        self.graph_aligner = graph_aligner
        self.graph_info = graph_info
        self.save_path = save_path
        self.num_workers = num_workers

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.feature_save_name = feature_save_name
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
        edge_data: DataFrameType,
        node_data: Optional[DataFrameType] = None,
        *args,
        **kwargs,
    ):
        """Fit the synthesizer on given graph data. Each component is fitted on the corresponding
        portions of the data, which can then be used to generate graphs using the `generate` function.

        Args:
            edge_data (DataFrameType): DataFrameType containing edge data.
            node_data (DataFrameType): DataFrameType containing node data (default: None)
        """

        assert (
            node_data is not None and self.node_feature_generator is not None
        ) or (
            node_data is None and self.node_feature_generator is None
        ), "both `node_data` and `node_feature_generator` \
                must be either provided or none"
        self.src_name = self.graph_info[MetaData.EDGE_DATA][MetaData.SRC_NAME]
        self.dst_name = self.graph_info[MetaData.EDGE_DATA][MetaData.DST_NAME]
        self.node_id = None
        if node_data is not None:
            try:
                self.node_id = self.graph_info[MetaData.NODE_DATA].get(
                    MetaData.NODE_ID, "id"
                )
            except KeyError as e:
                raise e("`MetaData.NODE_DATA` is missing from `graph_info`")
        # - feature generator fitting
        log.info("Fitting feature generator...")
        self.timer.start_counter("fit_t")
        if self.edge_feature_generator is not None:
            categorical_columns = self.graph_info[MetaData.EDGE_DATA].get(
                MetaData.CATEGORICAL_COLUMNS, []
            )
            categorical_columns = list(
                set(categorical_columns) - {self.src_name, self.dst_name}
            )
            cols = list(
                set(edge_data.columns) - {self.src_name, self.dst_name}
            )
            edge_features = edge_data[cols]
            self.edge_feature_generator.fit(
                edge_features, categorical_columns=categorical_columns
            )
        if self.node_feature_generator is not None:
            categorical_columns = self.graph_info[MetaData.NODE_DATA].get(
                MetaData.CATEGORICAL_COLUMNS, []
            )
            categorical_columns = list(
                set(categorical_columns) - {self.node_id}
            )
            cols = list(
                set(node_data.columns) - {self.src_name, self.dst_name}
            )
            node_features = node_data[cols]
            self.node_feature_generator.fit(
                node_features, categorical_columns=categorical_columns
            )
        self.timer.end_counter("fit_t", "FIT FEATURE TOOK")

        # - graph generator fitting
        log.info("Fitting graph generator...")
        edge_list_df = df_to_pandas(edge_data[[self.src_name, self.dst_name]])
        src_ids = edge_list_df[self.src_name].values
        dst_ids = edge_list_df[self.dst_name].values
        src_dst = list(zip(src_ids, dst_ids))
        if not self.is_directed:
            dst_src = list(zip(dst_ids, src_ids))
        graph = list(set(src_dst + dst_src))
        src_set = set(src_ids)
        dst_set = set(dst_ids)

        self.timer.start_counter("fit_s")
        self.graph_generator.fit(
            graph=graph,
            src_set=src_set,
            dst_set=dst_set,
            is_directed=self.is_directed,
        )
        self.timer.end_counter("fit_s", "FIT STRUCT TOOK")

        # - aligner fitting
        if (
            self.edge_feature_generator is not None
            or self.node_feature_generator is not None
        ) and self.graph_aligner is not None:
            data = {}
            data[MetaData.EDGE_DATA] = edge_data
            if node_data is not None:
                data[MetaData.NODE_DATA] = node_data
            self.timer.start_counter("fit_a")
            self.graph_aligner.fit(
                data,
                src_col=self.src_name,
                dst_col=self.dst_name,
                node_id_col=self.node_id,
            )
            self.timer.end_counter("fit_a", "FIT ALIGN TOOK")

    def generate(
        self,
        num_nodes_src_set,
        num_nodes_dst_set,
        num_edges_src_dst,
        num_edges_dst_src,
        batch_size_graph: int = 1_000_000,
        batch_size_tabular: int = 1_000,
        graph_noise: float = 0.5,
        *args,
        **kwargs,
    ):
        """generate graph with approximately `num_nodes_src_set`/`num_nodes_dst_set` nodes
         and exactly `num_edges_src_dst`/`num_edges_dst_src` edges from generator and tabular data

        Args:
            num_nodes_src_set (int): approximate number of source nodes to be generated
            num_nodes_dst_set (int): approximate number of destination nodes to be generated
            num_edges_src_dst (int): exact number of source->destination edges to be generated
            num_edges_dst_src (int): exact number of destination->source to be generated
            batch_size_graph (int): size of the edge chunk that will be
            generated in one generation step (default: 1_000_000).
            batch_size_tabular (int): batch size for the tabular feature generator (default: 1_000).
            graph_noise (float): graph noise param for generation (default: 0.5).
        """

        # - generate static graph
        log.info("Generating graph...")
        self.timer.start_counter("gen_s")
        graph = self.graph_generator.generate(
            num_nodes_src_set=num_nodes_src_set,
            num_nodes_dst_set=num_nodes_dst_set,
            num_edges_src_dst=num_edges_src_dst,
            num_edges_dst_src=num_edges_dst_src,
            is_directed=self.is_directed,
            noise=graph_noise,
            batch_size=batch_size_graph,
        )
        self.timer.end_counter("gen_s", "GEN STRUCT TOOK")

        if not self.is_directed:
            # - rid of duplicate edges
            # can also split array in half,
            # below is safer
            graph = list(set(map(frozenset, graph)))

        # - dump edge list
        generated_graph_path = os.path.join(
            self.save_path, self.edge_list_save_name
        )

        dump_generated_graph_to_txt(generated_graph_path, graph)
        num_edges = len(graph)
        num_nodes = num_nodes_src_set + num_nodes_dst_set
        graph = None
        gc.collect()

        edge_feature_save_name = "edge_" + self.feature_save_name
        node_feature_save_name = "node_" + self.feature_save_name
        edge_generated_table_path = os.path.join(
            self.save_path, edge_feature_save_name
        )
        if os.path.exists(edge_generated_table_path):
            os.remove(edge_generated_table_path)
       
        node_generated_table_path = os.path.join(
            self.save_path, node_feature_save_name
        )
        if os.path.exists(node_generated_table_path):
            os.remove(node_generated_table_path)
        
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
                save_name=edge_feature_save_name,
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
                save_name=node_feature_save_name,
            )
        self.timer.end_counter("gen_t", "GEN TABULAR TOOK")

        # - align graph + features
        node_generated_df = None
        edge_generated_df = None
        reader = (
            DFReader.get_dask_reader() if self.use_dask else DFReader.df_reader
        )
        if os.path.exists(edge_generated_table_path):
            edge_generated_df = reader.read_csv(edge_generated_table_path)

        if os.path.exists(node_generated_table_path):
            node_generated_df = reader.read_csv(node_generated_table_path)
        
        src_col = self.graph_info[MetaData.EDGE_DATA][MetaData.SRC_NAME]
        dst_col = self.graph_info[MetaData.EDGE_DATA][MetaData.DST_NAME]
        generated_graph_el_df = read_edge_list(
            generated_graph_path,
            col_names=[src_col, dst_col],
            reader="dask_cudf" if self.use_dask else "cudf",
        )

        log.info("Generating final graph...")
        # - there's nothing to align simply return generated graph
        if (edge_generated_df is None and node_generated_df is None) or self.graph_aligner is None:
            aligned_df = {MetaData.EDGE_DATA: generated_graph_el_df}
        else:  # - add edges/nodes features using aligner
            data = {
                MetaData.EDGE_LIST: generated_graph_el_df,
                MetaData.NODE_DATA: node_generated_df,
                MetaData.EDGE_DATA: edge_generated_df,
            }
            self.timer.start_counter("gen_a")
            aligned_df = self.graph_aligner.align(
                data, src_col=src_col, dst_col=dst_col
            )
            self.timer.end_counter("gen_a", "GEN ALIGN TOOK")
        self.timer.maybe_close()
        # - dump generated data
        log.info(f"Saving data to {self.save_path}")
        for k, v in aligned_df.items():
            aligned_df_save_path = os.path.join(
                self.save_path, f"{k}_{self.graph_save_name}"
            )
            if isinstance(v, (cudf.DataFrame, pd.DataFrame)):
                if isinstance(v, pd.DataFrame):
                    try:
                        v = cudf.from_pandas(v)
                    except:
                        pass
                v.to_csv(aligned_df_save_path, index=False)
            if isinstance(v, (dask_cudf.DataFrame, dd.DataFrame)):
                v = v.compute()
                v.to_csv(aligned_df_save_path, index=False)

        return aligned_df

    def cleanup_session(self):
        """clean up session and free up resources
        """
        if self.use_dask:
            self.dask_cluster.destroy_local_cluster()

    @classmethod
    def load(cls, path):
        meta_data = json.load(
            open(os.path.join(path, "synthesizer_metadata.json"), "r")
        )
        graph_generator = dynamic_import(meta_data['graph_generator_path']).load(os.path.join(path, 'graph_generator'))
        meta_data.pop('graph_generator_path')

        graph_aligner = None
        if os.path.exists(os.path.join(path, 'graph_aligner')):
            graph_aligner = dynamic_import(meta_data['graph_aligner_path']).load(os.path.join(path, 'graph_aligner'))
            meta_data.pop('graph_aligner_path')

        edge_feature_generator = None
        if os.path.exists(os.path.join(path, 'edge_feature_generator')):
            edge_feature_generator = dynamic_import(meta_data['edge_feature_generator_path']). \
                load(os.path.join(path, 'edge_feature_generator'))
            meta_data.pop('edge_feature_generator_path')

        node_feature_generator = None
        if os.path.exists(os.path.join(path, 'node_feature_generator')):
            node_feature_generator = dynamic_import(meta_data['node_feature_generator_path']). \
                load(os.path.join(path, 'node_feature_generator'))
            meta_data.pop('node_feature_generator_path')

        return cls(
            graph_generator=graph_generator,
            graph_aligner=graph_aligner,
            edge_feature_generator=edge_feature_generator,
            node_feature_generator=node_feature_generator,
            **meta_data,
        )


    def save(self, path):
        meta_data = {
            "graph_info": self.graph_info,
            "is_directed": self.is_directed,
            "save_path": self.save_path,
            "feature_save_name": self.feature_save_name,
            "edge_list_save_name": self.edge_list_save_name,
            "graph_save_name": self.graph_save_name,
            "timer_path": self.timer.path,
            "num_workers": self.num_workers,
            "gpu": self.gpu,
            "use_dask": self.use_dask,
            "graph_generator_path": get_object_path(self.graph_generator),
        }

        if not os.path.exists(path):
            os.makedirs(path)

        self.graph_generator.save(os.path.join(path, 'graph_generator'))

        if self.graph_aligner:
            self.graph_aligner.save(os.path.join(path, 'graph_aligner'))
            meta_data["graph_aligner_path"] = get_object_path(self.graph_aligner)

        if self.edge_feature_generator is not None:
            self.edge_feature_generator.save(os.path.join(path, 'edge_feature_generator'))
            meta_data['edge_feature_generator_path'] = get_object_path(self.edge_feature_generator)

        if self.node_feature_generator is not None:
            self.node_feature_generator.save(os.path.join(path, 'node_feature_generator'))
            meta_data['node_feature_generator_path'] = get_object_path(self.node_feature_generator)

        with open(os.path.join(path, "synthesizer_metadata.json"), "w") as fp:
            json.dump(meta_data, fp)

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--num-nodes-src-set",
            type=int,
            help="Number of nodes to generate in the source set. Applies to `StaticBipartiteGraphSynthesizer`.",
        )
        parser.add_argument(
            "--num-nodes-dst-set",
            type=int,
            help="Number of nodes to generate in the destination set. Applies to `StaticBipartiteGraphSynthesizer`.",
        )
        parser.add_argument(
            "--num-edges-src-dst",
            type=int,
            help="Number of edge to generate from the source set to the destination set. Applies to `StaticBipartiteGraphSynthesizer`.",
        )
        parser.add_argument(
            "--num-edges-dst-src",
            type=int,
            help="Number of edges to generate from the destination set to the source set. Applies to `StaticBipartiteGraphSynthesizer`.",
        )
