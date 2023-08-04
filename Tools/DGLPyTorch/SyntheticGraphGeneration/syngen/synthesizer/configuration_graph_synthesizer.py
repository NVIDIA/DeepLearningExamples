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
import json
import os
import shutil
import warnings
from typing import Optional, Literal

import pandas as pd

from syngen.configuration import SynGenDatasetFeatureSpec, SynGenConfiguration
from syngen.generator.tabular import tabular_generators_classes
from syngen.graph_aligner import aligner_classes
from syngen.generator.graph import get_structural_generator_class
from syngen.generator.tabular.utils import tabular_chunk_sample_generation
from syngen.utils.io_utils import (
    dump_generated_graph,
    load_graph,
    load_dataframe,
    merge_dataframe_files, dump_dataframe,
)
from syngen.utils.types import DataFrameType, MetaData, DataSourceInputType
from syngen.utils.utils import CustomTimer, dynamic_import, get_object_path, to_ndarray, df_to_pandas, ensure_path

logger = logging.getLogger(__name__)
log = logger

warnings.filterwarnings('ignore')


class ConfigurationGraphSynthesizer(object):
    """A configuration graph synthesizer. Supports generating graph datasets based on the provided configuration. This synthesizer requires a dataset to be fit on
    prior to generating graphs of similar properties.

    Args:
        configuration (SynGenConfiguration): configuration to be used during generation
        timer_path (srt): path to the file  where the  generation process timings will be saved
        num_workers (int): number of workers to speed up generation.
        save_path (str): path to the directory where the results will be saved
        gpu (bool): flag to use GPU graph generator (default: True ), if set to False CPU will be used.
        verbose (bool): print intermediate results (default: False)
    """
    def __init__(
        self,
        configuration: SynGenConfiguration,
        timer_path: Optional[str] = None,
        num_workers: int = 1,
        save_path: str = './',
        gpu: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        self.configuration = configuration
        self.num_workers = num_workers
        self.verbose = verbose
        self.timer = CustomTimer(timer_path, verbose=self.verbose)
        self.gpu = gpu
        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.structure_generators = None
        self.tabular_generators = None
        self.aligners = None

    def _fit_tabular_generators(self, tab_gen_configs, feature_info_list,
                                part: Literal[MetaData.NODES, MetaData.EDGES],
                                features_to_return=()):
        tabular_generators = []
        feature_info_dict = {feature[MetaData.NAME]: feature for feature in feature_info_list}
        feature_data_cache = {}
        for tab_gen_cfg in tab_gen_configs:
            gen_info = {'feature_file': tab_gen_cfg.get('feature_file')}
            tab_gen_class = tabular_generators_classes[tab_gen_cfg[MetaData.TYPE]]
            tab_gen_cfg[MetaData.PARAMS]['gpu'] = tab_gen_cfg[MetaData.PARAMS].get('gpu', self.gpu)
            tab_gen_cfg[MetaData.PARAMS]['verbose'] = tab_gen_cfg[MetaData.PARAMS].get('verbose', self.verbose)
            perform_fit = True
            enforce_fit = tab_gen_cfg.get('perform_fit', False)
            generator_dump_path = tab_gen_cfg.get(MetaData.DUMP_PATH, None)

            if generator_dump_path and os.path.exists(generator_dump_path) and not enforce_fit:
                tab_gen = tab_gen_class.load(generator_dump_path)
                perform_fit = False
            else:
                tab_gen = tab_gen_class(**tab_gen_cfg[MetaData.PARAMS])

            if tab_gen_cfg[MetaData.DATA_SOURCE][MetaData.TYPE] == DataSourceInputType.RANDOM:
                if perform_fit:
                    tab_gen.fit(columns=tab_gen_cfg[MetaData.FEATURES_LIST])
                if generator_dump_path and perform_fit:
                    tab_gen.save(generator_dump_path)
                tabular_generators.append((tab_gen, gen_info))
                continue

            categorical_features = []
            data_source_feature_info_list = None
            if not perform_fit:
                pass
            elif tab_gen_cfg[MetaData.DATA_SOURCE][MetaData.TYPE] == DataSourceInputType.DATASET:
                data_source_path = tab_gen_cfg[MetaData.DATA_SOURCE][MetaData.PATH]
            elif tab_gen_cfg[MetaData.DATA_SOURCE][MetaData.TYPE] == DataSourceInputType.CONFIGURATION:
                cfg = SynGenDatasetFeatureSpec.instantiate_from_preprocessed(
                    tab_gen_cfg[MetaData.DATA_SOURCE][MetaData.PATH])
                data_source_info = cfg.get_info(part, tab_gen_cfg[MetaData.DATA_SOURCE][MetaData.NAME])
                data_source_feature_info_list = data_source_info[MetaData.FEATURES]
                data_source_path = os.path.join(tab_gen_cfg[MetaData.DATA_SOURCE][MetaData.PATH],
                                                data_source_info[MetaData.FEATURES_PATH])
            else:
                raise ValueError("unsupported data_source type")

            for feature_name in tab_gen_cfg[MetaData.FEATURES_LIST]:
                if feature_info_dict[feature_name][MetaData.FEATURE_TYPE] == MetaData.CATEGORICAL:
                    categorical_features.append(feature_name)

            if not perform_fit and len(features_to_return) == 0:
                pass
            elif data_source_path in feature_data_cache:
                data = feature_data_cache[data_source_path]
            else:
                # FORCE_CPU_MEM_TRANSFER
                data = load_dataframe(data_source_path, feature_info=data_source_feature_info_list)
                feature_data_cache[data_source_path] = data

            if perform_fit:
                tab_gen.fit(data,
                            categorical_columns=categorical_features,
                            columns=tab_gen_cfg[MetaData.FEATURES_LIST],
                            verbose=self.verbose)

            if generator_dump_path and perform_fit:
                tab_gen.save(ensure_path(generator_dump_path))

            tabular_generators.append((tab_gen, gen_info))

        if features_to_return:
            return_dataframe = pd.DataFrame()
            for _, cache_data in feature_data_cache.items():
                columns_intersect = list(set(features_to_return) & set(cache_data.columns))
                return_dataframe[columns_intersect] = cache_data[columns_intersect]
            del feature_data_cache

            return_categorical_features = []
            for feature_name in features_to_return:
                if feature_info_dict[feature_name][MetaData.FEATURE_TYPE] == MetaData.CATEGORICAL:
                    return_categorical_features.append(feature_name)
            return tabular_generators, (return_dataframe, return_categorical_features)

        del feature_data_cache
        return tabular_generators

    def _fit_structural_generator(self, edge_type, return_graph=False):
        structure_gen_cfg = edge_type[MetaData.STRUCTURE_GENERATOR]

        is_bipartite = edge_type[MetaData.SRC_NODE_TYPE] != edge_type[MetaData.DST_NODE_TYPE]

        is_directed = edge_type[MetaData.DIRECTED]

        data_source_cfg = structure_gen_cfg[MetaData.DATA_SOURCE]
        is_random = data_source_cfg[MetaData.TYPE] == DataSourceInputType.RANDOM

        generator_class = get_structural_generator_class(
            structure_gen_cfg[MetaData.TYPE],
            is_bipartite=is_bipartite,
            is_random=is_random,
        )

        gen_info = dict(is_bipartite=is_bipartite,
                        is_directed=is_directed,
                        num_edges=edge_type[MetaData.COUNT],
                        noise=structure_gen_cfg[MetaData.PARAMS].get('noise', 0.5))

        structure_gen_cfg[MetaData.PARAMS]['gpu'] = structure_gen_cfg[MetaData.PARAMS].get('gpu', self.gpu)
        structure_gen_cfg[MetaData.PARAMS]['verbose'] = structure_gen_cfg[MetaData.PARAMS].get('verbose', self.verbose)
        perform_fit = True
        enforce_fit = structure_gen_cfg.get('perform_fit', False)

        generator_dump_path = structure_gen_cfg.get(MetaData.DUMP_PATH, None)

        if generator_dump_path and os.path.exists(generator_dump_path) and not enforce_fit:
            generator = generator_class.load(generator_dump_path)
            generator.gpu = structure_gen_cfg[MetaData.PARAMS]['gpu']
            generator.verbose = structure_gen_cfg[MetaData.PARAMS]['verbose']
            perform_fit = False
        else:
            generator = generator_class(
                **structure_gen_cfg[MetaData.PARAMS]
            )

        if not perform_fit and not return_graph:
            pass
        elif data_source_cfg[MetaData.TYPE] == DataSourceInputType.RANDOM:
            graph = None
        elif data_source_cfg[MetaData.TYPE] == DataSourceInputType.CONFIGURATION:
            cfg = SynGenDatasetFeatureSpec.instantiate_from_preprocessed(data_source_cfg[MetaData.PATH])
            data_source_edge_info = cfg.get_edge_info(data_source_cfg[MetaData.NAME])
            graph_src_set = cfg.get_node_info(data_source_edge_info[MetaData.SRC_NODE_TYPE])[MetaData.COUNT]
            graph_path = os.path.join(data_source_cfg[MetaData.PATH], data_source_edge_info[MetaData.STRUCTURE_PATH])
            graph = load_graph(graph_path)
        else:
            raise ValueError("unsupported data_source type")

        if is_bipartite:
            gen_info['is_directed'] = False
            gen_info['num_nodes_src_set'] = self.configuration.get_node_info(
                edge_type[MetaData.SRC_NODE_TYPE])[MetaData.COUNT]
            gen_info['num_nodes_dst_set'] = self.configuration.get_node_info(
                edge_type[MetaData.DST_NODE_TYPE])[MetaData.COUNT]

            if perform_fit:
                generator.fit(graph, src_set=None, dst_set=None,
                              is_directed=False, transform_graph=False)
        else:
            gen_info['num_nodes'] = self.configuration.get_node_info(edge_type[MetaData.SRC_NODE_TYPE])[MetaData.COUNT]
            gen_info['has_self_loop'] = structure_gen_cfg[MetaData.PARAMS].get('has_self_loop', False)
            if perform_fit:
                generator.fit(graph, is_directed=is_directed)

        if generator_dump_path and perform_fit:
            generator.save(generator_dump_path)

        if return_graph:
            return (generator, gen_info), graph, graph_src_set

        return generator, gen_info

    def _fit_aligners(self, aligner_cfgs, graphs_to_process, features_to_align):

        aligners = []
        for aligner_cfg in aligner_cfgs:
            aligner_class = aligner_classes[aligner_cfg[MetaData.TYPE]]

            aligner_graphs = {graph_name: graphs_to_process[graph_name] for graph_name in aligner_cfg[MetaData.GRAPHS]}
            aligner_node_features = {feature_name: features_to_align[MetaData.NODES][feature_name]
                                     for feature_name in aligner_cfg[MetaData.NODES]}
            aligner_edge_features = {feature_name: features_to_align[MetaData.EDGES][feature_name]
                                     for feature_name in aligner_cfg[MetaData.EDGES]}
            aligner = aligner_class(**aligner_cfg[MetaData.PARAMS])
            aligner.fit(aligner_graphs, aligner_node_features, aligner_edge_features)

            aligners.append((
                aligner,
                {
                    graph_name: {
                        MetaData.SRC_NODE_TYPE: graph_info[MetaData.SRC_NODE_TYPE],
                        MetaData.DST_NODE_TYPE: graph_info[MetaData.DST_NODE_TYPE]
                    }
                    for graph_name, graph_info in aligner_graphs.items()
                }
            ))

        del features_to_align
        del graphs_to_process
        return aligners

    def fit(
        self,
    ):
        """Fit the synthesizer on graph.
        """

        self.structure_generators = {}
        self.tabular_generators = {MetaData.NODES: {}, MetaData.EDGES: {}}
        self.aligners = []

        graphs_to_process = {}
        features_to_align = {MetaData.NODES: {}, MetaData.EDGES: {}}

        if MetaData.ALIGNERS in self.configuration:
            for aligner_cfg in self.configuration[MetaData.ALIGNERS]:
                for graph_name in aligner_cfg[MetaData.GRAPHS]:
                    graphs_to_process[graph_name] = None

                for part in [MetaData.NODES, MetaData.EDGES]:
                    if aligner_cfg[part]:
                        for part_name, feature_names in aligner_cfg[part].items():
                            if part_name not in features_to_align[part]:
                                features_to_align[part][part_name] = {
                                    MetaData.FEATURES_LIST: set(),
                                }
                            features_to_align[part][part_name][MetaData.FEATURES_LIST] |= set(feature_names)

        self.timer.start_counter('fit')
        self.timer.start_counter('fit_nodes')
        for node_type in self.configuration[MetaData.NODES]:
            node_name = node_type[MetaData.NAME]

            if MetaData.TABULAR_GENERATORS in node_type:
                self.timer.start_counter(f'fit_node_{node_name}')

                if node_name in features_to_align[MetaData.NODES]:
                    self.tabular_generators[MetaData.NODES][node_name], (features_data, cat_cols) = \
                        self._fit_tabular_generators(
                        node_type[MetaData.TABULAR_GENERATORS], node_type[MetaData.FEATURES], MetaData.NODES,
                        features_to_return=list(features_to_align[MetaData.NODES][node_name][MetaData.FEATURES_LIST])
                    )
                    features_to_align[MetaData.NODES][node_name][MetaData.FEATURES_DATA] = features_data
                    features_to_align[MetaData.NODES][node_name][MetaData.CATEGORICAL_COLUMNS] = cat_cols
                else:
                    self.tabular_generators[MetaData.NODES][node_name] = self._fit_tabular_generators(
                        node_type[MetaData.TABULAR_GENERATORS], node_type[MetaData.FEATURES], MetaData.NODES
                    )
                self.timer.end_counter(f'fit_node_{node_name}',
                                       f'NODE {node_name} FIT TOOK')
        self.timer.end_counter('fit_nodes', 'FIT NODES TOOK')

        self.timer.start_counter('fit_edges')
        for edge_type in self.configuration[MetaData.EDGES]:
            edge_name = edge_type[MetaData.NAME]

            if MetaData.STRUCTURE_GENERATOR in edge_type:
                self.timer.start_counter(f'fit_edges_struct_{edge_name}')
                if edge_name in graphs_to_process:
                    graphs_to_process[edge_name] = {
                        MetaData.SRC_NODE_TYPE: edge_type[MetaData.SRC_NODE_TYPE],
                        MetaData.DST_NODE_TYPE: edge_type[MetaData.DST_NODE_TYPE],
                    }
                    self.structure_generators[edge_name], \
                    graphs_to_process[edge_name][MetaData.STRUCTURE_DATA], \
                    graphs_to_process[edge_name]['src_size'] = self._fit_structural_generator(edge_type, return_graph=True)
                else:
                    self.structure_generators[edge_name] = self._fit_structural_generator(edge_type)

                self.timer.end_counter(f'fit_edges_struct_{edge_name}',
                                       f'EDGE {edge_name} STRUCTURAL FIT TOOK')

            if MetaData.TABULAR_GENERATORS in edge_type:
                self.timer.start_counter(f'fit_edges_tabular_{edge_name}')
                if edge_name in features_to_align[MetaData.EDGES]:
                    self.tabular_generators[MetaData.EDGES][edge_name], (features_data, cat_cols) = \
                        self._fit_tabular_generators(
                        edge_type[MetaData.TABULAR_GENERATORS], edge_type[MetaData.FEATURES], MetaData.EDGES,
                        features_to_return=list(features_to_align[MetaData.EDGES][edge_name][MetaData.FEATURES_LIST])
                    )
                    features_to_align[MetaData.EDGES][edge_name][MetaData.FEATURES_DATA] = features_data
                    features_to_align[MetaData.EDGES][edge_name][MetaData.CATEGORICAL_COLUMNS] = cat_cols
                else:
                    self.tabular_generators[MetaData.EDGES][edge_name] = self._fit_tabular_generators(
                        edge_type[MetaData.TABULAR_GENERATORS], edge_type[MetaData.FEATURES], MetaData.EDGES
                    )
                self.timer.end_counter(f'fit_edges_tabular_{edge_name}',
                                       f'EDGE {edge_name} TABULAR FIT TOOK')

        if MetaData.ALIGNERS in self.configuration:
            self.aligners = self._fit_aligners(self.configuration[MetaData.ALIGNERS],
                                               graphs_to_process,
                                               features_to_align)

        self.timer.end_counter('fit_edges', 'FIT EDGES TOOK')
        self.timer.end_counter('fit', 'FIT TOOK')

    def _generate_tabular_data(self, tabular_generators, num_samples, features_path, name):

        merge_data = features_path.endswith('.csv') or features_path.endswith('.parquet')

        if self.aligners:
            assert merge_data

        generated_dfs = []

        for tab_gen_id, (tab_gen, gen_info) in enumerate(tabular_generators):

            use_memmap = False
            if merge_data:
                save_path = os.path.join(self.save_path, 'temp_tab_gen_dir')
                fname = f"{name}_{tab_gen_id}" if len(tabular_generators) > 1 else name
            else:
                save_path = os.path.join(self.save_path, features_path)
                fname = 'chunk'

            os.makedirs(save_path, exist_ok=True)

            if gen_info['feature_file'] and gen_info['feature_file'].endswith('.npy') and tab_gen.supports_memmap:
                use_memmap = True
                fname = gen_info['feature_file']

            feature_files = tabular_chunk_sample_generation(
                tab_gen,
                n_samples=num_samples,
                save_path=save_path,
                fname=fname,
                num_workers=self.num_workers,
                use_memmap=use_memmap,
                verbose=self.verbose
            )

            if merge_data:
                generated_df = merge_dataframe_files(feature_files, format='parquet')
                generated_dfs.append(generated_df)
                shutil.rmtree(save_path)

        if merge_data:
            generated_dfs = pd.concat(generated_dfs, axis=1)
            dump_dataframe(generated_dfs, os.path.join(self.save_path, features_path), format=None)
        gc.collect()

    def generate(
        self,
        return_data=False,
        **kwargs,
    ):
        """ Generates graph

            Args:
                return_data(bool): if true load the generated data into the output configuration

        """
        node_type_to_node_counts = {node_type[MetaData.NAME]: node_type[MetaData.COUNT]
                                    for node_type in self.configuration[MetaData.NODES]}
        edge_type_to_edge_info = {edge_type[MetaData.NAME]: edge_type
                                  for edge_type in self.configuration[MetaData.EDGES]}

        output_config = self.configuration.copy()

        edge_type_name_to_idx = {edge_info[MetaData.NAME]: idx
                                 for idx, edge_info in enumerate(output_config[MetaData.EDGES])}
        node_type_name_to_idx = {node_info[MetaData.NAME]: idx
                                 for idx, node_info in enumerate(output_config[MetaData.NODES])}

        self.timer.start_counter("gen_s")
        for edge_type_name, (structure_generator, gen_info) in self.structure_generators.items():
            self.timer.start_counter(f'gen_edges_struct_{edge_type_name}')
            edge_info = edge_type_to_edge_info[edge_type_name]

            generated_graph_path = ensure_path(os.path.join(self.save_path, edge_info[MetaData.STRUCTURE_PATH]))

            merge_data = generated_graph_path.endswith('.csv') or \
                         generated_graph_path.endswith('.parquet')

            use_memmap = generated_graph_path.endswith('.npy')

            if not merge_data and not use_memmap:
                os.makedirs(generated_graph_path, exist_ok=True)

            if gen_info['is_bipartite']:
                num_nodes_src_set = node_type_to_node_counts[edge_info[MetaData.SRC_NODE_TYPE]] \
                    if node_type_to_node_counts[edge_info[MetaData.SRC_NODE_TYPE]] > -1 \
                    else gen_info['num_nodes_src_set']
                num_nodes_dst_set = node_type_to_node_counts[edge_info[MetaData.DST_NODE_TYPE]] \
                    if node_type_to_node_counts[edge_info[MetaData.DST_NODE_TYPE]] > -1 \
                    else gen_info['num_nodes_dst_set']
                graph, src_nodes, dst_nodes = structure_generator.generate(
                    num_edges_dst_src=gen_info['num_edges'],
                    num_edges_src_dst=gen_info['num_edges'],
                    num_nodes_src_set=num_nodes_src_set,
                    num_nodes_dst_set=num_nodes_dst_set,
                    is_directed=gen_info['is_directed'],
                    noise=gen_info.get('noise', 0.5),
                    return_node_ids=True,
                    apply_edge_mirroring=False,
                    transform_graph=False,
                    save_path=None if merge_data else generated_graph_path,
                )

                node_type_to_node_counts[edge_info[MetaData.SRC_NODE_TYPE]] = max(
                    node_type_to_node_counts[edge_info[MetaData.SRC_NODE_TYPE]],
                    src_nodes.max() + 1
                )
                node_type_to_node_counts[edge_info[MetaData.DST_NODE_TYPE]] = max(
                    node_type_to_node_counts[edge_info[MetaData.DST_NODE_TYPE]],
                    dst_nodes.max() + 1
                )
            else:
                num_nodes = node_type_to_node_counts[edge_info[MetaData.SRC_NODE_TYPE]] \
                    if node_type_to_node_counts[edge_info[MetaData.SRC_NODE_TYPE]] > -1 \
                    else gen_info['num_nodes']

                graph, node_ids = structure_generator.generate(
                    num_nodes=num_nodes,
                    num_edges=gen_info['num_edges'],
                    is_directed=gen_info['is_directed'],
                    has_self_loop=gen_info.get('has_self_loop', False),
                    noise=gen_info.get('noise', 0.5),
                    return_node_ids=True,
                    save_path=None if merge_data else generated_graph_path
                )
                node_type_to_node_counts[edge_info[MetaData.SRC_NODE_TYPE]] = max(
                    node_type_to_node_counts[edge_info[MetaData.SRC_NODE_TYPE]],
                    node_ids.max() + 1
                )

            if merge_data or not self.gpu:
                dump_generated_graph(generated_graph_path, graph)
            output_config[MetaData.EDGES][edge_type_name_to_idx[edge_type_name]][MetaData.COUNT] = \
                len(graph) if merge_data or use_memmap else int(graph)

            del graph
            gc.collect()
            self.timer.end_counter(f'gen_edges_struct_{edge_type_name}',
                                   f'EDGE {edge_type_name} STRUCT GEN TOOK')
        self.timer.end_counter("gen_s", "GEN STRUCT TOOK")

        for node_type_name, counts in node_type_to_node_counts.items():
            output_config[MetaData.NODES][node_type_name_to_idx[node_type_name]][MetaData.COUNT] = int(counts)

        self.timer.start_counter("gen_t_nodes")
        for node_type_name, tabular_generators in self.tabular_generators[MetaData.NODES].items():
            num_nodes = node_type_to_node_counts[node_type_name]
            features_path = output_config[MetaData.NODES][node_type_name_to_idx[node_type_name]][MetaData.FEATURES_PATH]
            self._generate_tabular_data(tabular_generators, num_nodes, features_path, node_type_name)
        self.timer.end_counter("gen_t_nodes", "GEN TABULAR NODE FEATURES TOOK")

        self.timer.start_counter("gen_t_edges")
        for edge_type_name, tabular_generators in self.tabular_generators[MetaData.EDGES].items():
            num_edges = output_config[MetaData.EDGES][edge_type_name_to_idx[edge_type_name]][MetaData.COUNT]
            features_path = output_config[MetaData.EDGES][edge_type_name_to_idx[edge_type_name]][MetaData.FEATURES_PATH]
            self._generate_tabular_data(tabular_generators, num_edges, features_path, edge_type_name)
        self.timer.end_counter("gen_t_edges", "GEN TABULAR EDGE FEATURES TOOK")

        self.timer.start_counter("gen_alignment")

        if self.aligners:
            for aligner, graphs_info in self.aligners:

                graphs_data = {}
                for graph_name, graph_info in graphs_info.items():
                    graphs_data[graph_name] = graph_info.copy()
                    if graph_info[MetaData.SRC_NODE_TYPE] != graph_info[MetaData.DST_NODE_TYPE]:
                        graphs_data[graph_name]['src_size'] = \
                        output_config[MetaData.NODES][node_type_name_to_idx[graph_info[MetaData.SRC_NODE_TYPE]]][
                            MetaData.COUNT]
                    graphs_data[graph_name][MetaData.STRUCTURE_DATA] = load_graph(os.path.join(
                        self.save_path,
                        output_config[MetaData.EDGES][edge_type_name_to_idx[graph_name]][MetaData.STRUCTURE_PATH]
                    ))

                node_features_data = {
                    node_name: load_dataframe(os.path.join(
                        self.save_path,
                        output_config[MetaData.NODES][node_type_name_to_idx[node_name]][MetaData.FEATURES_PATH]),
                        feature_info=output_config[MetaData.NODES][node_type_name_to_idx[node_name]][MetaData.FEATURES]
                    )
                    for node_name in aligner.features_to_correlate_node
                }

                edge_features_data = {
                    edge_name: load_dataframe(os.path.join(
                        self.save_path,
                        output_config[MetaData.EDGES][edge_type_name_to_idx[edge_name]][MetaData.FEATURES_PATH]),
                        feature_info=output_config[MetaData.EDGES][edge_type_name_to_idx[edge_name]][MetaData.FEATURES]
                    )
                    for edge_name in aligner.features_to_correlate_edge
                }

                aligned_data = aligner.align(
                    graphs_data,
                    node_features_data,
                    edge_features_data,
                )

                for node_name, tab_data in aligned_data[MetaData.NODES].items():
                    dump_dataframe(tab_data, os.path.join(
                            self.save_path,
                            output_config[MetaData.NODES][node_type_name_to_idx[node_name]][MetaData.FEATURES_PATH]
                        ), format=None
                    )
                for edge_name, tab_data in aligned_data[MetaData.EDGES].items():
                    dump_dataframe(tab_data, os.path.join(
                            self.save_path,
                            output_config[MetaData.EDGES][edge_type_name_to_idx[edge_name]][MetaData.FEATURES_PATH]
                        ), format=None
                    )
        self.timer.end_counter("gen_alignment", "GEN ALIGNMENT TAKE")

        with open(os.path.join(self.save_path, 'graph_metadata.json'), 'w') as f:
            json.dump(output_config, f, indent=4)

        output_config[MetaData.PATH] = self.save_path

        if return_data:
            for node_info in output_config[MetaData.NODES]:
                if node_info[MetaData.FEATURES_PATH]:
                    node_info[MetaData.FEATURES_DATA] = load_dataframe(os.path.join(
                        self.save_path, node_info[MetaData.FEATURES_PATH]
                    ))

            for edge_info in output_config[MetaData.EDGES]:
                if edge_info[MetaData.FEATURES_PATH]:
                    edge_info[MetaData.FEATURES_DATA] = load_dataframe(os.path.join(
                        self.save_path, edge_info[MetaData.FEATURES_PATH]
                    ))
                if edge_info[MetaData.STRUCTURE_PATH]:
                    edge_info[MetaData.STRUCTURE_DATA] = load_graph(os.path.join(
                        self.save_path, edge_info[MetaData.STRUCTURE_PATH],
                    ))
            return output_config
        return output_config

    def save(self, path):
        """ saves the synthesizer to disk

            Args:
                path (str): The path to save the synthesizer to
        """

        meta_data = {
            "configuration": self.configuration.copy(),
            "timer_path": self.timer.path,
            "num_workers": self.num_workers,
            "save_path": self.save_path,
            "gpu": self.gpu,
            "verbose": self.verbose,
        }
        if not os.path.exists(path):
            os.makedirs(path)

        if self.structure_generators:
            meta_data['struct_gens'] = {}
            for edge_name, (struct_gen, gen_info) in self.structure_generators.items():
                struct_gen.save(os.path.join(path, f'struct_gen_{edge_name}'))
                meta_data['struct_gens'][edge_name] = {
                    'gen_info': gen_info,
                    'object_path': get_object_path(struct_gen)
                }

        if self.tabular_generators:
            meta_data['tab_gens'] = {}
            for part, part_gens in self.tabular_generators.items():
                meta_data['tab_gens'][part] = {}
                for part_name, tab_gens in part_gens.items():
                    meta_data['tab_gens'][part][part_name] = []
                    for idx, (tab_gen, gen_info) in enumerate(tab_gens):
                        tab_gen.save(os.path.join(path, f'tab_gen_{part}_{part_name}_{idx}'))
                        meta_data['tab_gens'][part][part_name].append({
                            'gen_info': gen_info,
                            'object_path': get_object_path(tab_gen)
                        })

        if self.aligners:
            meta_data['aligners'] = []
            for idx, (aligner, graphs_info) in enumerate(self.aligners):
                aligner.save(os.path.join(path, f'aligner_{idx}'))
                meta_data['aligners'].append(
                    {
                        'object_path': get_object_path(aligner),
                        'graphs_info': graphs_info,
                    }
                )

        with open(os.path.join(path, "synthesizer_metadata.json"), "w") as fp:
            json.dump(meta_data, fp, indent=4)

    @classmethod
    def load(cls, path):
        """ load up a saved synthesizer object from disk.

            Args:
                path (str): The path to load the synthesizer from
        """

        with open(os.path.join(path, "synthesizer_metadata.json"), 'r') as f:
            meta_data = json.load(f)
        struct_gens = meta_data.pop('struct_gens', {})
        tab_gens = meta_data.pop('tab_gens', {})
        aligners = meta_data.pop('aligners', {})

        instance = cls(**meta_data)

        if struct_gens:
            instance.structure_generators = {
                edge_name: (
                    dynamic_import(data['object_path']).load(
                        os.path.join(path, f'struct_gen_{edge_name}')
                    ),
                    data['gen_info'],
                )
                for edge_name, data in struct_gens.items()
            }

        if tab_gens:
            instance.tabular_generators = {
                part: {
                    part_name: [
                        (
                            dynamic_import(data['object_path']).load(
                                os.path.join(path, f'tab_gen_{part}_{part_name}_{idx}')
                            ),
                            data['gen_info'],
                        )
                        for idx, data in enumerate(part_gens)
                    ]
                    for part_name, part_gens in part_data.items()
                }
                for part, part_data in tab_gens.items()
            }

        if aligners:
            instance.aligners = [
                (
                    dynamic_import(data['object_path']).load(
                        os.path.join(path, f'aligner_{idx}')
                    ),
                    data['graphs_info'],
                )
                for idx, data in enumerate(aligners)
            ]
        return instance
