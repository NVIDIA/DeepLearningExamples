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

import snap
from syngen.utils.types import MetaData


def safeSNAP(f):
    def wrapper(*args, **kwargs):
        graph = args[0]
        graph.maybe_load_snap()
        return f(*args, **kwargs)

    return wrapper


class Graph(object):
    def __init__(self, path=None, name=None, load_eagerly=False, is_directed=False, _snap_graph=None):
        self.path = path
        self.name = name
        self.is_directed = is_directed
        self.snapGraph = _snap_graph

        if load_eagerly:
            self.maybe_load_snap()

    def maybe_load_snap(self):
        if not self.snapGraph:
            graph_type = snap.TNGraph if self.is_directed else snap.TUNGraph
            self.snapGraph = snap.LoadConnList(graph_type, self.path)

    @staticmethod
    def instantiate_from_feature_spec(feature_spec, edge_name, graph_name=None):

        edge_info = feature_spec.get_edge_info(edge_name)

        is_bipartite = edge_info[MetaData.SRC_NODE_TYPE] != edge_info[MetaData.DST_NODE_TYPE]
        is_directed = edge_info[MetaData.DIRECTED]

        graph_type = snap.TNGraph if is_directed else snap.TUNGraph

        struct_data = feature_spec.get_structural_data(edge_name)

        if is_bipartite:
            num_src_nodes = feature_spec.get_node_info(edge_info[MetaData.SRC_NODE_TYPE])[MetaData.COUNT]
            num_dst_nodes = feature_spec.get_node_info(edge_info[MetaData.DST_NODE_TYPE])[MetaData.COUNT]

            num_nodes = num_src_nodes + num_dst_nodes
        else:
            num_nodes = feature_spec.get_node_info(edge_info[MetaData.SRC_NODE_TYPE])[MetaData.COUNT]

        snap_graph = graph_type.New(num_nodes, len(struct_data))

        for i in range(num_nodes):
            snap_graph.AddNode(i)

        for e in struct_data:
            snap_graph.AddEdge(int(e[0]), int(e[1]))

        return Graph(_snap_graph=snap_graph, is_directed=is_directed, name=graph_name)

    @safeSNAP
    def edge_count(self):
        return self.snapGraph.GetEdges()

    @safeSNAP
    def node_count(self):
        return self.snapGraph.GetNodes()

    @safeSNAP
    def get_edges(self):
        return [
            (EI.GetSrcNId(), EI.GetDstNId()) for EI in self.snapGraph.Edges()
        ]
