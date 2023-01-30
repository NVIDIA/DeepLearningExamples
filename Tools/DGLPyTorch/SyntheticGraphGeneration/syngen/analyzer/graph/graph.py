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


def safeSNAP(f):
    def wrapper(*args, **kwargs):
        graph = args[0]
        graph.maybe_load_snap()
        return f(*args, **kwargs)

    return wrapper


class Graph:
    def __init__(self, path, name=None, load_eagerly=False, is_directed=False):
        self.path = path
        self.name = name
        self.is_directed = is_directed
        self.snapGraph = None

        if load_eagerly:
            self.maybe_load_snap()

    def maybe_load_snap(self):
        if not self.snapGraph:
            graph_type = snap.TNGraph if self.is_directed else snap.TUNGraph
            self.snapGraph = snap.LoadConnList(graph_type, self.path)

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
