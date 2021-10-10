import warnings

import dgl
import torch


def to_np(x):
    return x.cpu().detach().numpy()


class PickleGraph:
    """Lightweight graph object for easy pickling. Does not support batched graphs."""

    def __init__(self, G=None, desired_keys=None):
        self.ndata = dict()
        self.edata = dict()

        if G is None:
            self.src = []
            self.dst = []
        else:
            if G.batch_size > 1:
                warnings.warn("Copying a batched graph to a PickleGraph is not supported. "
                              "All node and edge data will be copied, but batching information will be lost.")

            self.src, self.dst = (to_np(idx) for idx in G.all_edges())

            for k in G.ndata:
                if desired_keys is None or k in desired_keys:
                    self.ndata[k] = to_np(G.ndata[k])

            for k in G.edata:
                if desired_keys is None or k in desired_keys:
                    self.edata[k] = to_np(G.edata[k])

    def all_edges(self):
        return self.src, self.dst


def copy_dgl_graph(G):
    if G.batch_size == 1:
        src, dst = G.all_edges()
        G2 = dgl.DGLGraph((src, dst))
        for edge_key in list(G.edata.keys()):
            G2.edata[edge_key] = torch.clone(G.edata[edge_key])
        for node_key in list(G.ndata.keys()):
            G2.ndata[node_key] = torch.clone(G.ndata[node_key])
        return G2
    else:
        list_of_graphs = dgl.unbatch(G)
        list_of_copies = []

        for batch_G in list_of_graphs:
            list_of_copies.append(copy_dgl_graph(batch_G))

        return dgl.batch(list_of_copies)


def update_relative_positions(G, *, relative_position_key='d', absolute_position_key='x'):
    """For each directed edge in the graph, calculate the relative position of the destination node with respect
    to the source node. Write the relative positions to the graph as edge data."""
    src, dst = G.all_edges()
    absolute_positions = G.ndata[absolute_position_key]
    G.edata[relative_position_key] = absolute_positions[dst] - absolute_positions[src]
