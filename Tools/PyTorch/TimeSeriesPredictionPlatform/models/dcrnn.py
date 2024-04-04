# Copyright (c) 2024 NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse import linalg
import dgl
import dgl.function as fn
import dgl.ops as ops

from .tft_pyt.modeling import LazyEmbedding

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = np.diag(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx)
    random_walk_mx = torch.from_numpy(random_walk_mx)
    return random_walk_mx


def calculate_dual_random_walk_matrix(adj_mx):
    L0 = calculate_random_walk_matrix(adj_mx).T
    L1 = calculate_random_walk_matrix(adj_mx.T).T
    return L0, L1


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    L = L.astype(np.float32).todense()
    return torch.from_numpy(L)


class DCGRUCell(torch.nn.Module):
    def __init__(self, num_units, max_diffusion_step, nonlinearity='tanh'):
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_units = num_units
        self.gconv1 = Gconv(self._num_units*2, self._num_units, max_diffusion_step, 0.0)
        self.gconv2 = Gconv(self._num_units, self._num_units, max_diffusion_step, 0.0)

    def forward(self, graph, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.
        """
        _inputs = torch.cat([inputs, hx], dim=-1)
        x = self.gconv1(graph, _inputs)

        value = torch.sigmoid(x)
        r, u = value.chunk(2, dim=-1)

        _inputs = torch.cat([inputs, r * hx], dim=-1)
        c = self.gconv2(graph, _inputs)

        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state


class Gconv(torch.nn.Module):
    def __init__(self, output_size, hidden_size, max_diffusion_step, bias_start=0.0):
        assert max_diffusion_step > 0
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self._max_diffusion_step = max_diffusion_step

        self.num_matrices = 2 * self._max_diffusion_step + 1
        self.lin = torch.nn.LazyLinear(self.output_size)
        def _reset_parameters(self):
            torch.nn.init.xavier_normal_(self.weight)
            torch.nn.init.constant_(self.bias, bias_start)
        bound_method = _reset_parameters.__get__(self.lin, self.lin.__class__)
        self.lin.reset_parameters = bound_method

    @staticmethod
    def calculate_random_walk_matrix(adj_mx):
        d = adj_mx.sum(1)
        d_inv = d.pow(-1)
        d_inv[torch.isinf(d_inv)] = 0.
        random_walk_mx = d_inv.unsqueeze(1).mul(adj_mx)
        return random_walk_mx


    def rwLaplacian(self,feat, graph):
        rev = graph.reverse()

        # L0
        out_degree = ops.copy_e_sum(rev, graph.edata['w']) #adj_mx.sum(1)
        graph.ndata['_h'] = feat[...,0] * out_degree.pow(-1).unsqueeze(-1)
        graph.update_all(fn.u_mul_e('_h', 'w', 'm') , fn.sum('m', '_h'))

        # L1
        in_degree = ops.copy_e_sum(graph, graph.edata['w']) #adj_mx.sum(0)
        rev.edata['w'] = graph.edata['w']
        rev.ndata['_h'] = feat[...,1] * in_degree.pow(-1).unsqueeze(-1)
        rev.update_all(fn.u_mul_e('_h', 'w', 'm') , fn.sum('m', '_h'))

        return torch.stack((graph.ndata.pop('_h'), rev.ndata.pop('_h')), dim=-1)

    def forward(self, graph, inputs):
        batch_size = graph.batch_size

        # Caching
        # We assume that all graphs are the same in sructure!
        if not hasattr(self, 'adj_mx'):
            with torch.no_grad():
                samples = dgl.unbatch(graph)
                adj_mx = torch.sparse_coo_tensor(indices=samples[0].adjacency_matrix().coalesce().indices().to(inputs.device),
                        values=samples[0].edata['w'].to(inputs.device)).to_dense()
                L0 = Gconv.calculate_random_walk_matrix(adj_mx).T
                L1 = Gconv.calculate_random_walk_matrix(adj_mx.T).T
                self.register_buffer('adj_mx', adj_mx, persistent=False)
                self.register_buffer('L0', L0, persistent=False)
                self.register_buffer('L1', L1, persistent=False)
        if hasattr(self, f'L_{batch_size}'):
            L = getattr(self, f'L_{batch_size}')
        else:
            L = torch.block_diag(*[l for l in (self.L0,self.L1) for _ in range(batch_size)]).to_sparse()
            setattr(self, f'L_{batch_size}', L)

        x0 = torch.cat((inputs,inputs), dim=0)
        x1 = torch.sparse.mm(L, x0)
        dif_outs = [inputs, *x1.chunk(2, dim=0)] 

        for k in range(2, self._max_diffusion_step + 1):
            x2 = 2 * torch.sparse.mm(L, x1) - x0
            dif_outs += x2.chunk(2, dim=0)
            x1, x0 = x2, x1

        x = torch.stack(dif_outs, dim=-1)
        x = x.reshape(graph.num_nodes(), -1)
        x = self.lin(x)
        return x



class RNNStack(nn.Module):
    def __init__(self, num_rnn_layers, max_diffusion_step, rnn_units, nonlinearity='tanh'):
        super().__init__()
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self.dcgru_layers = nn.ModuleList([DCGRUCell(rnn_units, max_diffusion_step, nonlinearity=nonlinearity) for _ in range(self.num_rnn_layers)])

    def forward(self, graph, inputs, hidden_state=None):
        if hidden_state is None:
            hidden_state = inputs.new_zeros((self.num_rnn_layers, graph.num_nodes(), self.rnn_units))
                                     
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(graph, output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow

class DCRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_diffusion_step = int(config.get('max_diffusion_step', 2))
        self.num_nodes = int(config.get('num_nodes', 1))
        self.num_rnn_layers = int(config.get('num_rnn_layers', 1))
        self.rnn_units = int(config.get('rnn_units'))
        self.activation = config.get('activation')
        self.output_dim = int(config.get('output_dim', 1))
        self.horizon = int(config.get('horizon', 1))  # for the decoder
        self.encoder_model = RNNStack(self.num_rnn_layers, self.max_diffusion_step, self.rnn_units, self.activation)
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.decoder_model = RNNStack(self.num_rnn_layers, self.max_diffusion_step, self.rnn_units, self.activation)
        self.cl_decay_steps = int(config.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(config.get('use_curriculum_learning', False))
        self.seq_len = int(config.get('encoder_length'))  # for the encoder
        self.batches_seen = 0

        self.use_embedding = config.use_embedding
        ### New embedding
        if self.use_embedding:
            self.config.hidden_size = self.config.input_dim
            self.embedding = LazyEmbedding(self.config)
        self.include_static_data = config.get('include_static_data', False)
        ####

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, graph):
        encoder_hidden_state = None
        h = graph.ndata['h']
        for t in range(self.seq_len):
            _, encoder_hidden_state = self.encoder_model(graph, h[:,t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, graph, encoder_hidden_state, labels=None):
        decoder_hidden_state = encoder_hidden_state
        decoder_input = encoder_hidden_state.new_zeros((graph.num_nodes(), 1))

        outputs = []

        for t in range(self.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(graph, decoder_input, decoder_hidden_state)
            decoder_output = self.projection_layer(decoder_output)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(self.batches_seen):
                    decoder_input = labels[:,t].view(-1,1)
        outputs = torch.stack(outputs, dim=1)
        return outputs

    def forward(self, batch):
        if self.use_embedding:
            # New embedding
            _batch = {
                    k:v[:, :self.seq_len] 
                    if v is not None and v.numel() else None 
                    for k,v in batch.ndata.items() 
                    if 'ID' not in k and 'id' not in k
                    }
            emb = self.embedding(_batch)
            emb = [e.view(*e.shape[:-2], -1) for e in emb if e is not None]
            emb[0] = emb[0].unsqueeze(1).expand(emb[0].shape[0], self.seq_len, *emb[0].shape[1:])
            if not self.include_static_data:
                emb = emb[1:]
            batch.ndata['h'] = torch.cat(emb, dim=-1)
            ####
        else:
            t = batch.ndata['k_cont'][:, :self.seq_len, 2:]
            t = torch.einsum('btk,k->bt', t, t.new([1, 0.16]))
            batch.ndata['h'] = torch.cat([batch.ndata['target'][:, :self.seq_len], t.unsqueeze(-1)], dim=-1)

        if self.training:
            labels = batch.ndata['target'][:, self.seq_len:].view(-1, self.num_nodes, self.horizon).transpose(1,2)
        else:
            labels = None

        encoder_hidden_state = self.encoder(batch)
        outputs = self.decoder(batch, encoder_hidden_state, labels)
        self.batches_seen += 1
        return outputs
