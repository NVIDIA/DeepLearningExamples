import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from performer_pytorch import SelfAttention

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# for gradient checkpointing
def create_custom_forward(module, **kwargs):
    def custom_forward(*inputs):
        return module(*inputs, **kwargs)
    return custom_forward

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = self.a_2*(x-mean)
        x /= std
        x += self.b_2
        return x

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff, p_drop=0.1):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p_drop, inplace=True)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, src):
        src = self.linear2(self.dropout(F.relu_(self.linear1(src))))
        return src

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, heads, k_dim=None, v_dim=None, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        if k_dim == None:
            k_dim = d_model
        if v_dim == None:
            v_dim = d_model

        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scaling = 1/math.sqrt(self.d_k)

        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(k_dim, d_model)
        self.to_value = nn.Linear(v_dim, d_model)
        self.to_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, key, value, return_att=False):
        batch, L1 = query.shape[:2]
        batch, L2 = key.shape[:2]
        q = self.to_query(query).view(batch, L1, self.heads, self.d_k).permute(0,2,1,3) # (B, h, L, d_k)
        k = self.to_key(key).view(batch, L2, self.heads, self.d_k).permute(0,2,1,3) # (B, h, L, d_k)
        v = self.to_value(value).view(batch, L2, self.heads, self.d_k).permute(0,2,1,3)
        #
        attention = torch.matmul(q, k.transpose(-2, -1))*self.scaling
        attention = F.softmax(attention, dim=-1) # (B, h, L1, L2)
        attention = self.dropout(attention)
        #
        out = torch.matmul(attention, v) # (B, h, L, d_k)
        out = out.permute(0,2,1,3).contiguous().view(batch, L1, -1)
        #
        out = self.to_out(out)
        if return_att:
            attention = 0.5*(attention + attention.permute(0,1,3,2))
            return out, attention.permute(0,2,3,1)
        return out

# Own implementation for tied multihead attention
class TiedMultiheadAttention(nn.Module):
    def __init__(self, d_model, heads, k_dim=None, v_dim=None, dropout=0.1):
        super(TiedMultiheadAttention, self).__init__()
        if k_dim == None:
            k_dim = d_model
        if v_dim == None:
            v_dim = d_model

        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scaling = 1/math.sqrt(self.d_k)

        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(k_dim, d_model)
        self.to_value = nn.Linear(v_dim, d_model)
        self.to_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, key, value, return_att=False):
        B, N, L = query.shape[:3]
        q = self.to_query(query).view(B, N, L, self.heads, self.d_k).permute(0,1,3,2,4).contiguous() # (B, N, h, l, k)
        k = self.to_key(key).view(B, N, L, self.heads, self.d_k).permute(0,1,3,4,2).contiguous() # (B, N, h, k, l)
        v = self.to_value(value).view(B, N, L, self.heads, self.d_k).permute(0,1,3,2,4).contiguous() # (B, N, h, l, k)
        #
        #attention = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(N*self.d_k) # (B, N, h, L, L)
        #attention = attention.sum(dim=1) # tied attention (B, h, L, L)
        scale = self.scaling / math.sqrt(N)
        q = q * scale
        attention = torch.einsum('bnhik,bnhkj->bhij', q, k)
        attention = F.softmax(attention, dim=-1) # (B, h, L, L)
        attention = self.dropout(attention)
        attention = attention.unsqueeze(1) # (B, 1, h, L, L)
        #
        out = torch.matmul(attention, v) # (B, N, h, L, d_k)
        out = out.permute(0,1,3,2,4).contiguous().view(B, N, L, -1)
        #
        out = self.to_out(out)
        if return_att:
            attention = attention.squeeze(1)
            attention = 0.5*(attention + attention.permute(0,1,3,2))
            attention = attention.permute(0,3,1,2)
            return out, attention
        return out

class SequenceWeight(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(SequenceWeight, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, msa):
        B, N, L = msa.shape[:3]
        
        msa = msa.permute(0,2,1,3) # (B, L, N, K)
        tar_seq = msa[:,:,0].unsqueeze(2) # (B, L, 1, K)
        
        q = self.to_query(tar_seq).view(B, L, 1, self.heads, self.d_k).permute(0,1,3,2,4).contiguous() # (B, L, h, 1, k)
        k = self.to_key(msa).view(B, L, N, self.heads, self.d_k).permute(0,1,3,4,2).contiguous() # (B, L, h, k, N)
        
        q = q * self.scale
        attn = torch.matmul(q, k) # (B, L, h, 1, N)
        attn = F.softmax(attn, dim=-1)
        return self.dropout(attn)

# Own implementation for multihead attention (Input shape: Batch, Len, Emb)
class SoftTiedMultiheadAttention(nn.Module):
    def __init__(self, d_model, heads, k_dim=None, v_dim=None, dropout=0.1):
        super(SoftTiedMultiheadAttention, self).__init__()
        if k_dim == None:
            k_dim = d_model
        if v_dim == None:
            v_dim = d_model

        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.seq_weight = SequenceWeight(d_model, heads, dropout=dropout)
        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(k_dim, d_model)
        self.to_value = nn.Linear(v_dim, d_model)
        self.to_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, key, value, return_att=False):
        B, N, L = query.shape[:3]
        #
        seq_weight = self.seq_weight(query) # (B, L, h, 1, N)
        seq_weight = seq_weight.permute(0,4,2,1,3) # (B, N, h, l, -1)
        #
        q = self.to_query(query).view(B, N, L, self.heads, self.d_k).permute(0,1,3,2,4).contiguous() # (B, N, h, l, k)
        k = self.to_key(key).view(B, N, L, self.heads, self.d_k).permute(0,1,3,4,2).contiguous() # (B, N, h, k, l)
        v = self.to_value(value).view(B, N, L, self.heads, self.d_k).permute(0,1,3,2,4).contiguous() # (B, N, h, l, k)
        #
        #attention = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(N*self.d_k) # (B, N, h, L, L)
        #attention = attention.sum(dim=1) # tied attention (B, h, L, L)
        q = q * seq_weight # (B, N, h, l, k)
        k = k * self.scale
        attention = torch.einsum('bnhik,bnhkj->bhij', q, k)
        attention = F.softmax(attention, dim=-1) # (B, h, L, L)
        attention = self.dropout(attention)
        attention = attention # (B, 1, h, L, L)
        del q, k, seq_weight
        #
        #out = torch.matmul(attention, v) # (B, N, h, L, d_k)
        out = torch.einsum('bhij,bnhjk->bnhik', attention, v)
        out = out.permute(0,1,3,2,4).contiguous().view(B, N, L, -1)
        #
        out = self.to_out(out)
        
        if return_att:
            attention = attention.squeeze(1)
            attention = 0.5*(attention + attention.permute(0,1,3,2))
            attention = attention.permute(0,2,3,1)
            return out, attention
        return out

class DirectMultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, heads, dropout=0.1):
        super(DirectMultiheadAttention, self).__init__()
        self.heads = heads
        self.proj_pair = nn.Linear(d_in, heads)
        self.drop = nn.Dropout(dropout, inplace=True)
        # linear projection to get values from given msa
        self.proj_msa = nn.Linear(d_out, d_out)
        # projection after applying attention
        self.proj_out = nn.Linear(d_out, d_out)
    
    def forward(self, src, tgt):
        B, N, L = tgt.shape[:3]
        attn_map = F.softmax(self.proj_pair(src), dim=1).permute(0,3,1,2) # (B, h, L, L)
        attn_map = self.drop(attn_map).unsqueeze(1)
        
        # apply attention
        value = self.proj_msa(tgt).permute(0,3,1,2).contiguous().view(B, -1, self.heads, N, L) # (B,-1, h, N, L)
        tgt = torch.matmul(value, attn_map).view(B, -1, N, L).permute(0,2,3,1) # (B,N,L,K)
        tgt = self.proj_out(tgt)
        return tgt

class MaskedDirectMultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, heads, d_k=32, dropout=0.1):
        super(MaskedDirectMultiheadAttention, self).__init__()
        self.heads = heads
        self.scaling = 1/math.sqrt(d_k)
        
        self.to_query = nn.Linear(d_in, heads*d_k)
        self.to_key   = nn.Linear(d_in, heads*d_k)
        self.to_value = nn.Linear(d_out, d_out)
        self.to_out   = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, key, value, mask):
        batch, N, L = value.shape[:3] 
        #
        # project to query, key, value
        q = self.to_query(query).view(batch, L, self.heads, -1).permute(0,2,1,3) # (B, h, L, -1)
        k = self.to_key(key).view(batch, L, self.heads, -1).permute(0,2,1,3) # (B, h, L, -1)
        v = self.to_value(value).view(batch, N, L, self.heads, -1).permute(0,3,1,2,4) # (B, h, N, L, -1)
        #
        q = q*self.scaling
        attention = torch.matmul(q, k.transpose(-2, -1)) # (B, h, L, L)
        attention = attention.masked_fill(mask < 0.5, torch.finfo(q.dtype).min)
        attention = F.softmax(attention, dim=-1) # (B, h, L1, L2)
        attention = self.dropout(attention) # (B, h, 1, L, L)
        #
        #out = torch.matmul(attention, v) # (B, h, N, L, d_out//h)
        out = torch.einsum('bhij,bhnjk->bhnik', attention, v) # (B, h, N, L, d_out//h)
        out = out.permute(0,2,3,1,4).contiguous().view(batch, N, L, -1)
        #
        out = self.to_out(out)
        return out

# Use PreLayerNorm for more stable training
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, p_drop=0.1, performer_opts=None, use_tied=False):
        super(EncoderLayer, self).__init__()
        self.use_performer = performer_opts is not None
        self.use_tied = use_tied
        # multihead attention
        if self.use_performer:
            self.attn = SelfAttention(dim=d_model, heads=heads, dropout=p_drop, 
                                      generalized_attention=True, **performer_opts)
        elif use_tied:
            self.attn = SoftTiedMultiheadAttention(d_model, heads, dropout=p_drop)
        else:
            self.attn = MultiheadAttention(d_model, heads, dropout=p_drop)
        # feedforward
        self.ff = FeedForwardLayer(d_model, d_ff, p_drop=p_drop)

        # normalization module
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop, inplace=True)
        self.dropout2 = nn.Dropout(p_drop, inplace=True)

    def forward(self, src, return_att=False):
        # Input shape for multihead attention: (BATCH, SRCLEN, EMB)
        # multihead attention w/ pre-LayerNorm
        B, N, L = src.shape[:3]
        src2 = self.norm1(src)
        if not self.use_tied:
            src2 = src2.reshape(B*N, L, -1)
        if return_att:
            src2, att = self.attn(src2, src2, src2, return_att=return_att)
            src2 = src2.reshape(B,N,L,-1)
        else:
            src2 = self.attn(src2, src2, src2).reshape(B,N,L,-1)
        src = src + self.dropout1(src2)

        # feed-forward
        src2 = self.norm2(src) # pre-normalization
        src2 = self.ff(src2)
        src = src + self.dropout2(src2)
        if return_att:
            return src, att
        return src

# AxialTransformer with tied attention for L dimension
class AxialEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, p_drop=0.1, performer_opts=None,
                 use_tied_row=False, use_tied_col=False, use_soft_row=False):
        super(AxialEncoderLayer, self).__init__()
        self.use_performer = performer_opts is not None
        self.use_tied_row = use_tied_row
        self.use_tied_col = use_tied_col
        self.use_soft_row = use_soft_row
        # multihead attention
        if use_tied_row:
            self.attn_L = TiedMultiheadAttention(d_model, heads, dropout=p_drop)
        elif use_soft_row:
            self.attn_L = SoftTiedMultiheadAttention(d_model, heads, dropout=p_drop)
        else:
            if self.use_performer:
                self.attn_L = SelfAttention(dim=d_model, heads=heads, dropout=p_drop, 
                                            generalized_attention=True, **performer_opts)
            else:
                self.attn_L = MultiheadAttention(d_model, heads, dropout=p_drop)
        if use_tied_col:
            self.attn_N = TiedMultiheadAttention(d_model, heads, dropout=p_drop)
        else:
            if self.use_performer:
                self.attn_N = SelfAttention(dim=d_model, heads=heads, dropout=p_drop, 
                                            generalized_attention=True, **performer_opts)
            else:
                self.attn_N = MultiheadAttention(d_model, heads, dropout=p_drop)

        # feedforward
        self.ff = FeedForwardLayer(d_model, d_ff, p_drop=p_drop)

        # normalization module
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop, inplace=True)
        self.dropout2 = nn.Dropout(p_drop, inplace=True)
        self.dropout3 = nn.Dropout(p_drop, inplace=True)

    def forward(self, src, return_att=False):
        # Input shape for multihead attention: (BATCH, NSEQ, NRES, EMB)
        # Tied multihead attention w/ pre-LayerNorm
        B, N, L = src.shape[:3]
        src2 = self.norm1(src)
        if self.use_tied_row or self.use_soft_row:
            src2 = self.attn_L(src2, src2, src2) # Tied attention over L
        else:
            src2 = src2.reshape(B*N, L, -1)
            src2 = self.attn_L(src2, src2, src2)
            src2 = src2.reshape(B, N, L, -1)
        src = src + self.dropout1(src2)
        
        # attention over N
        src2 = self.norm2(src)
        if self.use_tied_col:
            src2 = src2.permute(0,2,1,3)
            src2 = self.attn_N(src2, src2, src2) # Tied attention over N
            src2 = src2.permute(0,2,1,3)
        else:
            src2 = src2.permute(0,2,1,3).reshape(B*L, N, -1)
            src2 = self.attn_N(src2, src2, src2) # attention over N
            src2 = src2.reshape(B, L, N, -1).permute(0,2,1,3)
        src = src + self.dropout2(src2)

        # feed-forward
        src2 = self.norm3(src) # pre-normalization
        src2 = self.ff(src2)
        src = src + self.dropout3(src2)
        return src

class Encoder(nn.Module):
    def __init__(self, enc_layer, n_layer):
        super(Encoder, self).__init__()
        self.layers = _get_clones(enc_layer, n_layer)
        self.n_layer = n_layer
   
    def forward(self, src, return_att=False):
        output = src
        for layer in self.layers:
            output = layer(output, return_att=return_att)
        return output

class CrossEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, d_k, d_v, performer_opts=None, p_drop=0.1):
        super(CrossEncoderLayer, self).__init__()
        self.use_performer = performer_opts is not None
        
        # multihead attention
        if self.use_performer:
            self.attn = SelfAttention(dim=d_model, k_dim=d_k, heads=heads, dropout=p_drop,
                                      generalized_attention=True, **performer_opts)
        else:
            self.attn = MultiheadAttention(d_model, heads, k_dim=d_k, v_dim=d_v, dropout=p_drop)
        # feedforward
        self.ff = FeedForwardLayer(d_model, d_ff, p_drop=p_drop)

        # normalization module
        self.norm = LayerNorm(d_k)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop, inplace=True)
        self.dropout2 = nn.Dropout(p_drop, inplace=True)

    def forward(self, src, tgt):
        # Input:
        #   For MSA to Pair: src (N, L, K), tgt (L, L, C)
        #   For Pair to MSA: src (L, L, C), tgt (N, L, K)
        # Input shape for multihead attention: (SRCLEN, BATCH, EMB)
        # multihead attention
        # pre-normalization
        src = self.norm(src)
        tgt2 = self.norm1(tgt)
        tgt2 = self.attn(tgt2, src, src) # projection to query, key, value are done in MultiheadAttention module
        tgt = tgt + self.dropout1(tgt2)
        
        # Feed forward
        tgt2 = self.norm2(tgt)
        tgt2 = self.ff(tgt2)
        tgt = tgt + self.dropout2(tgt2)
        
        return tgt

class DirectEncoderLayer(nn.Module):
    def __init__(self, heads, d_in, d_out, d_ff, symmetrize=True, p_drop=0.1):
        super(DirectEncoderLayer, self).__init__()
        self.symmetrize = symmetrize

        self.attn = DirectMultiheadAttention(d_in, d_out, heads, dropout=p_drop)
        self.ff = FeedForwardLayer(d_out, d_ff, p_drop=p_drop)

        # dropouts
        self.drop_1 = nn.Dropout(p_drop, inplace=True)
        self.drop_2 = nn.Dropout(p_drop, inplace=True)
        # LayerNorm
        self.norm = LayerNorm(d_in)
        self.norm1 = LayerNorm(d_out)
        self.norm2 = LayerNorm(d_out)

    def forward(self, src, tgt):
        # Input:
        #  For pair to msa: src=pair (B, L, L, C), tgt=msa (B, N, L, K)
        B, N, L = tgt.shape[:3]
        # get attention map
        if self.symmetrize:
            src = 0.5*(src + src.permute(0,2,1,3))
        src = self.norm(src)
        tgt2 = self.norm1(tgt)
        tgt2 = self.attn(src, tgt2)
        tgt = tgt + self.drop_1(tgt2)

        # feed-forward
        tgt2 = self.norm2(tgt.view(B*N,L,-1)).view(B,N,L,-1)
        tgt2 = self.ff(tgt2)
        tgt = tgt + self.drop_2(tgt2)

        return tgt

class CrossEncoder(nn.Module):
    def __init__(self, enc_layer, n_layer):
        super(CrossEncoder, self).__init__()
        self.layers = _get_clones(enc_layer, n_layer)
        self.n_layer = n_layer
    def forward(self, src, tgt):
        output = tgt
        for layer in self.layers:
            output = layer(src, output)
        return output


