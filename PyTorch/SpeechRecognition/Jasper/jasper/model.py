# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import filter_warnings


activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
}


def init_weights(m, mode='xavier_uniform'):
    if type(m) == nn.Conv1d or type(m) == MaskedConv1d:
        if mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            raise ValueError("Unknown Initialization mode: {0}".format(mode))

    elif type(m) == nn.BatchNorm1d:
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (kernel_size // 2) * dilation


class MaskedConv1d(nn.Conv1d):
    """1D convolution with sequence masking
    """
    __constants__ = ["masked"]
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, masked=True):
        super(MaskedConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.masked = masked

    def get_seq_len(self, lens):
        # rounding_mode not available in 20.10 container
        # return torch.div((lens + 2 * self.padding[0] - self.dilation[0]
        #                   * (self.kernel_size[0] - 1) - 1), self.stride[0], rounding_mode="floor") + 1
        return torch.floor((lens + 2 * self.padding[0] - self.dilation[0]
                            * (self.kernel_size[0] - 1) - 1) / self.stride[0]).long() + 1

    def forward(self, x, x_lens=None):
        if self.masked:
            max_len = x.size(2)
            idxs = torch.arange(max_len, dtype=x_lens.dtype, device=x_lens.device)
            mask = idxs.expand(x_lens.size(0), max_len) >= x_lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
            x_lens = self.get_seq_len(x_lens)

        return super(MaskedConv1d, self).forward(x), x_lens


class JasperBlock(nn.Module):
    __constants__ = ["use_conv_masks"]

    """Jasper Block. See https://arxiv.org/pdf/1904.03288.pdf
    """
    def __init__(self, infilters, filters, repeat=3, kernel_size=11, stride=1,
                 dilation=1, padding='same', dropout=0.2, activation=None,
                 residual=True, residual_panes=[], use_conv_masks=False):
        super(JasperBlock, self).__init__()

        assert padding == "same", "Only 'same' padding is supported."

        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.use_conv_masks = use_conv_masks
        self.conv = nn.ModuleList()
        for i in range(repeat):
            self.conv.extend(self._conv_bn(infilters if i == 0 else filters,
                                           filters,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           dilation=dilation,
                                           padding=padding_val))
            if i < repeat - 1:
                self.conv.extend(self._act_dropout(dropout, activation))

        self.res = nn.ModuleList() if residual else None
        res_panes = residual_panes.copy()
        self.dense_residual = residual

        if residual:
            if len(residual_panes) == 0:
                res_panes = [infilters]
                self.dense_residual = False

            for ip in res_panes:
                self.res.append(nn.ModuleList(
                    self._conv_bn(ip, filters, kernel_size=1)))

        self.out = nn.Sequential(*self._act_dropout(dropout, activation))

    def _conv_bn(self, in_channels, out_channels, **kw):
        return [MaskedConv1d(in_channels, out_channels,
                             masked=self.use_conv_masks, **kw),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)]

    def _act_dropout(self, dropout=0.2, activation=None):
        return [activation or nn.Hardtanh(min_val=0.0, max_val=20.0),
                nn.Dropout(p=dropout)]

    def forward(self, xs, xs_lens=None):
        if not self.use_conv_masks:
            xs_lens = 0

        # forward convolutions
        out = xs[-1]
        lens = xs_lens
        for i, l in enumerate(self.conv):
            if isinstance(l, MaskedConv1d):
                out, lens = l(out, lens)
            else:
                out = l(out)

        # residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if j == 0:  #  and self.use_conv_mask:
                        res_out, _ = res_layer(res_out, xs_lens)
                    else:
                        res_out = res_layer(res_out)
                out += res_out

        # output
        out = self.out(out)
        if self.res is not None and self.dense_residual:
            out = xs + [out]
        else:
            out = [out]

        if self.use_conv_masks:
            return out, lens
        else:
            return out, None


class JasperEncoder(nn.Module):
    __constants__ = ["use_conv_masks"]

    def __init__(self, in_feats, activation, frame_splicing=1,
                 init='xavier_uniform', use_conv_masks=False, blocks=[]):
        super(JasperEncoder, self).__init__()

        self.use_conv_masks = use_conv_masks
        self.layers = nn.ModuleList()

        in_feats *= frame_splicing
        all_residual_panes = []
        for i,blk in enumerate(blocks):

            blk['activation'] = activations[activation]()

            has_residual_dense = blk.pop('residual_dense', False)
            if has_residual_dense:
                all_residual_panes += [in_feats]
                blk['residual_panes'] = all_residual_panes
            else:
                blk['residual_panes'] = []

            self.layers.append(
                JasperBlock(in_feats, use_conv_masks=use_conv_masks, **blk))

            in_feats = blk['filters']

        self.apply(lambda x: init_weights(x, mode=init))

    def forward(self, x, x_lens=None):
        out, out_lens = [x], x_lens
        for l in self.layers:
            out, out_lens = l(out, out_lens)

        return out, out_lens


class JasperDecoderForCTC(nn.Module):
    def __init__(self, in_feats, n_classes, init='xavier_uniform'):
        super(JasperDecoderForCTC, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_feats, n_classes, kernel_size=1, bias=True),)
        self.apply(lambda x: init_weights(x, mode=init))

    def forward(self, enc_out):
        out = self.layers(enc_out[-1]).transpose(1, 2)
        return F.log_softmax(out, dim=2)


class GreedyCTCDecoder(nn.Module):
    @torch.no_grad()
    def forward(self, log_probs, log_prob_lens=None):

        if log_prob_lens is not None:
            max_len = log_probs.size(1)
            idxs = torch.arange(max_len, dtype=log_prob_lens.dtype,
                                device=log_prob_lens.device)
            mask = idxs.unsqueeze(0) >= log_prob_lens.unsqueeze(1)
            log_probs[:,:,-1] = log_probs[:,:,-1].masked_fill(mask, float("Inf"))

        return log_probs.argmax(dim=-1, keepdim=False).int()


class Jasper(nn.Module):
    def __init__(self, encoder_kw, decoder_kw, transpose_in=False):
        super(Jasper, self).__init__()
        self.transpose_in = transpose_in
        self.encoder = JasperEncoder(**encoder_kw)
        self.decoder = JasperDecoderForCTC(**decoder_kw)

    def forward(self, x, x_lens=None):
        if self.encoder.use_conv_masks:
            assert x_lens is not None
            enc, enc_lens = self.encoder(x, x_lens)
            out = self.decoder(enc)
            return out, enc_lens
        else:
            if self.transpose_in:
                x = x.transpose(1, 2)
            enc, _ = self.encoder(x)
            out = self.decoder(enc)
            return out  # torchscript refuses to output None

    # TODO Explicitly add x_lens=None for inference (now x can be a Tensor or tuple)
    def infer(self, x, x_lens=None):
        if self.encoder.use_conv_masks:
            return self.forward(x, x_lens)
        else:
            ret = self.forward(x)
            return ret, len(ret)


class CTCLossNM:
    def __init__(self, n_classes):
        self._criterion = nn.CTCLoss(blank=n_classes-1, reduction='none')

    def __call__(self, log_probs, targets, input_length, target_length):
        input_length = input_length.long()
        target_length = target_length.long()
        targets = targets.long()
        loss = self._criterion(log_probs.transpose(1, 0), targets, input_length,
                               target_length)
        # note that this is different from reduction = 'mean'
        # because we are not dividing by target lengths
        return torch.mean(loss)
