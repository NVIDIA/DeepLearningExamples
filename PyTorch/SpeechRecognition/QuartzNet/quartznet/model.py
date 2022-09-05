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


def compute_new_kernel_size(kernel_size, kernel_width):
    new_kernel_size = max(int(kernel_size * kernel_width), 1)
    # If kernel is even shape, round up to make it odd
    if new_kernel_size % 2 == 0:
        new_kernel_size += 1
    return new_kernel_size


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (kernel_size // 2) * dilation


class GroupShuffle(nn.Module):
    def __init__(self, groups, channels):
        super(GroupShuffle, self).__init__()
        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        sh = x.shape
        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])
        return x


class MaskedConv1d(nn.Conv1d):
    """1D convolution with sequence masking
    """
    __constants__ = ["masked"]

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, use_mask=True,
                 heads=-1):

        # Jasper refactor compat
        assert heads == -1  # Unsupported
        masked = use_mask

        super(MaskedConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.masked = masked

    def get_seq_len(self, lens):
        pad, ks = self.padding[0], self.kernel_size[0]
        return torch.div(lens + 2 * pad - self.dilation[0] * (ks - 1) - 1,
                         self.stride[0], rounding_mode='trunc') + 1

    def forward(self, x, x_lens=None):
        if self.masked:
            max_len = x.size(2)
            idxs = torch.arange(max_len, dtype=x_lens.dtype, device=x.device)
            mask = idxs.expand(x_lens.size(0), max_len) >= x_lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
            x_lens = self.get_seq_len(x_lens)

        return super(MaskedConv1d, self).forward(x), x_lens


class JasperBlock(nn.Module):
    __constants__ = ["conv_mask", "separable", "res", "mconv"]

    def __init__(self, infilters, filters, repeat=3, kernel_size=11,
                 kernel_size_factor=1, stride=1, dilation=1, padding='same',
                 dropout=0.2, activation=None, residual=True, groups=1,
                 separable=False, heads=-1, normalization="batch",
                 norm_groups=1, residual_panes=[], use_conv_masks=False):
        super(JasperBlock, self).__init__()

        # Fix params being passed as list, but default to ints
        wrap = lambda v: [v] if type(v) is int else v
        kernel_size = wrap(kernel_size)
        dilation = wrap(dilation)
        padding = wrap(padding)
        stride = wrap(stride)

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")

        kernel_size_factor = float(kernel_size_factor)
        if type(kernel_size) in (list, tuple):
            kernel_size = [compute_new_kernel_size(k, kernel_size_factor)
                           for k in kernel_size]
        else:
            kernel_size = compute_new_kernel_size(kernel_size,
                                                  kernel_size_factor)

        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.conv_mask = use_conv_masks
        self.separable = separable

        infilters_loop = infilters
        conv = nn.ModuleList()

        for _ in range(repeat - 1):
            conv.extend(
                self._get_conv_bn_layer(
                    infilters_loop, filters, kernel_size=kernel_size,
                    stride=stride, dilation=dilation, padding=padding_val,
                    groups=groups, heads=heads, separable=separable,
                    normalization=normalization, norm_groups=norm_groups)
            )
            conv.extend(self._get_act_dropout_layer(drop_prob=dropout,
                                                    activation=activation))
            infilters_loop = filters

        conv.extend(
            self._get_conv_bn_layer(
                infilters_loop, filters, kernel_size=kernel_size, stride=stride,
                dilation=dilation, padding=padding_val, groups=groups,
                heads=heads, separable=separable, normalization=normalization,
                norm_groups=norm_groups)
        )
        self.mconv = conv

        res_panes = residual_panes.copy()
        self.dense_residual = residual

        if residual:
            res_list = nn.ModuleList()

            if len(residual_panes) == 0:
                res_panes = [infilters]
                self.dense_residual = False
            for ip in res_panes:
                res_list.append(nn.ModuleList(
                    self._get_conv_bn_layer(ip, filters, kernel_size=1,
                                            normalization=normalization,
                                            norm_groups=norm_groups, stride=[1])
                ))

            self.res = res_list
        else:
            self.res = None

        self.mout = nn.Sequential(*self._get_act_dropout_layer(
            drop_prob=dropout, activation=activation))

    def _get_conv(self, in_channels, out_channels, kernel_size=11, stride=1,
                  dilation=1, padding=0, bias=False, groups=1, heads=-1,
                  separable=False):

        kw = {'in_channels': in_channels, 'out_channels': out_channels,
              'kernel_size': kernel_size, 'stride': stride, 'dilation': dilation,
              'padding': padding, 'bias': bias, 'groups': groups}

        if self.conv_mask:
            return MaskedConv1d(**kw, heads=heads, use_mask=self.conv_mask)
        else:
            return nn.Conv1d(**kw)

    def _get_conv_bn_layer(self, in_channels, out_channels, kernel_size=11,
                           stride=1, dilation=1, padding=0, bias=False,
                           groups=1, heads=-1, separable=False,
                           normalization="batch", norm_groups=1):
        if norm_groups == -1:
            norm_groups = out_channels

        if separable:
            layers = [
                self._get_conv(in_channels, in_channels, kernel_size,
                               stride=stride, dilation=dilation, padding=padding,
                               bias=bias, groups=in_channels, heads=heads),
                self._get_conv(in_channels, out_channels, kernel_size=1,
                               stride=1, dilation=1, padding=0, bias=bias,
                               groups=groups),
            ]
        else:
            layers = [
                self._get_conv(in_channels, out_channels, kernel_size,
                               stride=stride, dilation=dilation,
                               padding=padding, bias=bias, groups=groups)
            ]

        if normalization == "group":
            layers.append(nn.GroupNorm(num_groups=norm_groups,
                                       num_channels=out_channels))
        elif normalization == "instance":
            layers.append(nn.GroupNorm(num_groups=out_channels,
                                       num_channels=out_channels))
        elif normalization == "layer":
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))

        elif normalization == "batch":
            layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))
        else:
            raise ValueError(
                f"Normalization method ({normalization}) does not match"
                f" one of [batch, layer, group, instance]."
            )

        if groups > 1:
            layers.append(GroupShuffle(groups, out_channels))
        return layers

    def _get_act_dropout_layer(self, drop_prob=0.2, activation=None):
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [activation, nn.Dropout(p=drop_prob)]
        return layers

    def forward(self, xs, xs_lens=None):
        if not self.conv_mask:
            xs_lens = 0

        # compute forward convolutions
        out = xs[-1]
        lens = xs_lens
        for i, l in enumerate(self.mconv):
            # if we're doing masked convolutions, we need to pass in and
            # possibly update the sequence lengths
            # if (i % 4) == 0 and self.conv_mask:
            if isinstance(l, MaskedConv1d):
                out, lens = l(out, lens)
            else:
                out = l(out)

        # compute the residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if isinstance(res_layer, MaskedConv1d):
                        res_out, _ = res_layer(res_out, xs_lens)
                    else:
                        res_out = res_layer(res_out)

                out = out + res_out

        # compute the output
        out = self.mout(out)
        if self.res is not None and self.dense_residual:
            out = xs + [out]
        else:
            out = [out]

        return (out, lens) if self.conv_mask else (out, None)


class JasperEncoder(nn.Module):
    __constants__ = ["use_conv_masks"]

    def __init__(self, in_feats, activation, frame_splicing=1,
                 init='xavier_uniform', use_conv_masks=False, blocks=[]):
        super(JasperEncoder, self).__init__()

        self.use_conv_masks = use_conv_masks
        self.layers = nn.ModuleList()

        in_feats *= frame_splicing
        all_residual_panes = []
        for i, blk in enumerate(blocks):

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
        for layer in self.layers:
            out, out_lens = layer(out, out_lens)

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
    def forward(self, log_probs):
        return log_probs.argmax(dim=-1, keepdim=False).int()


class QuartzNet(nn.Module):
    def __init__(self, encoder_kw, decoder_kw, transpose_in=False):
        super(QuartzNet, self).__init__()
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
            return out  # XXX torchscript refuses to output None

    # TODO Explicitly add x_lens=None for inference (now x can be a Tensor or tuple)
    def infer(self, x):
        if self.encoder.use_conv_masks:
            return self.forward(x)
        else:
            ret = self.forward(x[0])
            return ret, len(ret)


class CTCLossNM:
    def __init__(self, n_classes):
        self._criterion = nn.CTCLoss(blank=n_classes-1, reduction='none')

    def __call__(self, log_probs, targets, input_length, target_length):
        input_length = input_length.long()
        target_length = target_length.long()
        targets = targets.long()
        loss = self._criterion(log_probs.transpose(1, 0), targets,
                               input_length, target_length)
        # note that this is different from reduction = 'mean'
        # because we are not dividing by target lengths
        return torch.mean(loss)
