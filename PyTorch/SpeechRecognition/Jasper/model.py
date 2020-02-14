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

from apex import amp
import torch
import torch.nn as nn
from parts.features import FeatureFactory
import random


jasper_activations = {
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

class AudioPreprocessing(nn.Module):
    """GPU accelerated audio preprocessing
    """
    __constants__ = ["optim_level"]
    def __init__(self, **kwargs):
        nn.Module.__init__(self)    # For PyTorch API
        self.optim_level = kwargs.get('optimization_level', 0)
        self.featurizer = FeatureFactory.from_config(kwargs)
        self.transpose_out = kwargs.get("transpose_out", False)

    @torch.no_grad()
    def forward(self, input_signal, length):
        processed_signal = self.featurizer(input_signal, length)
        processed_length = self.featurizer.get_seq_len(length)    
        if self.transpose_out:
            processed_signal.transpose_(2,1)
            return processed_signal, processed_length
        else:
            return processed_signal, processed_length
                
class SpectrogramAugmentation(nn.Module):
    """Spectrogram augmentation
    """
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.spec_cutout_regions = SpecCutoutRegions(kwargs)
        self.spec_augment = SpecAugment(kwargs)

    @torch.no_grad()
    def forward(self, input_spec):
        augmented_spec = self.spec_cutout_regions(input_spec)
        augmented_spec = self.spec_augment(augmented_spec)
        return augmented_spec

class SpecAugment(nn.Module):
    """Spec augment. refer to https://arxiv.org/abs/1904.08779
    """
    def __init__(self, cfg):
        super(SpecAugment, self).__init__()
        self.cutout_x_regions = cfg.get('cutout_x_regions', 0)
        self.cutout_y_regions = cfg.get('cutout_y_regions', 0)

        self.cutout_x_width = cfg.get('cutout_x_width', 10)
        self.cutout_y_width = cfg.get('cutout_y_width', 10)

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        mask = torch.zeros(x.shape).byte()
        for idx in range(sh[0]):
            for _ in range(self.cutout_x_regions):
                cutout_x_left = int(random.uniform(0, sh[1] - self.cutout_x_width))

                mask[idx, cutout_x_left:cutout_x_left + self.cutout_x_width, :] = 1

            for _ in range(self.cutout_y_regions):
                cutout_y_left = int(random.uniform(0, sh[2] - self.cutout_y_width))

                mask[idx, :, cutout_y_left:cutout_y_left + self.cutout_y_width] = 1

        x = x.masked_fill(mask.to(device=x.device), 0)

        return x

class SpecCutoutRegions(nn.Module):
    """Cutout. refer to https://arxiv.org/pdf/1708.04552.pdf
    """
    def __init__(self, cfg):
        super(SpecCutoutRegions, self).__init__()

        self.cutout_rect_regions = cfg.get('cutout_rect_regions', 0)
        self.cutout_rect_time = cfg.get('cutout_rect_time', 5)
        self.cutout_rect_freq = cfg.get('cutout_rect_freq', 20)

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        mask = torch.zeros(x.shape, dtype=torch.uint8)

        for idx in range(sh[0]):
            for i in range(self.cutout_rect_regions):
                cutout_rect_x = int(random.uniform(
                        0, sh[1] - self.cutout_rect_freq))
                cutout_rect_y = int(random.uniform(
                        0, sh[2] - self.cutout_rect_time))

                mask[idx, cutout_rect_x:cutout_rect_x + self.cutout_rect_freq,
                         cutout_rect_y:cutout_rect_y + self.cutout_rect_time] = 1

        x = x.masked_fill(mask.to(device=x.device), 0)

        return x

class JasperEncoder(nn.Module):
    __constants__ = ["use_conv_mask"]    
    """Jasper encoder
    """
    def __init__(self, **kwargs):
        cfg = {}
        for key, value in kwargs.items():
            cfg[key] = value

        nn.Module.__init__(self)
        self._cfg = cfg

        activation = jasper_activations[cfg['encoder']['activation']]()
        self.use_conv_mask = cfg['encoder'].get('convmask', False)
        feat_in = cfg['input']['features'] * cfg['input'].get('frame_splicing', 1)
        init_mode = cfg.get('init_mode', 'xavier_uniform')

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for lcfg in cfg['jasper']:
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            encoder_layers.append(
                JasperBlock(feat_in, lcfg['filters'], repeat=lcfg['repeat'],
                                        kernel_size=lcfg['kernel'], stride=lcfg['stride'],
                                        dilation=lcfg['dilation'], dropout=lcfg['dropout'],
                                        residual=lcfg['residual'], activation=activation,
                                        residual_panes=dense_res, use_conv_mask=self.use_conv_mask))
            feat_in = lcfg['filters']

        self.encoder = nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if self.use_conv_mask:
            audio_signal, length = x
            return self.encoder(([audio_signal], length))
        else:
            return self.encoder([x])

class JasperDecoderForCTC(nn.Module):
    """Jasper decoder
    """
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self._feat_in = kwargs.get("feat_in")
        self._num_classes = kwargs.get("num_classes")
        init_mode = kwargs.get('init_mode', 'xavier_uniform')

        self.decoder_layers = nn.Sequential(
            nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True),)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, encoder_output):
        out = self.decoder_layers(encoder_output[-1]).transpose(1, 2)
        return nn.functional.log_softmax(out, dim=2)

class JasperEncoderDecoder(nn.Module):
    """Contains jasper encoder and decoder
    """
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.transpose_in=kwargs.get("transpose_in", False)
        self.jasper_encoder = JasperEncoder(**kwargs.get("jasper_model_definition"))
        self.jasper_decoder = JasperDecoderForCTC(feat_in=kwargs.get("feat_in"),
                                                  num_classes=kwargs.get("num_classes"))
        
    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    
    def forward(self, x):
        if self.jasper_encoder.use_conv_mask:
            t_encoded_t, t_encoded_len_t = self.jasper_encoder(x)
        else:
            if self.transpose_in:
                x = x.transpose(1, 2)   
            t_encoded_t = self.jasper_encoder(x)
            
        out = self.jasper_decoder(t_encoded_t)
        if self.jasper_encoder.use_conv_mask:
            return out, t_encoded_len_t
        else:
            return out

    def infer(self, x):
        if self.jasper_encoder.use_conv_mask:
            return self.forward(x)
        else:
            ret = self.forward(x[0])
            return ret, len(ret)
        
    
class Jasper(JasperEncoderDecoder):
    """Contains data preprocessing, spectrogram augmentation, jasper encoder and decoder
    """
    def __init__(self, **kwargs):
        JasperEncoderDecoder.__init__(self, **kwargs)
        feature_config = kwargs.get("feature_config")
        if self.transpose_in:
            feature_config["transpose"] = True
        self.audio_preprocessor = AudioPreprocessing(**feature_config)
        self.data_spectr_augmentation = SpectrogramAugmentation(**feature_config)


class MaskedConv1d(nn.Conv1d):
    """1D convolution with sequence masking
    """
    __constants__ = ["use_conv_mask"]
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                             padding=0, dilation=1, groups=1, bias=False, use_conv_mask=True):
        super(MaskedConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                                                             stride=stride,
                                                                             padding=padding, dilation=dilation,
                                                                             groups=groups, bias=bias)
        self.use_conv_mask = use_conv_mask

    def get_seq_len(self, lens):
        return ((lens + 2 * self.padding[0] - self.dilation[0] * (
            self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)

    def forward(self, inp):
        if self.use_conv_mask:
            x, lens = inp
            max_len = x.size(2)
            idxs = torch.arange(max_len).to(lens.dtype).to(lens.device).expand(len(lens), max_len)
            mask = idxs >= lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
            del mask
            del idxs
            lens = self.get_seq_len(lens)
            return super(MaskedConv1d, self).forward(x), lens
        else:
            return super(MaskedConv1d, self).forward(inp)


class JasperBlock(nn.Module):
    __constants__ = ["use_conv_mask", "conv"]

    """Jasper Block. See https://arxiv.org/pdf/1904.03288.pdf
    """
    def __init__(self, inplanes, planes, repeat=3, kernel_size=11, stride=1,
                             dilation=1, padding='same', dropout=0.2, activation=None,
                             residual=True, residual_panes=[], use_conv_mask=False):
        super(JasperBlock, self).__init__()

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")


        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.use_conv_mask = use_conv_mask
        self.conv = nn.ModuleList()
        inplanes_loop = inplanes
        for _ in range(repeat - 1):
            self.conv.extend(
                self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size,
                                                                stride=stride, dilation=dilation,
                                                                padding=padding_val))
            self.conv.extend(
                self._get_act_dropout_layer(drop_prob=dropout, activation=activation))
            inplanes_loop = planes
        self.conv.extend(
            self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size,
                                                            stride=stride, dilation=dilation,
                                                            padding=padding_val))

        self.res = nn.ModuleList() if residual else None
        res_panes = residual_panes.copy()
        self.dense_residual = residual
        if residual:
            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                self.res.append(nn.ModuleList(
                    modules=self._get_conv_bn_layer(ip, planes, kernel_size=1)))
        self.out = nn.Sequential(
            *self._get_act_dropout_layer(drop_prob=dropout, activation=activation))

    def _get_conv_bn_layer(self, in_channels, out_channels, kernel_size=11,
                                                 stride=1, dilation=1, padding=0, bias=False):
        layers = [
            MaskedConv1d(in_channels, out_channels, kernel_size, stride=stride,
                                     dilation=dilation, padding=padding, bias=bias,
                                     use_conv_mask=self.use_conv_mask),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]
        return layers

    def _get_act_dropout_layer(self, drop_prob=0.2, activation=None):
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [
            activation,
            nn.Dropout(p=drop_prob)
        ]
        return layers

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_):
        if self.use_conv_mask:
            xs, lens_orig = input_
        else:
            xs = input_
            lens_orig = 0
        # compute forward convolutions
        out = xs[-1]
        lens = lens_orig
        for i, l in enumerate(self.conv):
            if self.use_conv_mask and isinstance(l, MaskedConv1d):
                out, lens = l((out, lens))
            else:
                out = l(out)
        # compute the residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if j == 0 and self.use_conv_mask:
                        res_out, _ = res_layer((res_out, lens_orig))
                    else:
                        res_out = res_layer(res_out)
                out += res_out

        # compute the output
        out = self.out(out)
        if self.res is not None and self.dense_residual:
            out = xs + [out]
        else:
            out = [out]

        if self.use_conv_mask:
            return out, lens
        else:
            return out

class GreedyCTCDecoder(nn.Module):
    """ Greedy CTC Decoder
    """
    def __init__(self, **kwargs):
        nn.Module.__init__(self)    # For PyTorch API
    @torch.no_grad()
    def forward(self, log_probs):
            argmx = log_probs.argmax(dim=-1, keepdim=False).int()
            return argmx

class CTCLossNM:
    """ CTC loss
    """
    def __init__(self, **kwargs):
        self._blank = kwargs['num_classes'] - 1
        self._criterion = nn.CTCLoss(blank=self._blank, reduction='none')

    def __call__(self, log_probs, targets, input_length, target_length):
        input_length = input_length.long()
        target_length = target_length.long()
        targets = targets.long()
        loss = self._criterion(log_probs.transpose(1, 0), targets, input_length,
                                                     target_length)
        # note that this is different from reduction = 'mean'
        # because we are not dividing by target lengths
        return torch.mean(loss)
