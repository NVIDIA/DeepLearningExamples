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

import torch
from data.data_utils import InputTypes, DataTypes
from matplotlib import pyplot as plt
from models.interpretability import InterpretableModelBase
from models.tft_pyt.modeling import TemporalFusionTransformer
from mpl_toolkits.axes_grid1 import make_axes_locatable


class InterpretableTFTBase(InterpretableModelBase):
    def __init__(self, *args, **kwargs):
        super(InterpretableTFTBase, self).__init__(*args, **kwargs)

    @classmethod
    def _get_future_features(cls, features):
        future_features = [feature.name for feature in features if feature.feature_type == InputTypes.KNOWN
                           and feature.feature_embed_type == DataTypes.CATEGORICAL] \
                        + [feature.name for feature in features if feature.feature_type == InputTypes.KNOWN
                            and feature.feature_embed_type == DataTypes.CONTINUOUS]
        return future_features

    @classmethod
    def _get_history_features(cls, features):
        history_features = [feature.name for feature in features if feature.feature_type == InputTypes.OBSERVED
                            and feature.feature_embed_type == DataTypes.CATEGORICAL] \
                         + [feature.name for feature in features if feature.feature_type == InputTypes.OBSERVED
                            and feature.feature_embed_type == DataTypes.CONTINUOUS] \
                         + [feature.name for feature in features if feature.feature_type == InputTypes.KNOWN
                            and feature.feature_embed_type == DataTypes.CATEGORICAL] \
                         + [feature.name for feature in features if feature.feature_type == InputTypes.KNOWN
                            and feature.feature_embed_type == DataTypes.CONTINUOUS] \
                         + [feature.name for feature in features if feature.feature_type == InputTypes.TARGET]

        return history_features

    @classmethod
    def _get_heatmap_fig(cls, tensor, features, max_size=16, min_size=4):
        shape = tensor.shape
        ratio = max(max(shape) // max_size, 1)
        fig_size = max(shape[1] / ratio, min_size), max(shape[0] / ratio, min_size)
        fig = plt.figure(figsize=fig_size)
        ticks = list(range(shape[0]))
        plt.yticks(ticks, features)
        plt.xlabel('Time step')
        plt.imshow(tensor, cmap='hot', interpolation='nearest')
        plt.colorbar()
        return fig

    @classmethod
    def _get_vsn_fig(cls, activations, sample_number, features):
        _, sparse_matrix = activations
        sample_sparse_matrix = sparse_matrix[sample_number]
        final_tensor = sample_sparse_matrix.permute(1, 0)
        fig = cls._get_heatmap_fig(final_tensor.detach().cpu(), features)
        return fig

    @classmethod
    def _get_attention_heatmap_fig(cls, heads, max_size=16, min_size=4):
        row_size = max(min_size, max_size / len(heads))
        fig, axes = plt.subplots(1, len(heads), figsize=(max_size, row_size))
        for i, (head, ax) in enumerate(zip(heads, axes), 1):
            im = ax.imshow(head, cmap='hot', interpolation='nearest')
            if i < len(heads):
                ax.set_title(f'HEAD {i}')
            else:
                ax.set_title('MEAN')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
        return fig

    @classmethod
    def _get_attn_heads(cls, activations, sample_number):
        heads = []
        _, attn_prob = activations
        sample_attn_prob = attn_prob[sample_number]
        n_heads = sample_attn_prob.shape[0]
        for head_index in range(n_heads):
            head = sample_attn_prob[head_index]
            heads.append(head.detach().cpu())
        mean_head = torch.mean(sample_attn_prob, dim=0)
        heads.append(mean_head.detach().cpu())
        fig = cls._get_attention_heatmap_fig(heads)
        return fig

    def _get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output

        return hook

    def get_activations(self, sample_number, features):
        assert self.activations, "There are no activations available"
        return {
            "history_vsn": self._get_vsn_fig(self.activations['history_vsn'], sample_number,
                                             self._get_history_features(features)),
            "future_vsn": self._get_vsn_fig(self.activations['future_vsn'], sample_number,
                                            self._get_future_features(features)),
            "attention": self._get_attn_heads(self.activations['attention'], sample_number)
        }

    def _register_interpretable_hooks(self):
        self.TFTpart2.history_vsn.register_forward_hook(self._get_activation('history_vsn'))
        self.TFTpart2.future_vsn.register_forward_hook(self._get_activation('future_vsn'))
        self.TFTpart2.attention.register_forward_hook(self._get_activation('attention'))


class InterpretableTFT(TemporalFusionTransformer, InterpretableTFTBase):
    def __init__(self, *args, **kwargs):
        TemporalFusionTransformer.__init__(self, *args, **kwargs)
        InterpretableTFTBase.__init__(self, *args, **kwargs)
