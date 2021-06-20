# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Layers package definition."""
from official.nlp.modeling.layers.attention import *  # pylint: disable=wildcard-import
from official.nlp.modeling.layers.dense_einsum import DenseEinsum
from official.nlp.modeling.layers.masked_softmax import MaskedSoftmax
from official.nlp.modeling.layers.on_device_embedding import OnDeviceEmbedding
from official.nlp.modeling.layers.position_embedding import PositionEmbedding
from official.nlp.modeling.layers.self_attention_mask import SelfAttentionMask
from official.nlp.modeling.layers.transformer import Transformer
from official.nlp.modeling.layers.transformer_scaffold import TransformerScaffold
