# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

import json
import copy
from dataclasses import dataclass
import logging
import paddle
import paddle.nn as nn
try:
    from paddle.incubate.nn import FusedTransformerEncoderLayer
except ImportError:
    FusedTransformerEncoderLayer = None

__all__ = [
    'BertModel', 'BertForPretraining', 'BertPretrainingHeads',
    'BertForQuestionAnswering'
]


@dataclass
class BertConfig:
    vocab_size: int = 30528
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    output_all_encoded_layers: bool = False
    pad_token_id: int = 0

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertEmbeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, bert_config):
        super().__init__()
        self.word_embeddings = nn.Embedding(bert_config.vocab_size,
                                            bert_config.hidden_size)
        self.position_embeddings = nn.Embedding(
            bert_config.max_position_embeddings, bert_config.hidden_size)
        self.token_type_embeddings = nn.Embedding(bert_config.type_vocab_size,
                                                  bert_config.hidden_size)
        self.layer_norm = nn.LayerNorm(bert_config.hidden_size, epsilon=1e-12)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        """
        Args:
            See class `BertModel`.
        """
        ones = paddle.ones_like(input_ids, dtype="int64")
        seq_length = paddle.cumsum(ones, axis=-1)
        position_ids = seq_length - ones
        position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(nn.Layer):
    """
    Pool the result of BertEncoder.
    """

    def __init__(self, hidden_size, pool_act=nn.Tanh()):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = pool_act

    def forward(self, hidden_states):
        """
        Args:
            hidden_states(Tensor): A Tensor of hidden_states.
        """
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Layer):
    """
    The bare BERT Model transformer outputting raw hidden-states.

    Args:
        bert_config(BertConfig): A BertConfig class instance with the configuration
            to build a new model
    """

    def __init__(self, bert_config):
        super().__init__()
        self.bert_config = bert_config
        self.embeddings = BertEmbeddings(bert_config)

        self.fuse = True if FusedTransformerEncoderLayer is not None else False
        self.fuse = False

        if self.fuse:
            self.encoder = nn.LayerList([
                FusedTransformerEncoderLayer(
                    bert_config.hidden_size,
                    bert_config.num_attention_heads,
                    bert_config.intermediate_size,
                    dropout_rate=bert_config.hidden_dropout_prob,
                    activation=bert_config.hidden_act,
                    attn_dropout_rate=bert_config.attention_probs_dropout_prob,
                    act_dropout_rate=0.)
                for _ in range(bert_config.num_hidden_layers)
            ])
        else:
            logging.warning(
                "FusedTransformerEncoderLayer is not supported by the running Paddle. "
                "TransformerEncoderLayer will be used.")
            encoder_layer = nn.TransformerEncoderLayer(
                bert_config.hidden_size,
                bert_config.num_attention_heads,
                bert_config.intermediate_size,
                dropout=bert_config.hidden_dropout_prob,
                activation=bert_config.hidden_act,
                attn_dropout=bert_config.attention_probs_dropout_prob,
                act_dropout=0,
                fuse_qkv=bert_config.fuse_mha)
            self.encoder = nn.TransformerEncoder(encoder_layer,
                                                 bert_config.num_hidden_layers)

        self.pooler = BertPooler(bert_config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids(Tensor):
                A Tensor of shape [batch_size, sequence_length] with the word token
                indices in the vocabulary. Data type should be `int64`.
            token_type_ids(Tensor, optional):
                An optional Tensor of shape [batch_size, sequence_length] with the token types
                indices selected in [0, type_vocab_size - 1].
                If `type_vocab_size` is 2, indices can either be 0 or 1. Type 0 corresponds
                to a `sentence A` and type 1 corresponds to a `sentence B` token.
                (see BERT paper for more details). Its data type should be `int64`
                Defaults: None, which means we don't add segment embeddings.
            attention_mask(Tensor, optional):
                An optional Tensor of shape [batch_size, sequence_length] with indices of
                mask used in multi-head attention to avoid performing attention on to some
                unwanted positions, usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                Defaults: None.

        Returns:
            encoder_output(Tensor):
                A Tensor of shape [batch_size, sequence_length, hidden_size] contains hidden-states at the last
                layer of the model. The data type should be float32.

            pooled_output(Tensor):
                A Tensor of shape [batch_size, hidden_size] which is the output of a classifier pretrained on
                top of the hidden state associated to the first character of the input (`CLS`) to train on the
                Next-Sentence task (see BERT's paper).
        """

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids != self.bert_config.pad_token_id).astype('int32'),
                axis=[1, 2])
        else:
            if attention_mask.ndim == 2:
                # attention_mask [batch_size, sequence_length] -> [batch_size, 1, 1, sequence_length]
                attention_mask = attention_mask.unsqueeze(axis=[1, 2])

        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids)

        if self.fuse:
            encoder_output = embedding_output
            for layer in self.encoder:
                encoder_output = layer(encoder_output, attention_mask)
        else:
            encoder_output = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output


class BertForQuestionAnswering(nn.Layer):
    """
    BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Args:
        bert_config(BertConfig): a BertConfig class instance with the configuration to build a new model.
    """

    def __init__(self, bert_config):
        super().__init__()
        self.bert = BertModel(bert_config)
        self.classifier = nn.Linear(bert_config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            See class `BertModel`.

        Returns:
        start_logits(Tensor):
            A tensor of shape [batch_size, sequence_length] indicates the start position token.
        end_logits(Tensor):
            A tensor of shape [batch_size, sequence_length] indicates the end position token.
        """

        encoder_output, _ = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)

        logits = self.classifier(encoder_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


class BertLMPredictionHead(nn.Layer):
    """
    Bert Model with a `language modeling` head on top for CLM fine-tuning.

    Args:
        hidden_size(int): See class `BertConfig`.
        vocab_size(int): See class `BertConfig`.
        activation(str): Activation function used in the language modeling task.
        embedding_weights(Tensor, optional):
            An optional Tensor of shape [vocab_size, hidden_size] used to map hidden_states
            to logits of the masked token prediction. The data type should be float32.
            Defaults: None, which means use the same weights of the embedding layer.
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super().__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=1e-12)
        self.decoder_weight = self.create_parameter(
            shape=[vocab_size, hidden_size],
            dtype=self.transform.weight.dtype,
            is_bias=False) if embedding_weights is None else embedding_weights
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = paddle.tensor.matmul(
            hidden_states, self.decoder_weight,
            transpose_y=True) + self.decoder_bias
        return hidden_states


class BertPretrainingHeads(nn.Layer):
    """
    Perform language modeling task and next sentence classification task.

    Args:
        hidden_size(int): See class `BertConfig`.
        vocab_size(int): See class `BertConfig`.
        activation(str): Activation function used in the language modeling task.
        embedding_weights (Tensor, optional):
            An optional Tensor of shape [vocab_size, hidden_size] used to map hidden_states
            to logits of the masked token prediction. The data type should be float32.
            Defaults: None, which means use the same weights of the embedding layer.
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super().__init__()
        self.predictions = BertLMPredictionHead(hidden_size, vocab_size,
                                                activation, embedding_weights)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, encoder_output, pooled_output, masked_lm_labels):
        """
        Args:
            sequence_output(Tensor):
                A Tensor of shape [batch_size, sequence_length, hidden_size] with hidden-states
                at the last layer of bert model. It's data type should be float32.
            pooled_output(Tensor):
                A Tensor of shape [batch_size, hidden_size] with output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32.
            masked_positions(Tensor, optional):
                An optional tensor of shape [batch_size, mask_token_num] indicates positions to be masked
                in the position embedding. Its data type should be int64. Default: None
        Returns:
            prediction_scores(Tensor):
                A Tensor with the scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, the shape is [batch_size, mask_token_num, vocab_size].
            seq_relationship_score(Tensor):
                A Tensor of shape [batch_size, 2] with the scores of next sentence prediction.
                Its data type should be float32.
        """

        sequence_flattened = paddle.index_select(
            encoder_output.reshape([-1, encoder_output.shape[-1]]),
            paddle.nonzero(masked_lm_labels.reshape([-1]) != -1).squeeze(),
            axis=0)
        prediction_scores = self.predictions(sequence_flattened)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertForPretraining(nn.Layer):
    """
    Bert Model with pretraining tasks on top.

    Args:
        bert_config(Class BertConfig): An instance of class `BertConfig`.
    """

    def __init__(self, bert_config):
        super().__init__()
        self.bert = BertModel(bert_config)
        self.cls = BertPretrainingHeads(
            bert_config.hidden_size,
            bert_config.vocab_size,
            bert_config.hidden_act,
            embedding_weights=self.bert.embeddings.word_embeddings.weight)

    def forward(self, input_ids, token_type_ids, attention_mask,
                masked_lm_labels):
        """

        Args:
            input_ids(Tensor): See class `BertModel`.
            token_type_ids(Tensor, optional): See class `BertModel`.
            attention_mask(Tensor, optional): See class `BertModel`.
            masked_positions(Tensor, optional): See class `BertPretrainingHeads`.

        Returns:
            prediction_scores(Tensor):
                A Tensor with the scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size].
            seq_relationship_score(Tensor):
                A Tensor of shape [batch_size, 2] with the scores of next sentence prediction.
                Its data type should be float32.
        """
        with paddle.static.amp.fp16_guard():
            outputs = self.bert(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask)
            sequence_output, pooled_output = outputs[:2]
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, masked_lm_labels)
            return prediction_scores, seq_relationship_score
