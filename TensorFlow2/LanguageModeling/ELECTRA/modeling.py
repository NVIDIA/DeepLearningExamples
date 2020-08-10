# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import logging

import tensorflow as tf

from configuration import ElectraConfig

from file_utils import add_start_docstrings, add_start_docstrings_to_callable
from modeling_utils import ACT2FN, TFBertEncoder, TFBertPreTrainedModel
from modeling_utils import get_initializer, shape_list
from tokenization_utils import BatchEncoding


logger = logging.getLogger(__name__)


TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "google/electra-small-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-generator/tf_model.h5",
    "google/electra-base-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base-generator/tf_model.h5",
    "google/electra-large-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-large-generator/tf_model.h5",
    "google/electra-small-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-discriminator/tf_model.h5",
    "google/electra-base-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base-discriminator/tf_model.h5",
    "google/electra-large-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-large-discriminator/tf_model.h5",
}


class TFElectraEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.initializer_range = config.initializer_range

        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.embedding_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="position_embeddings",
        )
        self.token_type_embeddings = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.embedding_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="token_type_embeddings",
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def build(self, input_shape):
        """Build shared word embedding layer """
        with tf.name_scope("word_embeddings"):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.word_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )
        super().build(input_shape)

    def call(self, inputs, mode="embedding", training=False):
        """Get token embeddings of inputs.
        Args:
            inputs: list of three int64 tensors with shape [batch_size, length]: (input_ids, position_ids, token_type_ids)
            mode: string, a valid value is one of "embedding" and "linear".
        Returns:
            outputs: (1) If mode == "embedding", output embedding tensor, float32 with
                shape [batch_size, length, embedding_size]; (2) mode == "linear", output
                linear tensor, float32 with shape [batch_size, length, vocab_size].
        Raises:
            ValueError: if mode is not valid.

        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        if mode == "embedding":
            return self._embedding(inputs, training=training)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, inputs, training=False):
        """Applies embedding based on inputs tensor."""
        input_ids, position_ids, token_type_ids, inputs_embeds = inputs

        if input_ids is not None:
            input_shape = shape_list(input_ids)
        else:
            input_shape = shape_list(inputs_embeds)[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        if inputs_embeds is None:
            inputs_embeds = tf.gather(self.word_embeddings, input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.
            Args:
                inputs: A float32 tensor with shape [batch_size, length, hidden_size]
            Returns:
                float32 tensor with shape [batch_size, length, vocab_size].
        """
        batch_size = shape_list(inputs)[0]
        length = shape_list(inputs)[1]

        x = tf.reshape(inputs, [-1, self.embedding_size])
        logits = tf.matmul(x, self.word_embeddings, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])


class TFElectraDiscriminatorPredictions(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(config.hidden_size, name="dense")
        self.dense_prediction = tf.keras.layers.Dense(1, name="dense_prediction")
        self.config = config

    def call(self, discriminator_hidden_states, training=False):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = ACT2FN[self.config.hidden_act](hidden_states)
        logits = tf.squeeze(self.dense_prediction(hidden_states))

        return logits


class TFElectraGeneratorPredictions(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dense = tf.keras.layers.Dense(config.embedding_size, name="dense")

    def call(self, generator_hidden_states, training=False):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = ACT2FN["gelu"](hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class TFElectraPreTrainedModel(TFBertPreTrainedModel):

    config_class = ElectraConfig
    pretrained_model_archive_map = TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "electra"

    def get_extended_attention_mask(self, attention_mask, input_shape):
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def get_head_mask(self, head_mask):
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        return head_mask


class TFElectraMainLayer(TFElectraPreTrainedModel):

    config_class = ElectraConfig

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.embeddings = TFElectraEmbeddings(config, name="embeddings")

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = tf.keras.layers.Dense(config.hidden_size, name="embeddings_project")
        self.encoder = TFBertEncoder(config, name="encoder")
        self.config = config

    def get_input_embeddings(self):
        return self.embeddings

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        raise NotImplementedError

    def call(
        self,
        inputs,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        training=False,
    ):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
            position_ids = inputs[3] if len(inputs) > 3 else position_ids
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            inputs_embeds = inputs[5] if len(inputs) > 5 else inputs_embeds
            assert len(inputs) <= 6, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            assert len(inputs) <= 6, "Too many inputs."
        else:
            input_ids = inputs

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        head_mask = self.get_head_mask(head_mask)

        hidden_states = self.embeddings([input_ids, position_ids, token_type_ids, inputs_embeds], training=training)

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states, training=training)

        hidden_states = self.encoder([hidden_states, extended_attention_mask, head_mask], training=training)

        return hidden_states


ELECTRA_START_DOCSTRING = r"""
    This model is a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ sub-class.
    Use it as a regular TF 2.0 Keras Model and
    refer to the TF 2.0 documentation for all matter related to general usage and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

            - having all inputs as keyword arguments (like PyTorch models), or
            - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :obj:`tf.keras.Model.fit()` method which currently requires having
        all the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors
        in the first positional argument :

        - a single Tensor with input_ids only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({'input_ids': input_ids, 'token_type_ids': token_type_ids})`

    Parameters:
        config (:class:`~transformers.ElectraConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ELECTRA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.ElectraTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        head_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, embedding_dim)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        training (:obj:`boolean`, `optional`, defaults to :obj:`False`):
            Whether to activate dropout modules (if set to :obj:`True`) during training or to de-activate them
            (if set to :obj:`False`) for evaluation.

"""


@add_start_docstrings(
    "The bare Electra Model transformer outputting raw hidden-states without any specific head on top. Identical to "
    "the BERT model except that it uses an additional linear layer between the embedding layer and the encoder if the "
    "hidden size and embedding size are different."
    ""
    "Both the generator and discriminator checkpoints may be loaded into this model.",
    ELECTRA_START_DOCSTRING,
)
class TFElectraModel(TFElectraPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.electra = TFElectraMainLayer(config, name="electra")

    def get_input_embeddings(self):
        return self.electra.embeddings

    @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        r"""
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.ElectraConfig`) and inputs:
        last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import ElectraTokenizer, TFElectraModel

        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        model = TFElectraModel.from_pretrained('google/electra-small-discriminator')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        """
        outputs = self.electra(inputs, **kwargs)
        return outputs


@add_start_docstrings(
    """
Electra model with a binary classification head on top as used during pre-training for identifying generated
tokens.

Even though both the discriminator and generator may be loaded into this model, the discriminator is
the only model of the two to have the correct classification head to be used for this model.""",
    ELECTRA_START_DOCSTRING,
)
class TFElectraForPreTraining(TFElectraPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.electra = TFElectraMainLayer(config, name="electra")
        self.discriminator_predictions = TFElectraDiscriminatorPredictions(config, name="discriminator_predictions")

    def get_input_embeddings(self):
        return self.electra.embeddings

    @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        training=False,
    ):
        r"""
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.ElectraConfig`) and inputs:
        scores (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import ElectraTokenizer, TFElectraForPreTraining

        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        model = TFElectraForPreTraining.from_pretrained('google/electra-small-discriminator')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        scores = outputs[0]
        """

        discriminator_hidden_states = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, training=training
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        logits = self.discriminator_predictions(discriminator_sequence_output)
        output = (logits,)
        output += discriminator_hidden_states[1:]

        return output  # (loss), scores, (hidden_states), (attentions)


class TFElectraMaskedLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias")
        super().build(input_shape)

    def call(self, hidden_states, training=False):
        hidden_states = self.input_embeddings(hidden_states, mode="linear")
        hidden_states = hidden_states + self.bias
        return hidden_states


@add_start_docstrings(
    """
Electra model with a language modeling head on top.

Even though both the discriminator and generator may be loaded into this model, the generator is
the only model of the two to have been trained for the masked language modeling task.""",
    ELECTRA_START_DOCSTRING,
)
class TFElectraForMaskedLM(TFElectraPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.vocab_size = config.vocab_size
        self.electra = TFElectraMainLayer(config, name="electra")
        self.generator_predictions = TFElectraGeneratorPredictions(config, name="generator_predictions")
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.generator_lm_head = TFElectraMaskedLMHead(config, self.electra.embeddings, name="generator_lm_head")

    def get_input_embeddings(self):
        return self.electra.embeddings

    def get_output_embeddings(self):
        return self.generator_lm_head

    @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        training=False,
    ):
        r"""
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.ElectraConfig`) and inputs:
        prediction_scores (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import ElectraTokenizer, TFElectraForMaskedLM

        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
        model = TFElectraForMaskedLM.from_pretrained('google/electra-small-generator')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        prediction_scores = outputs[0]

        """

        generator_hidden_states = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, training=training
        )
        generator_sequence_output = generator_hidden_states[0]
        prediction_scores = self.generator_predictions(generator_sequence_output, training=training)
        prediction_scores = self.generator_lm_head(prediction_scores, training=training)
        output = (prediction_scores,)
        output += generator_hidden_states[1:]

        return output  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


@add_start_docstrings(
    """
Electra model with a token classification head on top.

Both the discriminator and generator may be loaded into this model.""",
    ELECTRA_START_DOCSTRING,
)
class TFElectraForTokenClassification(TFElectraPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.electra = TFElectraMainLayer(config, name="electra")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels, name="classifier")

    @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        training=False,
    ):
        r"""
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.ElectraConfig`) and inputs:
        scores (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import ElectraTokenizer, TFElectraForTokenClassification

        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        model = TFElectraForTokenClassification.from_pretrained('google/electra-small-discriminator')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        scores = outputs[0]
        """

        discriminator_hidden_states = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, training=training
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        logits = self.classifier(discriminator_sequence_output)
        output = (logits,)
        output += discriminator_hidden_states[1:]

        return output  # (loss), scores, (hidden_states), (attentions)


class TFPoolerStartLogits(tf.keras.Model):
    """ Compute SQuAD start_logits from sequence hidden states. """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.dense = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="start_logit_pooler_dense"
        )

    def call(self, hidden_states, p_mask=None, next_layer_dtype=tf.float32):
        """ Args:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape `(batch_size, seq_len)`
                invalid position mask such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        x = tf.squeeze(self.dense(hidden_states), axis=-1,
                       name="squeeze_start_logit_pooler")

        if p_mask is not None:
            if self.dense.dtype == tf.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class TFPoolerEndLogits(tf.keras.Model):
    """ Compute SQuAD end_logits from sequence hidden states and start token hidden state.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.dense_0 = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range),
            name="end_logit_pooler_dense_0"
        )

        self.activation = tf.keras.layers.Activation('tanh')  # nn.Tanh()
        self.LayerNorm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=config.layer_norm_eps,
                                                            name="end_logit_pooler_LayerNorm")
        self.dense_1 = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="end_logit_pooler_dense_1"
        )

    def call(self, hidden_states, start_states=None, start_positions=None, p_mask=None, training=False,
             next_layer_dtype=tf.float32):
        """ Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.
            **start_states**: ``torch.LongTensor`` of shape identical to hidden_states
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
                Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        assert (
                start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None and training:
            bsz, slen, hsz = hidden_states.shape
            start_states = tf.gather(hidden_states, start_positions[:, None], axis=1,
                                     batch_dims=1)  # shape (bsz, 1, hsz)
            start_states = tf.broadcast_to(start_states, (bsz, slen, hsz))  # shape (bsz, slen, hsz)

        x = self.dense_0(tf.concat([hidden_states, start_states], axis=-1))
        x = self.activation(x)
        if training:
            # since we are not doing beam search, add dimension with value=1. corresponds to dimension with top_k during inference - if not layernorm crashes
            x = tf.expand_dims(x, axis=2)
        x = self.LayerNorm(x)

        if training:
            # undo the additional dimension added above
            x = tf.squeeze(self.dense_1(x), axis=[-1, -2])
        else:
            x = tf.squeeze(self.dense_1(x), axis=-1)

        if p_mask is not None:
            if next_layer_dtype == tf.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class TFPoolerAnswerClass(tf.keras.Model):
    """ Compute SQuAD 2.0 answer class from classification and start tokens hidden states. """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.dense_0 = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range),
            name="pooler_answer_class_dense_0"
        )

        self.activation = tf.keras.layers.Activation('tanh')
        self.dense_1 = tf.keras.layers.Dense(
            1, use_bias=False, kernel_initializer=get_initializer(config.initializer_range),
            name="pooler_answer_class_dense_1"
        )

    def call(self, hidden_states, start_states=None, start_positions=None, cls_index=None):
        """
        Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.
            **start_states**: ``torch.LongTensor`` of shape identical to ``hidden_states``.
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span.
            **cls_index**: torch.LongTensor of shape ``(batch_size,)``
                position of the CLS token. If None, take the last token.
            note(Original repo):
                no dependency on end_feature so that we can obtain one single `cls_logits`
                for each sample
        """
        assert (
                start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_states = tf.gather(hidden_states, start_positions[:, None], axis=1,
                                     batch_dims=1)  # shape (bsz, 1, hsz)
            start_states = tf.squeeze(start_states, axis=1)  # shape (bsz, hsz)

        if cls_index is not None:
            cls_token_state = tf.gather(hidden_states, cls_index[:, None], axis=1, batch_dims=1)  # shape (bsz, 1, hsz)
            cls_token_state = tf.squeeze(cls_token_state, axis=1)  # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, 0, :]  # shape (bsz, hsz)

        x = self.dense_0(tf.concat([start_states, cls_token_state], axis=-1))
        x = self.activation(x)
        x = tf.squeeze(self.dense_1(x), axis=-1)

        return x


class TFElectraForQuestionAnswering(TFElectraPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config, args)

        self.start_n_top = args.beam_size  # config.start_n_top
        self.end_n_top = args.beam_size  # config.end_n_top
        self.joint_head = args.joint_head
        self.v2 = args.version_2_with_negative
        self.electra = TFElectraMainLayer(config, name="electra")
        self.num_hidden_layers = config.num_hidden_layers
        self.amp = config.amp

        ##old head
        if not self.joint_head:
            self.qa_outputs = tf.keras.layers.Dense(
                2, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs")
        else:
            self.start_logits = TFPoolerStartLogits(config, name='start_logits')
            self.end_logits = TFPoolerEndLogits(config, name='end_logits')
            if self.v2:
                self.answer_class = TFPoolerAnswerClass(config, name='answer_class')

    def call(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            start_positions=None,
            end_positions=None,
            cls_index=None,
            p_mask=None,
            is_impossible=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            training=False,
    ):
        outputs = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, training=training
        )
        discriminator_sequence_output = outputs[0]

        # Simple head model
        if not self.joint_head:
            logits = self.qa_outputs(discriminator_sequence_output)
            [start_logits, end_logits] = tf.split(logits, 2, axis=-1)
            start_logits = tf.squeeze(start_logits, axis=-1, name="squeeze_start_logit")
            end_logits = tf.squeeze(end_logits, axis=-1, name="squeeze_end_logit")
            outputs = (start_logits, end_logits) + outputs
            return outputs

        start_logits = self.start_logits(discriminator_sequence_output, p_mask=p_mask,
                                         next_layer_dtype=self.end_logits.dense_0.dtype)
        if training:  # start_positions is not None and end_positions is not None:

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(discriminator_sequence_output, start_positions=start_positions, p_mask=p_mask,
                                         training=training,
                                         next_layer_dtype=tf.float16 if self.amp else tf.float32)

            if self.v2:  # cls_index is not None:#cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(discriminator_sequence_output, start_positions=start_positions,
                                               cls_index=cls_index)

            else:
                cls_logits = None

            outputs = (start_logits, end_logits, cls_logits) + outputs

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = discriminator_sequence_output.shape
            start_n_top = min(self.start_n_top, slen)
            end_n_top = min(self.end_n_top, slen)
            start_log_probs = tf.nn.log_softmax(start_logits, axis=-1, name="start_logit_softmax")  # shape (bsz, slen)

            start_top_log_probs, start_top_index = tf.math.top_k(start_log_probs, k=start_n_top,
                                                                 name="start_log_probs_top_k")

            start_states = tf.gather(discriminator_sequence_output, start_top_index, axis=1,
                                     batch_dims=1)  # shape (bsz, start_n_top, hsz)
            start_states = tf.broadcast_to(tf.expand_dims(start_states, axis=1),
                                           [bsz, slen, start_n_top, hsz])  # shape (bsz, slen, start_n_top, hsz)

            discriminator_sequence_output_expanded = tf.broadcast_to(
                tf.expand_dims(discriminator_sequence_output, axis=2),
                list(start_states.shape))  # shape (bsz, slen, start_n_top, hsz)

            p_mask = tf.expand_dims(p_mask, axis=-1) if p_mask is not None else None
            end_logits = self.end_logits(discriminator_sequence_output_expanded, start_states=start_states,
                                         p_mask=p_mask, next_layer_dtype=tf.float16 if self.amp else tf.float32)  # self.answer_class.dense_0.dtype)
            end_log_probs = tf.nn.log_softmax(end_logits, axis=1,
                                              name="end_logit_softmax")  # shape (bsz, slen, start_n_top)

            # need to transpose because tf.math.top_k works on default axis=-1
            end_log_probs = tf.transpose(end_log_probs, perm=[0, 2, 1])
            end_top_log_probs, end_top_index = tf.math.top_k(
                end_log_probs, k=end_n_top)  # shape (bsz, end_n_top, start_n_top).perm(0,2,1)
            end_top_log_probs = tf.reshape(end_top_log_probs, (
                -1, start_n_top * end_n_top))  # shape (bsz, self.start_n_top * self.end_n_top)
            end_top_index = tf.reshape(end_top_index,
                                       (-1, start_n_top * end_n_top))  # shape (bsz, self.start_n_top * self.end_n_top)
            if self.v2:  # cls_index is not None:
                start_p = tf.nn.softmax(start_logits, axis=-1, name="start_softmax")
                start_states = tf.einsum(
                    "blh,bl->bh", discriminator_sequence_output, start_p
                )  # get the representation of START as weighted sum of hidden states
                # explicitly setting cls_index to None
                cls_logits = self.answer_class(
                    discriminator_sequence_output, start_states=start_states, cls_index=None)
                # one single `cls_logits` for each sample
            else:
                cls_logits = tf.fill([bsz], 0.0)

            outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits) + outputs

        # return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits
        return outputs
