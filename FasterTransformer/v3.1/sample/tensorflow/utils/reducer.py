# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import abc
import tensorflow as tf

def pad_in_time(x, padding_length):
    """Helper function to pad a tensor in the time dimension and retain the static depth dimension."""
    return tf.pad(x, [[0, 0], [0, padding_length], [0, 0]])


def align_in_time(x, length):
    """Aligns the time dimension of :obj:`x` with :obj:`length`."""
    time_dim = tf.shape(x)[1]
    return tf.cond(
        tf.less(time_dim, length),
        true_fn=lambda: pad_in_time(x, length - time_dim),
        false_fn=lambda: x[:, :length])


def pad_with_identity(x, sequence_length, max_sequence_length, identity_values=0, maxlen=None):
    """Pads a tensor with identity values up to :obj:`max_sequence_length`.
    Args:
      x: A ``tf.Tensor`` of shape ``[batch_size, time, depth]``.
      sequence_length: The true sequence length of :obj:`x`.
      max_sequence_length: The sequence length up to which the tensor must contain
        :obj:`identity values`.
      identity_values: The identity value.
      maxlen: Size of the output time dimension. Default is the maximum value in
        obj:`max_sequence_length`.
    Returns:
      A ``tf.Tensor`` of shape ``[batch_size, maxlen, depth]``.
    """
    if maxlen is None:
        maxlen = tf.reduce_max(max_sequence_length)

    mask = tf.sequence_mask(sequence_length, maxlen=maxlen, dtype=x.dtype)
    mask = tf.expand_dims(mask, axis=-1)
    mask_combined = tf.sequence_mask(
        max_sequence_length, maxlen=maxlen, dtype=x.dtype)
    mask_combined = tf.expand_dims(mask_combined, axis=-1)

    identity_mask = mask_combined * (1.0 - mask)

    x = pad_in_time(x, maxlen - tf.shape(x)[1])
    x = x * mask + (identity_mask * identity_values)

    return x


def pad_n_with_identity(inputs, sequence_lengths, identity_values=0):
    """Pads each input tensors with identity values up to
    ``max(sequence_lengths)`` for each batch.
    Args:
      inputs: A list of ``tf.Tensor``.
      sequence_lengths: A list of sequence length.
      identity_values: The identity value.
    Returns:
      A tuple ``(padded, max_sequence_length)`` which are respectively a list of
      ``tf.Tensor`` where each tensor are padded with identity and the combined
      sequence length.
    """
    max_sequence_length = tf.reduce_max(sequence_lengths, axis=0)
    maxlen = tf.reduce_max([tf.shape(x)[1] for x in inputs])
    padded = [
        pad_with_identity(
            x, length, max_sequence_length, identity_values=identity_values, maxlen=maxlen)
        for x, length in zip(inputs, sequence_lengths)]
    return padded, max_sequence_length


class Reducer():
    """Base class for reducers."""

    def zip_and_reduce(self, x, y):
        """Zips the :obj:`x` with :obj:`y` structures together and reduces all
        elements. If the structures are nested, they will be flattened first.
        Args:
          x: The first structure.
          y: The second structure.
        Returns:
          The same structure as :obj:`x` and :obj:`y` where each element from
          :obj:`x` is reduced with the correspond element from :obj:`y`.
        Raises:
          ValueError: if the two structures are not the same.
        """
        tf.nest.assert_same_structure(x, y)
        x_flat = tf.nest.flatten(x)
        y_flat = tf.nest.flatten(y)
        reduced = list(map(self, zip(x_flat, y_flat)))
        return tf.nest.pack_sequence_as(x, reduced)

    def __call__(self, inputs, sequence_length=None):
        """Reduces all input elements.
        Args:
          inputs: A list of ``tf.Tensor``.
          sequence_length: The length of each input, if reducing sequences.
        Returns:
          If :obj:`sequence_length` is set, a tuple
          ``(reduced_input, reduced_length)``, otherwise a reduced ``tf.Tensor``
          only.
        """
        if sequence_length is None:
            return self.reduce(inputs)
        else:
            return self.reduce_sequence(inputs, sequence_lengths=sequence_length)

    @abc.abstractmethod
    def reduce(self, inputs):
        """See :meth:`opennmt.layers.Reducer.__call__`."""
        raise NotImplementedError()

    @abc.abstractmethod
    def reduce_sequence(self, inputs, sequence_lengths):
        """See :meth:`opennmt.layers.Reducer.__call__`."""
        raise NotImplementedError()


class SumReducer(Reducer):
    """A reducer that sums the inputs."""

    def reduce(self, inputs):
        return tf.add_n(inputs)

    def reduce_sequence(self, inputs, sequence_lengths):
        padded, combined_length = pad_n_with_identity(
            inputs, sequence_lengths, identity_values=0)
        return self.reduce(padded), combined_length
