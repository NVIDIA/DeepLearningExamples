# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

""" Transforms for 3D data augmentation """
import tensorflow as tf


def apply_transforms(samples, labels, mean, stdev, transforms):
    """ Apply a chain of transforms to a pair of samples and labels """
    for _t in transforms:
        if _t is not None:
            samples, labels = _t(samples, labels, mean, stdev)
    return samples, labels


def apply_test_transforms(samples, mean, stdev, transforms):
    """ Apply a chain of transforms to a samples using during test """
    for _t in transforms:
        if _t is not None:
            samples = _t(samples, labels=None, mean=mean, stdev=stdev)
    return samples


class PadXYZ: # pylint: disable=R0903
    """ Pad volume in three dimensiosn """
    def __init__(self, shape=None):
        """ Add padding

        :param shape: Target shape
        """
        self.shape = shape

    def __call__(self, samples, labels, mean, stdev):
        """ Run op

        :param samples: Sample arrays
        :param labels: Label arrays
        :param mean: Mean (unused)
        :param stdev:  Std (unused)
        :return: Padded samples and labels
        """
        paddings = tf.constant([[0, 0], [0, 0], [0, 5], [0, 0]])
        samples = tf.pad(samples, paddings, "CONSTANT")
        if labels is None:
            return samples
        labels = tf.pad(labels, paddings, "CONSTANT")
        return samples, labels


class CenterCrop: # pylint: disable=R0903
    """ Produce a central crop in 3D """
    def __init__(self, shape):
        """ Create op

        :param shape: Target shape for crop
        """
        self.shape = shape

    def __call__(self, samples, labels, mean, stdev):
        """ Run op

        :param samples: Sample arrays
        :param labels: Label arrays
        :param mean: Mean (unused)
        :param stdev:  Std (unused)
        :return: Cropped samples and labels
        """
        shape = samples.get_shape()
        delta = [(shape[i].value - self.shape[i]) // 2 for i in range(len(self.shape))]
        samples = samples[
            delta[0]:delta[0] + self.shape[0],
            delta[1]:delta[1] + self.shape[1],
            delta[2]:delta[2] + self.shape[2]]
        if labels is None:
            return samples
        labels = labels[
            delta[0]:delta[0] + self.shape[0],
            delta[1]:delta[1] + self.shape[1],
            delta[2]:delta[2] + self.shape[2]]
        return samples, labels


class RandomCrop3D: # pylint: disable=R0903
    """ Produce a random 3D crop """
    def __init__(self, shape, margins=(0, 0, 0)):
        """ Create op

        :param shape: Target shape
        :param margins: Margins within to perform the crop
        """
        self.shape = shape
        self.margins = margins

    def __call__(self, samples, labels, mean, stdev):
        """ Run op

        :param samples: Sample arrays
        :param labels: Label arrays
        :param mean: Mean (unused)
        :param stdev:  Std (unused)
        :return: Cropped samples and labels
        """
        shape = samples.get_shape()
        min_ = tf.constant(self.margins, dtype=tf.float32)
        max_ = tf.constant([shape[0].value - self.shape[0] - self.margins[0],
                            shape[1].value - self.shape[1] - self.margins[1],
                            shape[2].value - self.shape[2] - self.margins[2]],
                           dtype=tf.float32)
        center = tf.random_uniform((len(self.shape),), minval=min_, maxval=max_)
        center = tf.cast(center, dtype=tf.int32)
        samples = samples[center[0]:center[0] + self.shape[0],
                          center[1]:center[1] + self.shape[1],
                          center[2]:center[2] + self.shape[2]]
        if labels is None:
            return samples
        labels = labels[center[0]:center[0] + self.shape[0],
                        center[1]:center[1] + self.shape[1],
                        center[2]:center[2] + self.shape[2]]
        return samples, labels


class NormalizeImages: # pylint: disable=R0903
    """ Run zscore normalization """
    def __call__(self, samples, labels, mean, stdev):
        """ Run op

        :param samples: Sample arrays
        :param labels: Label arrays
        :param mean: Mean
        :param stdev:  Std
        :return: Normalized samples and labels
        """
        mask = tf.math.greater(samples, 0)
        samples = tf.where(mask, (samples - tf.cast(mean, samples.dtype)) / (tf.cast(stdev + 1e-8, samples.dtype)),
                           samples)

        if labels is None:
            return samples
        return samples, labels


class Cast: # pylint: disable=R0903
    """ Cast samples and labels to different precision """
    def __init__(self, dtype=tf.float32):
        self._dtype = dtype

    def __call__(self, samples, labels, mean, stdev):
        """ Run op

        :param samples: Sample arrays
        :param labels: Label arrays
        :param mean: Mean (unused)
        :param stdev:  Std (unused)
        :return: Casted samples and labels
        """
        if labels is None:
            return tf.cast(samples, dtype=self._dtype)
        return tf.cast(samples, dtype=self._dtype), labels


class RandomHorizontalFlip: # pylint: disable=R0903
    """ Randomly flip horizontally a pair of samples and labels"""
    def __init__(self, threshold=0.5):
        self._threshold = threshold

    def __call__(self, samples, labels, mean, stdev):
        """ Run op

        :param samples: Sample arrays
        :param labels: Label arrays
        :param mean: Mean (unused)
        :param stdev:  Std (unused)
        :return: Flipped samples and labels
        """
        h_flip = tf.random_uniform([]) > self._threshold

        samples = tf.cond(h_flip, lambda: tf.reverse(samples, axis=[1]), lambda: samples)
        labels = tf.cond(h_flip, lambda: tf.reverse(labels, axis=[1]), lambda: labels)

        return samples, labels


class RandomVerticalFlip: # pylint: disable=R0903
    """ Randomly flip vertically a pair of samples and labels"""
    def __init__(self, threshold=0.5):
        self._threshold = threshold

    def __call__(self, samples, labels, mean, stdev):
        """ Run op

        :param samples: Sample arrays
        :param labels: Label arrays
        :param mean: Mean (unused)
        :param stdev:  Std (unused)
        :return: Flipped samples and labels
        """
        h_flip = tf.random_uniform([]) > self._threshold

        samples = tf.cond(h_flip, lambda: tf.reverse(samples, axis=[0]), lambda: samples)
        labels = tf.cond(h_flip, lambda: tf.reverse(labels, axis=[0]), lambda: labels)

        return samples, labels


class RandomGammaCorrection: # pylint: disable=R0903
    """ Random gamma correction over samples """
    def __init__(self, gamma_range=(0.8, 1.5), keep_stats=False, threshold=0.5, epsilon=1e-8):
        self._gamma_range = gamma_range
        self._keep_stats = keep_stats
        self._eps = epsilon
        self._threshold = threshold

    def __call__(self, samples, labels, mean, stdev):
        """ Run op

        :param samples: Sample arrays
        :param labels: Label arrays
        :param mean: Mean (unused)
        :param stdev:  Std (unused)
        :return: Gamma corrected samples
        """
        augment = tf.random_uniform([]) > self._threshold
        gamma = tf.random_uniform([], minval=self._gamma_range[0], maxval=self._gamma_range[1])

        x_min = tf.math.reduce_min(samples)
        x_range = tf.math.reduce_max(samples) - x_min

        samples = tf.cond(augment,
                          lambda: tf.math.pow(((samples - x_min) / float(x_range + self._eps)),
                                              gamma) * x_range + x_min,
                          lambda: samples)
        return samples, labels


class RandomBrightnessCorrection: # pylint: disable=R0903
    """ Random brightness correction over samples """
    def __init__(self, alpha=0.1, threshold=0.5, per_channel=True):
        self._alpha_range = [1.0 - alpha, 1.0 + alpha]
        self._threshold = threshold
        self._per_channel = per_channel

    def __call__(self, samples, labels, mean, stdev):
        """ Run op

        :param samples: Sample arrays
        :param labels: Label arrays
        :param mean: Mean (unused)
        :param stdev:  Std (unused)
        :return: Brightness corrected samples
        """
        mask = tf.math.greater(samples, 0)
        size = samples.get_shape()[-1].value if self._per_channel else 1
        augment = tf.random_uniform([]) > self._threshold
        correction = tf.random_uniform([size],
                                       minval=self._alpha_range[0],
                                       maxval=self._alpha_range[1],
                                       dtype=samples.dtype)

        samples = tf.cond(augment,
                          lambda: tf.where(mask, samples + correction, samples),
                          lambda: samples)

        return samples, labels


class OneHotLabels: # pylint: disable=R0903
    """ One hot encoding of labels """
    def __init__(self, n_classes=1):
        self._n_classes = n_classes

    def __call__(self, samples, labels, mean, stdev):
        """ Run op

        :param samples: Sample arrays (unused)
        :param labels: Label arrays
        :param mean: Mean (unused)
        :param stdev:  Std (unused)
        :return: One hot encoded labels
        """
        return samples, tf.one_hot(labels, self._n_classes)
