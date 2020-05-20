#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
#============================================================================

"""Utils to handle parameters IO."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import yaml

import tensorflow as tf


def save_hparams_to_yaml(hparams, file_path):
    with tf.io.gfile.GFile(file_path, 'w') as f:
        try:
            hparams_val = hparams.values()
        except AttributeError:
            hparams_val = hparams.__dict__
        yaml.dump(hparams_val, f)


def override_hparams(hparams, dict_or_string_or_yaml_file):
    """Override a given hparams using a dict or a string or a JSON file.

  Args:
    hparams: a HParams object to be overridden.
    dict_or_string_or_yaml_file: a Python dict, or a comma-separated string,
      or a path to a YAML file specifying the parameters to be overridden.

  Returns:
    hparams: the overridden HParams object.

  Raises:
    ValueError: if failed to override the parameters.
  """
    if not dict_or_string_or_yaml_file:
        return hparams

    if isinstance(dict_or_string_or_yaml_file, dict):

        for key, val in dict_or_string_or_yaml_file.items():

            if key not in hparams:
                try:  # TF 1.x
                    hparams.add_hparam(key, val)
                except AttributeError:  # TF 2.x
                    try:  # Dict
                        hparams[key] = val
                    except TypeError:  # Namespace
                        setattr(hparams, key, val)
            else:
                raise ValueError("Parameter `%s` is already defined" % key)

        # hparams.override_from_dict(dict_or_string_or_yaml_file)

    elif isinstance(dict_or_string_or_yaml_file, six.string_types):
        try:
            hparams.parse(dict_or_string_or_yaml_file)

        except ValueError as parse_error:
            try:
                with tf.io.gfile.GFile(dict_or_string_or_yaml_file) as f:
                    hparams.override_from_dict(yaml.load(f))

            except Exception as read_error:
                parse_message = ('Failed to parse config string: %s\n' % parse_error.message)
                read_message = ('Failed to parse yaml file provided. %s' % read_error.message)
                raise ValueError(parse_message + read_message)

    else:
        raise ValueError('Unknown input type to parse.')
    return hparams
