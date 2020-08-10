# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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
import pickle
import sys
import unicodedata
import six

import horovod.tensorflow as hvd
import tensorflow as tf


def get_rank():
    try:
        return hvd.rank()
    except:
        return 0


def get_world_size():
    try:
        return hvd.size()
    except:
        return 1


def is_main_process():
    return get_rank() == 0


def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Training Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Training Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s


def load_json(path):
  with tf.io.gfile.GFile(path, "r") as f:
    return json.load(f)


def write_json(o, path):
  if "/" in path:
    tf.io.gfile.makedirs(path.rsplit("/", 1)[0])
  with tf.io.gfile.GFile(path, "w") as f:
    json.dump(o, f)


def load_pickle(path):
  with tf.io.gfile.GFile(path, "rb") as f:
    return pickle.load(f)


def write_pickle(o, path):
  if "/" in path:
    tf.io.gfile.makedirs(path.rsplit("/", 1)[0])
  with tf.io.gfile.GFile(path, "wb") as f:
    pickle.dump(o, f, -1)


def mkdir(path):
  if not tf.io.gfile.exists(path):
    tf.io.gfile.makedirs(path)


def rmrf(path):
  if tf.io.gfile.exists(path):
    tf.io.gfile.rmtree(path)


def rmkdir(path):
  rmrf(path)
  mkdir(path)


def log(*args):
  msg = " ".join(map(str, args))
  sys.stdout.write(msg + "\n")
  sys.stdout.flush()


def log_config(config):
  for key, value in sorted(config.__dict__.items()):
    log(key, value)
  log()


def heading(*args):
  log(80 * "=")
  log(*args)
  log(80 * "=")


def nest_dict(d, prefixes, delim="_"):
  """Go from {prefix_key: value} to {prefix: {key: value}}."""
  nested = {}
  for k, v in d.items():
    for prefix in prefixes:
      if k.startswith(prefix + delim):
        if prefix not in nested:
          nested[prefix] = {}
        nested[prefix][k.split(delim, 1)[1]] = v
      else:
        nested[k] = v
  return nested


def flatten_dict(d, delim="_"):
  """Go from {prefix: {key: value}} to {prefix_key: value}."""
  flattened = {}
  for k, v in d.items():
    if isinstance(v, dict):
      for k2, v2 in v.items():
        flattened[k + delim + k2] = v2
    else:
      flattened[k] = v
  return flattened


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")