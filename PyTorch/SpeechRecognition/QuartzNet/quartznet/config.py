# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import copy
import inspect
from ast import literal_eval
from contextlib import suppress
from numbers import Number

import yaml

from common.audio import GainPerturbation, ShiftPerturbation, SpeedPerturbation
from common.dataset import AudioDataset
from common.features import (CutoutAugment, FilterbankFeatures, SpecAugment)
from quartznet.model import JasperDecoderForCTC, JasperBlock, JasperEncoder


def default_args(klass):
    sig = inspect.signature(klass.__init__)
    return {k: v.default for k, v in sig.parameters.items() if k != 'self'}


def load(fpath):

    cfg = yaml.safe_load(open(fpath, 'r'))

    # Reload to deep copy shallow copies, which were made with yaml anchors
    yaml.Dumper.ignore_aliases = lambda *args: True
    cfg = yaml.dump(cfg)
    cfg = yaml.safe_load(cfg)
    return cfg


def validate_and_fill(klass, user_conf, ignore_unk=[], optional=[]):
    conf = default_args(klass)

    for k, v in user_conf.items():
        assert k in conf or k in ignore_unk, f'Unknown param {k} for {klass}'
        conf[k] = v

    # Keep only mandatory or optional-nonempty
    conf = {k: v for k, v in conf.items()
            if k not in optional or v is not inspect.Parameter.empty}

    # Validate
    for k, v in conf.items():
        assert v is not inspect.Parameter.empty, \
            f'Value for {k} not specified for {klass}'
    return conf


def input(conf_yaml, split='train'):

    conf = copy.deepcopy(conf_yaml[f'input_{split}'])
    conf_dataset = conf.pop('audio_dataset')
    conf_features = conf.pop('filterbank_features')

    # Validate known inner classes
    inner_classes = [
        (conf_dataset, 'speed_perturbation', SpeedPerturbation),
        (conf_dataset, 'gain_perturbation', GainPerturbation),
        (conf_dataset, 'shift_perturbation', ShiftPerturbation),
        (conf_features, 'spec_augment', SpecAugment),
        (conf_features, 'cutout_augment', CutoutAugment),
    ]
    for conf_tgt, key, klass in inner_classes:
        if key in conf_tgt:
            conf_tgt[key] = validate_and_fill(klass, conf_tgt[key])

    for k in conf:
        raise ValueError(f'Unknown key {k}')

    # Validate outer classes
    conf_dataset = validate_and_fill(
        AudioDataset, conf_dataset,
        optional=['data_dir', 'labels', 'manifest_fpaths'])

    # klass = feature_class(conf_features['feature_type'])
    # conf_features = validate_and_fill(
    #     klass, conf_features, ignore_unk=['feature_type'])

    conf_features = validate_and_fill(
        FilterbankFeatures, conf_features)  # , ignore_unk=['feature_type'])

    # Check params shared between classes
    shared = ['sample_rate', 'max_duration', 'pad_to_max_duration']
    for sh in shared:
        assert conf_dataset[sh] == conf_features[sh], (
            f'{sh} should match in Dataset and FeatureProcessor: '
            f'{conf_dataset[sh]}, {conf_features[sh]}')

    return conf_dataset, conf_features


def encoder(conf):
    """Validate config for JasperEncoder and subsequent JasperBlocks"""

    # Validate, but don't overwrite with defaults
    for blk in conf['quartznet']['encoder']['blocks']:
        validate_and_fill(JasperBlock, blk, optional=['infilters'],
                          ignore_unk=['residual_dense'])

    return validate_and_fill(JasperEncoder, conf['quartznet']['encoder'])


def decoder(conf, n_classes):
    decoder_kw = {'n_classes': n_classes, **conf['quartznet']['decoder']}
    return validate_and_fill(JasperDecoderForCTC, decoder_kw)


def apply_config_overrides(conf, args):
    if args.override_config is None:
        return
    for override_key_val in args.override_config:
        key, val = override_key_val.split('=')
        with suppress(TypeError, ValueError):
            val = literal_eval(val)
        apply_nested_config_override(conf, key, val)


def apply_nested_config_override(conf, key_str, val):
    fields = key_str.split('.')
    for f in fields[:-1]:
        conf = conf[f]
    f = fields[-1]
    assert (f not in conf
            or type(val) is type(conf[f])
            or (isinstance(val, Number) and isinstance(conf[f], Number)))
    conf[f] = val
