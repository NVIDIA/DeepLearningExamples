import copy
import inspect
import yaml

from .model import JasperDecoderForCTC, JasperBlock, JasperEncoder
from common.audio import GainPerturbation, ShiftPerturbation, SpeedPerturbation
from common.dataset import AudioDataset
from common.features import CutoutAugment, FilterbankFeatures, SpecAugment
from common.helpers import print_once


def default_args(klass):
    sig = inspect.signature(klass.__init__)
    return {k: v.default for k,v in sig.parameters.items() if k != 'self'}


def load(fpath):
    if fpath.endswith('.toml'):
        raise ValueError('.toml config format has been changed to .yaml')

    cfg = yaml.safe_load(open(fpath, 'r'))

    # Reload to deep copy shallow copies, which were made with yaml anchors
    yaml.Dumper.ignore_aliases = lambda *args: True
    cfg = yaml.dump(cfg)
    cfg = yaml.safe_load(cfg)
    return cfg


def validate_and_fill(klass, user_conf, ignore_unk=[], optional=[]):
    conf = default_args(klass)

    for k,v in user_conf.items():
        assert k in conf or k in ignore_unk, f'Unknown parameter {k} for {klass}'
        conf[k] = v

    # Keep only mandatory or optional-nonempty
    conf = {k:v for k,v in conf.items()
            if k not in optional or v is not inspect.Parameter.empty}

    # Validate
    for k,v in conf.items():
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

    conf_features = validate_and_fill(
        FilterbankFeatures, conf_features)

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
    for blk in conf['jasper']['encoder']['blocks']:
        validate_and_fill(JasperBlock, blk, optional=['infilters'],
                          ignore_unk=['residual_dense'])

    return validate_and_fill(JasperEncoder, conf['jasper']['encoder'])


def decoder(conf, n_classes):
    decoder_kw = {'n_classes': n_classes, **conf['jasper']['decoder']}
    return validate_and_fill(JasperDecoderForCTC, decoder_kw)


def apply_duration_flags(cfg, max_duration, pad_to_max_duration):
    if max_duration is not None:
        cfg['input_train']['audio_dataset']['max_duration'] = max_duration
        cfg['input_train']['filterbank_features']['max_duration'] = max_duration

    if pad_to_max_duration:
        assert cfg['input_train']['audio_dataset']['max_duration'] > 0
        cfg['input_train']['audio_dataset']['pad_to_max_duration'] = True
        cfg['input_train']['filterbank_features']['pad_to_max_duration'] = True
