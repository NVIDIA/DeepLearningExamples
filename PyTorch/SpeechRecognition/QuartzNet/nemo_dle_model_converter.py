import argparse
import io
import sys
from copy import deepcopy
from functools import reduce
from pathlib import Path
from subprocess import CalledProcessError, check_output

import torch
import yaml

import quartznet.config
from common import helpers
from common.features import FilterbankFeatures
from quartznet.config import load as load_yaml
from quartznet.model import QuartzNet, MaskedConv1d


# Corresponding DLE <-> NeMo config keys
cfg_key_map = {
    ("input_val", "audio_dataset", "sample_rate"): ("preprocessor", "sample_rate"),
    ("input_val", "filterbank_features", "dither"): ("preprocessor", "dither"),
    ("input_val", "filterbank_features", "frame_splicing"): ("preprocessor", "frame_splicing"),
    ("input_val", "filterbank_features", "n_fft"): ("preprocessor", "n_fft"),
    ("input_val", "filterbank_features", "n_filt"): ("preprocessor", "features"),
    ("input_val", "filterbank_features", "normalize"): ("preprocessor", "normalize"),
    ("input_val", "filterbank_features", "sample_rate"): ("preprocessor", "sample_rate"),
    ("input_val", "filterbank_features", "window"): ("preprocessor", "window"),
    ("input_val", "filterbank_features", "window_size"): ("preprocessor", "window_size"),
    ("input_val", "filterbank_features", "window_stride"): ("preprocessor", "window_stride"),
    ("labels",): ("decoder", "vocabulary"),
    ("quartznet", "decoder", "in_feats"): ("decoder", "feat_in"),
    ("quartznet", "encoder", "activation"): ("encoder", "activation"),
    ("quartznet", "encoder", "blocks"): ("encoder", "jasper"),
    ("quartznet", "encoder", "frame_splicing"): ("preprocessor", "frame_splicing"),
    ("quartznet", "encoder", "in_feats"): ("encoder", "feat_in"),
    ("quartznet", "encoder", "use_conv_masks"): ("encoder", "conv_mask"),
}


def load_nemo_ckpt(fpath):
    """Make a DeepLearningExamples state_dict and config from a .nemo file."""
    try:
        cmd = ['tar', 'Oxzf', fpath, './model_config.yaml']
        nemo_cfg = yaml.safe_load(io.BytesIO(check_output(cmd)))

        cmd = ['tar', 'Oxzf', fpath, './model_weights.ckpt']
        ckpt = torch.load(io.BytesIO(check_output(cmd)), map_location="cpu")

    except (FileNotFoundError, CalledProcessError):
        print('WARNING: Could not uncompress with tar. '
              'Falling back to the tarfile module (might take a few minutes).')
        import tarfile
        with tarfile.open(fpath, "r:gz") as tar:
            f = tar.extractfile(tar.getmember("./model_config.yaml"))
            nemo_cfg = yaml.safe_load(f)

            f = tar.extractfile(tar.getmember("./model_weights.ckpt"))
            ckpt = torch.load(f, map_location="cpu")

    remap = lambda k: (k.replace("encoder.encoder", "encoder.layers")
                       .replace("decoder.decoder_layers", "decoder.layers")
                       .replace("conv.weight", "weight"))
    dle_ckpt = {'state_dict': {remap(k): v for k, v in ckpt.items()
                               if "preproc" not in k}}
    dle_cfg = config_from_nemo(nemo_cfg)
    return dle_ckpt, dle_cfg


def save_nemo_ckpt(dle_ckpt, dle_cfg, dest_path):
    """Save a DeepLearningExamples model as a .nemo file."""
    cfg = deepcopy(dle_cfg)

    dle_ckpt = torch.load(dle_ckpt, map_location="cpu")["ema_state_dict"]

    # Build a DLE model instance and fill with weights
    symbols = helpers.add_ctc_blank(cfg['labels'])
    enc_kw = quartznet.config.encoder(cfg)
    dec_kw = quartznet.config.decoder(cfg, n_classes=len(symbols))
    model = QuartzNet(enc_kw, dec_kw)
    model.load_state_dict(dle_ckpt, strict=True)

    # Reaname core modules, e.g., encoder.layers -> encoder.encoder
    model.encoder._modules['encoder'] = model.encoder._modules.pop('layers')
    model.decoder._modules['decoder_layers'] = model.decoder._modules.pop('layers')

    # MaskedConv1d is made via composition in NeMo, and via inheritance in DLE
    # Params for MaskedConv1d in NeMo have an additional '.conv.' infix
    def rename_convs(module):
        for name in list(module._modules.keys()):
            submod = module._modules[name]

            if isinstance(submod, MaskedConv1d):
                module._modules[f'{name}.conv'] = module._modules.pop(name)
            else:
                rename_convs(submod)

    rename_convs(model.encoder.encoder)

    # Use FilterbankFeatures to calculate fbanks and store with model weights
    feature_processor = FilterbankFeatures(
        **dle_cfg['input_val']['filterbank_features'])

    nemo_ckpt = model.state_dict()
    nemo_ckpt["preprocessor.featurizer.fb"] = feature_processor.fb
    nemo_ckpt["preprocessor.featurizer.window"] = feature_processor.window

    nemo_cfg = config_to_nemo(dle_cfg)

    # Prepare the directory for zipping
    ckpt_files = dest_path / "ckpt_files"
    ckpt_files.mkdir(exist_ok=True, parents=False)
    with open(ckpt_files / "model_config.yaml", "w") as f:
        yaml.dump(nemo_cfg, f)
    torch.save(nemo_ckpt, ckpt_files / "model_weights.ckpt")

    with tarfile.open(dest_path / "quartznet.nemo", "w:gz") as tar:
        tar.add(ckpt_files, arcname="./")


def save_dle_ckpt(ckpt, cfg, dest_dir):
    torch.save(ckpt, dest_dir / "model.pt")
    with open(dest_dir / "model_config.yaml", "w") as f:
        yaml.dump(cfg, f)


def set_nested_item(tgt, src, tgt_keys, src_keys):
    """Assigns nested dict keys, e.g., d1[a][b][c] = d2[e][f][g][h]."""
    tgt_nested = reduce(lambda d, k: d[k], tgt_keys[:-1], tgt)
    tgt_nested[tgt_keys[-1]] = reduce(lambda d, k: d[k], src_keys, src)


def config_from_nemo(nemo_cfg):
    """Convert a DeepLearningExamples config to a NeMo format."""
    dle_cfg = {
        'name': 'QuartzNet',
        'input_val': {
            'audio_dataset': {
                'normalize_transcripts': True,
            },
            'filterbank_features': {
                'pad_align': 16,
            },
        },
        'quartznet': {
            'decoder': {},
            'encoder': {},
        },
    }

    for dle_keys, nemo_keys in cfg_key_map.items():
        try:
            set_nested_item(dle_cfg, nemo_cfg, dle_keys, nemo_keys)
        except KeyError:
            print(f'WARNING: Could not load config {nemo_keys} as {dle_keys}.')

    # mapping kernel_size is not expressable with cfg_map
    for block in dle_cfg["quartznet"]["encoder"]["blocks"]:
        block["kernel_size"] = block.pop("kernel")

    return dle_cfg


def config_to_nemo(dle_cfg):
    """Convert a DeepLearningExamples config to a NeMo format."""
    nemo_cfg = {
        "target": "nemo.collections.asr.models.ctc_models.EncDecCTCModel",
        "dropout": 0.0,
        "preprocessor": {
            "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
            "stft_conv": False,
        },
        "encoder": {
            "_target_": "nemo.collections.asr.modules.ConvASREncoder",
            "jasper": {}
        },
        "decoder": {
          "_target_": "nemo.collections.asr.modules.ConvASRDecoder",
        },
    }

    for dle_keys, nemo_keys in cfg_key_map.items():
        try:
            set_nested_item(nemo_cfg, dle_cfg, nemo_keys, dle_keys)
        except KeyError:
            print(f"WARNING: Could not load config {dle_keys} as {nemo_keys}.")

    nemo_cfg["sample_rate"] = nemo_cfg["preprocessor"]["sample_rate"]
    nemo_cfg["repeat"] = nemo_cfg["encoder"]["jasper"][1]["repeat"]
    nemo_cfg["separable"] = nemo_cfg["encoder"]["jasper"][1]["separable"]
    nemo_cfg["labels"] = nemo_cfg["decoder"]["vocabulary"]
    nemo_cfg["decoder"]["num_classes"] = len(nemo_cfg["decoder"]["vocabulary"])

    # mapping kernel_size is not expressable with cfg_map
    for block in nemo_cfg["encoder"]["jasper"]:
        if "kernel_size" in block:
            block["kernel"] = block.pop("kernel_size")

    return nemo_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuartzNet DLE <-> NeMo model converter.")
    parser.add_argument("source_model", type=Path,
                        help="A DLE or NeMo QuartzNet model to be converted (.pt or .nemo, respectively)")
    parser.add_argument("dest_dir", type=Path, help="Destination directory")
    parser.add_argument("--dle_config_yaml", type=Path,
                        help="A DLE config .yaml file, required only to convert DLE -> NeMo")
    args = parser.parse_args()

    ext = args.source_model.suffix.lower()
    if ext == ".nemo":
        ckpt, cfg = load_nemo_ckpt(args.source_model)
        save_dle_ckpt(ckpt, cfg, args.dest_dir)

    elif ext == ".pt":
        dle_cfg = load_yaml(args.dle_config_yaml)
        save_nemo_ckpt(args.source_model, dle_cfg, args.dest_dir)

    else:
        raise ValueError(f"Unknown extension {ext}.")

    print('Converted succesfully.')
