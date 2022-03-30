# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
import argparse
import json
import re
import sys

import torch

from common.text.symbols import get_symbols, get_pad_idx
from common.utils import DefaultAttrDict, AttrDict
from fastpitch.model import FastPitch
from fastpitch.model_jit import FastPitchJIT
from hifigan.models import Generator

try:
    from waveglow.model import WaveGlow
    from waveglow import model as glow
    from waveglow.denoiser import Denoiser
    sys.modules['glow'] = glow
except ImportError:
    print("WARNING: Couldn't import WaveGlow")


def parse_model_args(model_name, parser, add_help=False):
    if model_name == 'FastPitch':
        from fastpitch import arg_parser
        return arg_parser.parse_fastpitch_args(parser, add_help)

    elif model_name == 'HiFi-GAN':
        from hifigan import arg_parser
        return arg_parser.parse_hifigan_args(parser, add_help)

    elif model_name == 'WaveGlow':
        from waveglow.arg_parser import parse_waveglow_args
        return parse_waveglow_args(parser, add_help)

    else:
        raise NotImplementedError(model_name)


def get_model(model_name, model_config, device, bn_uniform_init=False,
              forward_is_infer=False, jitable=False):
    """Chooses a model based on name"""
    del bn_uniform_init  # unused (old name: uniform_initialize_bn_weight)

    if model_name == 'FastPitch':
        if jitable:
            model = FastPitchJIT(**model_config)
        else:
            model = FastPitch(**model_config)

    elif model_name == 'HiFi-GAN':
        model = Generator(model_config)

    elif model_name == 'WaveGlow':
        model = WaveGlow(**model_config)

    else:
        raise NotImplementedError(model_name)

    if forward_is_infer and hasattr(model, 'infer'):
        model.forward = model.infer

    return model.to(device)


def get_model_config(model_name, args, ckpt_config=None):
    """ Get config needed to instantiate the model """

    # Mark keys missing in `args` with an object (None is ambiguous)
    _missing = object()
    args = DefaultAttrDict(lambda: _missing, vars(args))

    # `ckpt_config` is loaded from the checkpoint and has the priority
    # `model_config` is based on args and fills empty slots in `ckpt_config`
    if model_name == 'FastPitch':
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=(len(get_symbols(args.symbol_set))
                       if args.symbol_set is not _missing else _missing),
            padding_idx=(get_pad_idx(args.symbol_set)
                         if args.symbol_set is not _missing else _missing),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
        )
    elif model_name == 'HiFi-GAN':
        if args.hifigan_config is not None:
            assert ckpt_config is None, (
                "Supplied --hifigan-config, but the checkpoint has a config. "
                "Drop the flag or remove the config from the checkpoint file.")
            print(f'HiFi-GAN: Reading model config from {args.hifigan_config}')
            with open(args.hifigan_config) as f:
                args = AttrDict(json.load(f))

        model_config = dict(
            # generator architecture
            upsample_rates=args.upsample_rates,
            upsample_kernel_sizes=args.upsample_kernel_sizes,
            upsample_initial_channel=args.upsample_initial_channel,
            resblock=args.resblock,
            resblock_kernel_sizes=args.resblock_kernel_sizes,
            resblock_dilation_sizes=args.resblock_dilation_sizes,
        )
    elif model_name == 'WaveGlow':
        model_config = dict(
            n_mel_channels=args.n_mel_channels,
            n_flows=args.flows,
            n_group=args.groups,
            n_early_every=args.early_every,
            n_early_size=args.early_size,
            WN_config=dict(
                n_layers=args.wn_layers,
                kernel_size=args.wn_kernel_size,
                n_channels=args.wn_channels
            )
        )
    else:
        raise NotImplementedError(model_name)

    # Start with ckpt_config, and fill missing keys from model_config
    final_config = {} if ckpt_config is None else ckpt_config.copy()
    missing_keys = set(model_config.keys()) - set(final_config.keys())
    final_config.update({k: model_config[k] for k in missing_keys})

    # If there was a ckpt_config, it should have had all args
    if ckpt_config is not None and len(missing_keys) > 0:
        print(f'WARNING: Keys {missing_keys} missing from the loaded config; '
              'using args instead.')

    assert all(v is not _missing for v in final_config.values())
    return final_config


def get_model_train_setup(model_name, args):
    """ Dump train setup for documentation purposes """
    if model_name == 'FastPitch':
        return dict()
    elif model_name == 'HiFi-GAN':
        return dict(
            # audio
            segment_size=args.segment_size,
            filter_length=args.filter_length,
            num_mels=args.num_mels,
            hop_length=args.hop_length,
            win_length=args.win_length,
            sampling_rate=args.sampling_rate,
            mel_fmin=args.mel_fmin,
            mel_fmax=args.mel_fmax,
            mel_fmax_loss=args.mel_fmax_loss,
            max_wav_value=args.max_wav_value,
            # other
            seed=args.seed,
            # optimization
            base_lr=args.learning_rate,
            lr_decay=args.lr_decay,
            epochs_all=args.epochs,
        )
    elif model_name == 'WaveGlow':
        return dict()
    else:
        raise NotImplementedError(model_name)


def load_model_from_ckpt(checkpoint_data, model, key='state_dict'):

    if key is None:
        return checkpoint_data['model'], None

    sd = checkpoint_data[key]
    sd = {re.sub('^module\.', '', k): v for k, v in sd.items()}
    status = model.load_state_dict(sd, strict=False)
    return model, status


def load_and_setup_model(model_name, parser, checkpoint, amp, device,
                         unk_args=[], forward_is_infer=False, jitable=False):
    if checkpoint is not None:
        ckpt_data = torch.load(checkpoint)
        print(f'{model_name}: Loading {checkpoint}...')
        ckpt_config = ckpt_data.get('config')
        if ckpt_config is None:
            print(f'{model_name}: No model config in the checkpoint; using args.')
        else:
            print(f'{model_name}: Found model config saved in the checkpoint.')
    else:
        ckpt_config = None
        ckpt_data = {}

    model_parser = parse_model_args(model_name, parser, add_help=False)
    model_args, model_unk_args = model_parser.parse_known_args()
    unk_args[:] = list(set(unk_args) & set(model_unk_args))

    model_config = get_model_config(model_name, model_args, ckpt_config)

    model = get_model(model_name, model_config, device,
                      forward_is_infer=forward_is_infer,
                      jitable=jitable)

    if checkpoint is not None:
        key = 'generator' if model_name == 'HiFi-GAN' else 'state_dict'
        model, status = load_model_from_ckpt(ckpt_data, model, key)

        missing = [] if status is None else status.missing_keys
        unexpected = [] if status is None else status.unexpected_keys

        # Attention is only used during training, we won't miss it
        if model_name == 'FastPitch':
            missing = [k for k in missing if not k.startswith('attention.')]
            unexpected = [k for k in unexpected if not k.startswith('attention.')]

        assert len(missing) == 0 and len(unexpected) == 0, (
            f'Mismatched keys when loading parameters. Missing: {missing}, '
            f'unexpected: {unexpected}.')

    if model_name == "WaveGlow":
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
        model = model.remove_weightnorm(model)

    elif model_name == 'HiFi-GAN':
        assert model_args.hifigan_config is not None or ckpt_config is not None, (
            'Use a HiFi-GAN checkpoint from NVIDIA DeepLearningExamples with '
            'saved config or supply --hifigan-config <json_file>.')
        model.remove_weight_norm()

    if amp:
        model.half()

    model.eval()
    return model.to(device), model_config, ckpt_data.get('train_setup', {})


def load_and_setup_ts_model(model_name, checkpoint, amp, device=None):
    print(f'{model_name}: Loading TorchScript checkpoint {checkpoint}...')
    model = torch.jit.load(checkpoint).eval()
    if device is not None:
        model = model.to(device)
    
    if amp:
        model.half()
    elif next(model.parameters()).dtype == torch.float16:
        raise ValueError('Trying to load FP32 model,'
                         'TS checkpoint is in FP16 precision.')
    return model


def convert_ts_to_trt(model_name, ts_model, parser, amp, unk_args=[]):
    trt_parser = _parse_trt_compilation_args(model_name, parser, add_help=False)
    trt_args, trt_unk_args = trt_parser.parse_known_args()
    unk_args[:] = list(set(unk_args) & set(trt_unk_args))

    if model_name == 'HiFi-GAN':
        return _convert_ts_to_trt_hifigan(
            ts_model, amp, trt_args.trt_min_opt_max_batch,
            trt_args.trt_min_opt_max_hifigan_length)
    else:
        raise NotImplementedError


def _parse_trt_compilation_args(model_name, parent, add_help=False):
    """
    Parse model and inference specific commandline arguments.
    """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help,
                                     allow_abbrev=False)
    trt = parser.add_argument_group(f'{model_name} Torch-TensorRT compilation parameters')
    trt.add_argument('--trt-min-opt-max-batch', nargs=3, type=int,
                     default=(1, 8, 16),
                     help='Torch-TensorRT min, optimal and max batch size')
    if model_name == 'HiFi-GAN':
        trt.add_argument('--trt-min-opt-max-hifigan-length', nargs=3, type=int,
                         default=(100, 800, 1200),
                         help='Torch-TensorRT min, optimal and max audio length (in frames)')
    return parser


def _convert_ts_to_trt_hifigan(ts_model, amp, trt_min_opt_max_batch,
                               trt_min_opt_max_hifigan_length, num_mels=80):
    import torch_tensorrt
    trt_dtype = torch.half if amp else torch.float
    print(f'Torch TensorRT: compiling HiFi-GAN for dtype {trt_dtype}.')
    min_shp, opt_shp, max_shp = zip(trt_min_opt_max_batch,
                                    (num_mels,) * 3,
                                    trt_min_opt_max_hifigan_length)
    compile_settings = {
        "inputs": [torch_tensorrt.Input(
            min_shape=min_shp,
            opt_shape=opt_shp,
            max_shape=max_shp,
            dtype=trt_dtype,
        )],
        "enabled_precisions": {trt_dtype},
        "require_full_compilation": True,
    }
    trt_model = torch_tensorrt.compile(ts_model, **compile_settings)
    print('Torch TensorRT: compilation successful.')
    return trt_model
