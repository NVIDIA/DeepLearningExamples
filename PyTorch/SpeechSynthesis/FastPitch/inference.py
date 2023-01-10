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
import itertools
import sys
import time
import warnings
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
from scipy.stats import norm
from scipy.io.wavfile import write
from torch.nn.functional import l1_loss
from torch.nn.utils.rnn import pad_sequence

import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

import models
from common import gpu_affinity
from common.tb_dllogger import (init_inference_metadata, stdout_metric_format,
                                unique_log_fpath)
from common.text import cmudict
from common.text.text_processing import get_text_processing
from common.utils import l2_promote
from fastpitch.pitch_transform import pitch_transform_custom
from hifigan.data_function import MAX_WAV_VALUE, mel_spectrogram
from hifigan.models import Denoiser
from waveglow import model as glow


CHECKPOINT_SPECIFIC_ARGS = [
    'sampling_rate', 'hop_length', 'win_length', 'p_arpabet', 'text_cleaners',
    'symbol_set', 'max_wav_value', 'prepend_space_to_text',
    'append_space_to_text']


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Full path to the input text (phareses separated by newlines)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output folder to save audio (file per phrase)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')
    parser.add_argument('--save-mels', action='store_true',
                        help='Save generator outputs to disk')
    parser.add_argument('--cuda', action='store_true',
                        help='Run inference on a GPU using CUDA')
    parser.add_argument('--cudnn-benchmark', action='store_true',
                        help='Enable cudnn benchmark mode')
    parser.add_argument('--l2-promote', action='store_true',
                        help='Increase max fetch granularity of GPU L2 cache')
    parser.add_argument('--fastpitch', type=str, default=None, required=False,
                        help='Full path to the spectrogram generator .pt file '
                             '(skip to synthesize from ground truth mels)')
    parser.add_argument('--waveglow', type=str, default=None, required=False,
                        help='Full path to a WaveGlow model .pt file')
    parser.add_argument('-s', '--waveglow-sigma-infer', default=0.9, type=float,
                        help='WaveGlow sigma')
    parser.add_argument('--hifigan', type=str, default=None, required=False,
                        help='Full path to a HiFi-GAN model .pt file')
    parser.add_argument('-d', '--denoising-strength', default=0.0, type=float,
                        help='Capture and subtract model bias to enhance audio')
    parser.add_argument('--hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--win-length', type=int, default=1024,
                        help='STFT win length for denoiser and mel loss')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        choices=[22050, 44100], help='Sampling rate')
    parser.add_argument('--max_wav_value', default=32768.0, type=float,
                        help='Maximum audiowave value')
    parser.add_argument('--amp', action='store_true',
                        help='Inference with AMP')
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='Warmup iterations before measuring performance')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Repeat inference for benchmarking')
    parser.add_argument('--torchscript', action='store_true',
                        help='Run inference with TorchScript model (convert to TS if needed)')
    parser.add_argument('--checkpoint-format', type=str,
                        choices=['pyt', 'ts'], default='pyt',
                        help='Input checkpoint format (PyT or TorchScript)')
    parser.add_argument('--torch-tensorrt', action='store_true',
                        help='Run inference with Torch-TensorRT model (compile beforehand)')
    parser.add_argument('--report-mel-loss', action='store_true',
                        help='Report mel loss in metrics')
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA averaged model (if saved in checkpoints)')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset (for loading extra data fields)')
    parser.add_argument('--speaker', type=int, default=0,
                        help='Speaker ID for a multi-speaker model')

    parser.add_argument('--affinity', type=str, default='single',
                        choices=['socket', 'single', 'single_unique',
                                 'socket_unique_interleaved',
                                 'socket_unique_continuous',
                                 'disabled'],
                        help='type of CPU affinity')

    transf = parser.add_argument_group('transform')
    transf.add_argument('--fade-out', type=int, default=10,
                        help='Number of fadeout frames at the end')
    transf.add_argument('--pace', type=float, default=1.0,
                        help='Adjust the pace of speech')
    transf.add_argument('--pitch-transform-flatten', action='store_true',
                        help='Flatten the pitch')
    transf.add_argument('--pitch-transform-invert', action='store_true',
                        help='Invert the pitch wrt mean value')
    transf.add_argument('--pitch-transform-amplify', type=float, default=1.0,
                        help='Multiplicative amplification of pitch variability. '
                             'Typical values are in the range (1.0, 3.0).')
    transf.add_argument('--pitch-transform-shift', type=float, default=0.0,
                        help='Raise/lower the pitch by <hz>')
    transf.add_argument('--pitch-transform-custom', action='store_true',
                        help='Apply the transform from pitch_transform.py')

    txt = parser.add_argument_group('Text processing parameters')
    txt.add_argument('--text-cleaners', type=str, nargs='*',
                     default=['english_cleaners_v2'],
                     help='Type of text cleaners for input text')
    txt.add_argument('--symbol-set', type=str, default='english_basic',
                     help='Define symbol set for input text')
    txt.add_argument('--p-arpabet', type=float, default=0.0, help='')
    txt.add_argument('--heteronyms-path', type=str,
                     default='cmudict/heteronyms', help='')
    txt.add_argument('--cmudict-path', type=str,
                     default='cmudict/cmudict-0.7b', help='')
    return parser


def load_fields(fpath):
    lines = [l.strip() for l in open(fpath, encoding='utf-8')]
    if fpath.endswith('.tsv'):
        columns = lines[0].split('\t')
        fields = list(zip(*[t.split('\t') for t in lines[1:]]))
    else:
        columns = ['text']
        fields = [lines]
    return {c: f for c, f in zip(columns, fields)}


def prepare_input_sequence(fields, device, symbol_set, text_cleaners,
                           batch_size=128, dataset=None, load_mels=False,
                           load_pitch=False, p_arpabet=0.0):
    tp = get_text_processing(symbol_set, text_cleaners, p_arpabet)

    fields['text'] = [torch.LongTensor(tp.encode_text(text))
                      for text in fields['text']]
    order = np.argsort([-t.size(0) for t in fields['text']])

    fields['text'] = [fields['text'][i] for i in order]
    fields['text_lens'] = torch.LongTensor([t.size(0) for t in fields['text']])

    for t in fields['text']:
        print(tp.sequence_to_text(t.numpy()))

    if load_mels:
        assert 'mel' in fields
        assert dataset is not None
        fields['mel'] = [
            torch.load(Path(dataset, fields['mel'][i])).t() for i in order]
        fields['mel_lens'] = torch.LongTensor([t.size(0) for t in fields['mel']])

    if load_pitch:
        assert 'pitch' in fields
        fields['pitch'] = [
            torch.load(Path(dataset, fields['pitch'][i])) for i in order]
        fields['pitch_lens'] = torch.LongTensor([t.size(0) for t in fields['pitch']])

    if 'output' in fields:
        fields['output'] = [fields['output'][i] for i in order]

    # cut into batches & pad
    batches = []
    for b in range(0, len(order), batch_size):
        batch = {f: values[b:b+batch_size] for f, values in fields.items()}
        for f in batch:
            if f == 'text':
                batch[f] = pad_sequence(batch[f], batch_first=True)
            elif f == 'mel' and load_mels:
                batch[f] = pad_sequence(batch[f], batch_first=True).permute(0, 2, 1)
            elif f == 'pitch' and load_pitch:
                batch[f] = pad_sequence(batch[f], batch_first=True)

            if type(batch[f]) is torch.Tensor:
                batch[f] = batch[f].to(device)
        batches.append(batch)

    return batches


def build_pitch_transformation(args):
    if args.pitch_transform_custom:
        def custom_(pitch, pitch_lens, mean, std):
            return (pitch_transform_custom(pitch * std + mean, pitch_lens)
                    - mean) / std
        return custom_

    fun = 'pitch'
    if args.pitch_transform_flatten:
        fun = f'({fun}) * 0.0'
    if args.pitch_transform_invert:
        fun = f'({fun}) * -1.0'
    if args.pitch_transform_amplify != 1.0:
        ampl = args.pitch_transform_amplify
        fun = f'({fun}) * {ampl}'
    if args.pitch_transform_shift != 0.0:
        hz = args.pitch_transform_shift
        fun = f'({fun}) + {hz} / std'

    if fun == 'pitch':
        return None

    return eval(f'lambda pitch, pitch_lens, mean, std: {fun}')


def setup_mel_loss_reporting(args, voc_train_setup):
    if args.denoising_strength > 0.0:
        print('WARNING: denoising will be included in vocoder mel loss')
    num_mels = voc_train_setup.get('num_mels', 80)
    fmin = voc_train_setup.get('mel_fmin', 0)
    fmax = voc_train_setup.get('mel_fmax', 8000)  # not mel_fmax_loss

    def compute_audio_mel_loss(gen_audios, gt_mels, mel_lens):
        gen_audios /= MAX_WAV_VALUE
        total_loss = 0
        for gen_audio, gt_mel, mel_len in zip(gen_audios, gt_mels, mel_lens):
            mel_len = mel_len.item()
            gen_audio = gen_audio[None, :mel_len * args.hop_length]
            gen_mel = mel_spectrogram(gen_audio, args.win_length, num_mels,
                                      args.sampling_rate, args.hop_length,
                                      args.win_length, fmin, fmax)[0]
            total_loss += l1_loss(gen_mel, gt_mel[:, :mel_len])
        return total_loss.item()

    return compute_audio_mel_loss


def compute_mel_loss(mels, lens, gt_mels, gt_lens):
    total_loss = 0
    for mel, len_, gt_mel, gt_len in zip(mels, lens, gt_mels, gt_lens):
        min_len = min(len_, gt_len)
        total_loss += l1_loss(gt_mel[:, :min_len], mel[:, :min_len])
    return total_loss.item()


class MeasureTime(list):
    def __init__(self, *args, cuda=True, **kwargs):
        super(MeasureTime, self).__init__(*args, **kwargs)
        self.cuda = cuda

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.cuda:
            torch.cuda.synchronize()
        self.append(time.time() - self.t0)

    def __add__(self, other):
        assert len(self) == len(other)
        return MeasureTime((sum(ab) for ab in zip(self, other)), cuda=self.cuda)


def main():
    """
    Launches text-to-speech inference on a single GPU.
    """
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()

    if args.affinity != 'disabled':
        nproc_per_node = torch.cuda.device_count()
        # print(nproc_per_node)
        affinity = gpu_affinity.set_affinity(
            0,
            nproc_per_node,
            args.affinity
        )
        print(f'Thread affinity: {affinity}')

    if args.l2_promote:
        l2_promote()
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.output is not None:
        Path(args.output).mkdir(parents=False, exist_ok=True)

    log_fpath = args.log_file or str(Path(args.output, 'nvlog_infer.json'))
    DLLogger.init(backends=[
        JSONStreamBackend(Verbosity.DEFAULT, log_fpath, append=True),
        JSONStreamBackend(Verbosity.DEFAULT, unique_log_fpath(log_fpath)),
        StdOutBackend(Verbosity.VERBOSE, metric_format=stdout_metric_format)
    ])
    init_inference_metadata(args.batch_size)
    [DLLogger.log("PARAMETER", {k: v}) for k, v in vars(args).items()]

    device = torch.device('cuda' if args.cuda else 'cpu')

    gen_train_setup = {}
    voc_train_setup = {}
    generator = None
    vocoder = None
    denoiser = None

    is_ts_based_infer = args.torch_tensorrt or args.torchscript

    assert args.checkpoint_format == 'pyt' or is_ts_based_infer, \
        'TorchScript checkpoint can be used only for TS or Torch-TRT' \
        ' inference. Please set --torchscript or --torch-tensorrt flag.'

    assert args.waveglow is None or args.hifigan is None, \
        "Specify a single vocoder model"

    def _load_pyt_or_ts_model(model_name, ckpt_path):
        if args.checkpoint_format == 'ts':
            model = models.load_and_setup_ts_model(model_name, ckpt_path,
                                                   args.amp, device)
            model_train_setup = {}
            return model, model_train_setup
        model, _, model_train_setup = models.load_and_setup_model(
            model_name, parser, ckpt_path, args.amp, device,
            unk_args=unk_args, forward_is_infer=True, jitable=is_ts_based_infer)

        if is_ts_based_infer:
            model = torch.jit.script(model)
        return model, model_train_setup

    if args.fastpitch is not None:
        gen_name = 'fastpitch'
        generator, gen_train_setup = _load_pyt_or_ts_model('FastPitch',
                                                           args.fastpitch)

    if args.waveglow is not None:
        voc_name = 'waveglow'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vocoder, _, voc_train_setup = models.load_and_setup_model(
                'WaveGlow', parser, args.waveglow, args.amp, device,
                unk_args=unk_args, forward_is_infer=True, jitable=False)

        if args.denoising_strength > 0.0:
            denoiser = Denoiser(vocoder, sigma=0.0,
                                win_length=args.win_length).to(device)

        # if args.torchscript:
        #     vocoder = torch.jit.script(vocoder)

        def generate_audio(mel):
            audios = vocoder(mel, sigma=args.waveglow_sigma_infer)
            if denoiser is not None:
                audios = denoiser(audios.float(), args.denoising_strength).squeeze(1)
            return audios

    elif args.hifigan is not None:
        voc_name = 'hifigan'
        vocoder, voc_train_setup = _load_pyt_or_ts_model('HiFi-GAN',
                                                         args.hifigan)

        if args.denoising_strength > 0.0:
            denoiser = Denoiser(vocoder, win_length=args.win_length).to(device)

        if args.torch_tensorrt:
            vocoder = models.convert_ts_to_trt('HiFi-GAN', vocoder, parser,
                                               args.amp, unk_args)

        def generate_audio(mel):
            audios = vocoder(mel).float()
            if denoiser is not None:
                audios = denoiser(audios.squeeze(1), args.denoising_strength)
            return audios.squeeze(1) * args.max_wav_value

    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    for k in CHECKPOINT_SPECIFIC_ARGS:

        v1 = gen_train_setup.get(k, None)
        v2 = voc_train_setup.get(k, None)

        assert v1 is None or v2 is None or v1 == v2, \
            f'{k} mismatch in spectrogram generator and vocoder'

        val = v1 or v2
        if val and getattr(args, k) != val:
            src = 'generator' if v2 is None else 'vocoder'
            print(f'Overwriting args.{k}={getattr(args, k)} with {val} '
                  f'from {src} checkpoint.')
            setattr(args, k, val)

    gen_kw = {'pace': args.pace,
              'speaker': args.speaker,
              'pitch_tgt': None,
              'pitch_transform': build_pitch_transformation(args)}

    if is_ts_based_infer and generator is not None:
        gen_kw.pop('pitch_transform')
        print('Note: --pitch-transform-* args are disabled with TorchScript. '
              'To condition on pitch, pass pitch_tgt as input.')

    if args.p_arpabet > 0.0:
        cmudict.initialize(args.cmudict_path, args.heteronyms_path)

    if args.report_mel_loss:
        mel_loss_fn = setup_mel_loss_reporting(args, voc_train_setup)

    fields = load_fields(args.input)
    batches = prepare_input_sequence(
        fields, device, args.symbol_set, args.text_cleaners, args.batch_size,
        args.dataset_path, load_mels=(generator is None or args.report_mel_loss),
        p_arpabet=args.p_arpabet)

    cycle = itertools.cycle(batches)
    # Use real data rather than synthetic - FastPitch predicts len
    for _ in tqdm(range(args.warmup_steps), 'Warmup'):
        with torch.no_grad():
            b = next(cycle)
            if generator is not None:
                mel, *_ = generator(b['text'])
            else:
                mel, mel_lens = b['mel'], b['mel_lens']
                if args.amp:
                    mel = mel.half()
            if vocoder is not None:
                audios = generate_audio(mel)

    gen_measures = MeasureTime(cuda=args.cuda)
    vocoder_measures = MeasureTime(cuda=args.cuda)

    all_utterances = 0
    all_samples = 0
    all_batches = 0
    all_letters = 0
    all_frames = 0
    gen_mel_loss_sum = 0
    voc_mel_loss_sum = 0

    reps = args.repeats
    log_enabled = reps == 1
    log = lambda s, d: DLLogger.log(step=s, data=d) if log_enabled else None

    for rep in (tqdm(range(reps), 'Inference') if reps > 1 else range(reps)):
        for b in batches:

            if generator is None:
                mel, mel_lens = b['mel'], b['mel_lens']
                if args.amp:
                    mel = mel.half()
            else:
                with torch.no_grad(), gen_measures:
                    mel, mel_lens, *_ = generator(b['text'], **gen_kw)

                if args.report_mel_loss:
                    gen_mel_loss_sum += compute_mel_loss(
                        mel, mel_lens, b['mel'], b['mel_lens'])

                gen_infer_perf = mel.size(0) * mel.size(2) / gen_measures[-1]
                all_letters += b['text_lens'].sum().item()
                all_frames += mel.size(0) * mel.size(2)
                log(rep, {f"{gen_name}_frames/s": gen_infer_perf})
                log(rep, {f"{gen_name}_latency": gen_measures[-1]})

                if args.save_mels:
                    for i, mel_ in enumerate(mel):
                        m = mel_[:, :mel_lens[i].item()].permute(1, 0)
                        fname = b['output'][i] if 'output' in b else f'mel_{i}.npy'
                        mel_path = Path(args.output, Path(fname).stem + '.npy')
                        np.save(mel_path, m.cpu().numpy())

            if vocoder is not None:
                with torch.no_grad(), vocoder_measures:
                    audios = generate_audio(mel)

                vocoder_infer_perf = (
                    audios.size(0) * audios.size(1) / vocoder_measures[-1])

                log(rep, {f"{voc_name}_samples/s": vocoder_infer_perf})
                log(rep, {f"{voc_name}_latency": vocoder_measures[-1]})

                if args.report_mel_loss:
                    voc_mel_loss_sum += mel_loss_fn(audios, mel, mel_lens)

                if args.output is not None and reps == 1:
                    for i, audio in enumerate(audios):
                        audio = audio[:mel_lens[i].item() * args.hop_length]

                        if args.fade_out:
                            fade_len = args.fade_out * args.hop_length
                            fade_w = torch.linspace(1.0, 0.0, fade_len)
                            audio[-fade_len:] *= fade_w.to(audio.device)

                        audio = audio / torch.max(torch.abs(audio))
                        fname = b['output'][i] if 'output' in b else f'audio_{all_utterances + i}.wav'
                        audio_path = Path(args.output, fname)
                        write(audio_path, args.sampling_rate, audio.cpu().numpy())

                if generator is not None:
                    log(rep, {"latency": (gen_measures[-1] + vocoder_measures[-1])})

            all_utterances += mel.size(0)
            all_samples += mel_lens.sum().item() * args.hop_length
            all_batches += 1

    log_enabled = True
    if generator is not None:
        gm = np.sort(np.asarray(gen_measures))
        rtf = all_samples / (all_utterances * gm.mean() * args.sampling_rate)
        rtf_at = all_samples / (all_batches * gm.mean() * args.sampling_rate)
        log((), {f"avg_{gen_name}_tokens/s": all_letters / gm.sum()})
        log((), {f"avg_{gen_name}_frames/s": all_frames / gm.sum()})
        log((), {f"avg_{gen_name}_latency": gm.mean()})
        log((), {f"avg_{gen_name}_RTF": rtf})
        log((), {f"avg_{gen_name}_RTF@{args.batch_size}": rtf_at})
        log((), {f"90%_{gen_name}_latency": gm.mean() + norm.ppf((1.0 + 0.90) / 2) * gm.std()})
        log((), {f"95%_{gen_name}_latency": gm.mean() + norm.ppf((1.0 + 0.95) / 2) * gm.std()})
        log((), {f"99%_{gen_name}_latency": gm.mean() + norm.ppf((1.0 + 0.99) / 2) * gm.std()})
        if args.report_mel_loss:
            log((), {f"avg_{gen_name}_mel-loss": gen_mel_loss_sum / all_utterances})
    if vocoder is not None:
        vm = np.sort(np.asarray(vocoder_measures))
        rtf = all_samples / (all_utterances * vm.mean() * args.sampling_rate)
        rtf_at = all_samples / (all_batches * vm.mean() * args.sampling_rate)
        log((), {f"avg_{voc_name}_samples/s": all_samples / vm.sum()})
        log((), {f"avg_{voc_name}_latency": vm.mean()})
        log((), {f"avg_{voc_name}_RTF": rtf})
        log((), {f"avg_{voc_name}_RTF@{args.batch_size}": rtf_at})
        log((), {f"90%_{voc_name}_latency": vm.mean() + norm.ppf((1.0 + 0.90) / 2) * vm.std()})
        log((), {f"95%_{voc_name}_latency": vm.mean() + norm.ppf((1.0 + 0.95) / 2) * vm.std()})
        log((), {f"99%_{voc_name}_latency": vm.mean() + norm.ppf((1.0 + 0.99) / 2) * vm.std()})
        if args.report_mel_loss:
            log((), {f"avg_{voc_name}_mel-loss": voc_mel_loss_sum / all_utterances})
    if generator is not None and vocoder is not None:
        m = gm + vm
        rtf = all_samples / (all_utterances * m.mean() * args.sampling_rate)
        rtf_at = all_samples / (all_batches * m.mean() * args.sampling_rate)
        log((), {"avg_samples/s": all_samples / m.sum()})
        log((), {"avg_letters/s": all_letters / m.sum()})
        log((), {"avg_latency": m.mean()})
        log((), {"avg_RTF": rtf})
        log((), {f"avg_RTF@{args.batch_size}": rtf_at})
        log((), {"90%_latency": m.mean() + norm.ppf((1.0 + 0.90) / 2) * m.std()})
        log((), {"95%_latency": m.mean() + norm.ppf((1.0 + 0.95) / 2) * m.std()})
        log((), {"99%_latency": m.mean() + norm.ppf((1.0 + 0.99) / 2) * m.std()})
    DLLogger.flush()


if __name__ == '__main__':
    main()
