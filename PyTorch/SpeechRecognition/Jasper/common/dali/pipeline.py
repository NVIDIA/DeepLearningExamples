# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import nvidia.dali
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import multiprocessing
import numpy as np
import torch
import math
import itertools


class DaliPipeline(nvidia.dali.pipeline.Pipeline):
    def __init__(self, *,
                 train_pipeline: bool,  # True if train pipeline, False if validation pipeline
                 device_id,
                 num_threads,
                 batch_size,
                 file_root: str,
                 file_list: str,
                 sample_rate,
                 discrete_resample_range: bool,
                 resample_range: list,
                 window_size,
                 window_stride,
                 nfeatures,
                 nfft,
                 frame_splicing_factor,
                 dither_coeff,
                 silence_threshold,
                 preemph_coeff,
                 pad_align,
                 max_duration,
                 mask_time_num_regions,
                 mask_time_min,
                 mask_time_max,
                 mask_freq_num_regions,
                 mask_freq_min,
                 mask_freq_max,
                 mask_both_num_regions,
                 mask_both_min_time,
                 mask_both_max_time,
                 mask_both_min_freq,
                 mask_both_max_freq,
                 preprocessing_device="gpu"):
        super().__init__(batch_size, num_threads, device_id)

        self._dali_init_log(locals())

        if torch.distributed.is_initialized():
            shard_id = torch.distributed.get_rank()
            n_shards = torch.distributed.get_world_size()
        else:
            shard_id = 0
            n_shards = 1

        self.preprocessing_device = preprocessing_device.lower()
        assert self.preprocessing_device == "cpu" or self.preprocessing_device == "gpu", \
            "Incorrect preprocessing device. Please choose either 'cpu' or 'gpu'"
        self.frame_splicing_factor = frame_splicing_factor
        assert frame_splicing_factor == 1, "DALI doesn't support frame splicing operation"

        self.resample_range = resample_range
        self.discrete_resample_range = discrete_resample_range

        self.train = train_pipeline
        self.sample_rate = sample_rate
        self.dither_coeff = dither_coeff
        self.nfeatures = nfeatures
        self.max_duration = max_duration
        self.mask_params = {
            'time_num_regions': mask_time_num_regions,
            'time_min': mask_time_min,
            'time_max': mask_time_max,
            'freq_num_regions': mask_freq_num_regions,
            'freq_min': mask_freq_min,
            'freq_max': mask_freq_max,
            'both_num_regions': mask_both_num_regions,
            'both_min_time': mask_both_min_time,
            'both_max_time': mask_both_max_time,
            'both_min_freq': mask_both_min_freq,
            'both_max_freq': mask_both_max_freq,
        }
        self.do_remove_silence = True if silence_threshold is not None else False

        self.read = ops.FileReader(device="cpu", file_root=file_root, file_list=file_list, shard_id=shard_id,
                                   num_shards=n_shards, shuffle_after_epoch=train_pipeline)

        # TODO change ExternalSource to Uniform for new DALI release
        if discrete_resample_range and resample_range is not None:
            self.speed_perturbation_coeffs = ops.ExternalSource(device="cpu", cycle=True,
                                                                source=self._discrete_resample_coeffs_generator)
        elif resample_range is not None:
            self.speed_perturbation_coeffs = ops.Uniform(device="cpu", range=resample_range)
        else:
            self.speed_perturbation_coeffs = None

        self.decode = ops.AudioDecoder(device="cpu", sample_rate=self.sample_rate if resample_range is None else None,
                                       dtype=types.FLOAT, downmix=True)

        self.normal_distribution = ops.NormalDistribution(device=preprocessing_device)

        self.preemph = ops.PreemphasisFilter(device=preprocessing_device, preemph_coeff=preemph_coeff)

        self.spectrogram = ops.Spectrogram(device=preprocessing_device, nfft=nfft,
                                           window_length=window_size * sample_rate,
                                           window_step=window_stride * sample_rate)

        self.mel_fbank = ops.MelFilterBank(device=preprocessing_device, sample_rate=sample_rate, nfilter=self.nfeatures,
                                           normalize=True)

        self.log_features = ops.ToDecibels(device=preprocessing_device, multiplier=np.log(10), reference=1.0,
                                           cutoff_db=math.log(1e-20))

        self.get_shape = ops.Shapes(device=preprocessing_device)

        self.normalize = ops.Normalize(device=preprocessing_device, axes=[1])

        self.pad = ops.Pad(device=preprocessing_device, axes=[1], fill_value=0, align=pad_align)

        # Silence trimming
        self.get_nonsilent_region = ops.NonsilentRegion(device="cpu", cutoff_db=silence_threshold)
        self.trim_silence = ops.Slice(device="cpu", normalized_anchor=False, normalized_shape=False, axes=[0])
        self.to_float = ops.Cast(device="cpu", dtype=types.FLOAT)

        # Spectrogram masking
        self.spectrogram_cutouts = ops.ExternalSource(source=self._cutouts_generator, num_outputs=2, cycle=True)
        self.mask_spectrogram = ops.Erase(device=preprocessing_device, axes=[0, 1], fill_value=0,
                                          normalized_anchor=True)

    @classmethod
    def from_config(cls, train_pipeline: bool, device_id, batch_size, file_root: str, file_list: str, config_data: dict,
                    config_features: dict, device_type: str = "gpu", do_resampling: bool = True,
                    num_cpu_threads=multiprocessing.cpu_count()):

        max_duration = config_data['max_duration']
        sample_rate = config_data['sample_rate']
        silence_threshold = -60 if config_data['trim_silence'] else None

        # TODO Take into account resampling probablity
        # TODO     config_features['speed_perturbation']['p']

        if do_resampling and config_data['speed_perturbation'] is not None:
            resample_range = [config_data['speed_perturbation']['min_rate'],
                              config_data['speed_perturbation']['max_rate']]
            discrete_resample_range = config_data['speed_perturbation']['discrete']
        else:
            resample_range = None
            discrete_resample_range = False

        window_size = config_features['window_size']
        window_stride = config_features['window_stride']
        nfeatures = config_features['n_filt']
        nfft = config_features['n_fft']
        frame_splicing_factor = config_features['frame_splicing']
        dither_coeff = config_features['dither']
        pad_align = config_features['pad_align']
        pad_to_max_duration = config_features['pad_to_max_duration']
        assert not pad_to_max_duration, "Padding to max duration currently not supported in DALI"
        preemph_coeff = .97

        config_spec = config_features['spec_augment']
        if config_spec is not None:
            mask_time_num_regions = config_spec['time_masks']
            mask_time_min = config_spec['min_time']
            mask_time_max = config_spec['max_time']
            mask_freq_num_regions = config_spec['freq_masks']
            mask_freq_min = config_spec['min_freq']
            mask_freq_max = config_spec['max_freq']
        else:
            mask_time_num_regions = 0
            mask_time_min = 0
            mask_time_max = 0
            mask_freq_num_regions = 0
            mask_freq_min = 0
            mask_freq_max = 0

        config_cutout = config_features['cutout_augment']
        if config_cutout is not None:
            mask_both_num_regions = config_cutout['masks']
            mask_both_min_time = config_cutout['min_time']
            mask_both_max_time = config_cutout['max_time']
            mask_both_min_freq = config_cutout['min_freq']
            mask_both_max_freq = config_cutout['max_freq']
        else:
            mask_both_num_regions = 0
            mask_both_min_time = 0
            mask_both_max_time = 0
            mask_both_min_freq = 0
            mask_both_max_freq = 0

        return cls(train_pipeline=train_pipeline,
                   device_id=device_id,
                   preprocessing_device=device_type,
                   num_threads=num_cpu_threads,
                   batch_size=batch_size,
                   file_root=file_root,
                   file_list=file_list,
                   sample_rate=sample_rate,
                   discrete_resample_range=discrete_resample_range,
                   resample_range=resample_range,
                   window_size=window_size,
                   window_stride=window_stride,
                   nfeatures=nfeatures,
                   nfft=nfft,
                   frame_splicing_factor=frame_splicing_factor,
                   dither_coeff=dither_coeff,
                   silence_threshold=silence_threshold,
                   preemph_coeff=preemph_coeff,
                   pad_align=pad_align,
                   max_duration=max_duration,
                   mask_time_num_regions=mask_time_num_regions,
                   mask_time_min=mask_time_min,
                   mask_time_max=mask_time_max,
                   mask_freq_num_regions=mask_freq_num_regions,
                   mask_freq_min=mask_freq_min,
                   mask_freq_max=mask_freq_max,
                   mask_both_num_regions=mask_both_num_regions,
                   mask_both_min_time=mask_both_min_time,
                   mask_both_max_time=mask_both_max_time,
                   mask_both_min_freq=mask_both_min_freq,
                   mask_both_max_freq=mask_both_max_freq)

    @staticmethod
    def _dali_init_log(args: dict):
        if (not torch.distributed.is_initialized() or (
                torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)):  # print once
            max_len = max([len(ii) for ii in args.keys()])
            fmt_string = '\t%' + str(max_len) + 's : %s'
            print('Initializing DALI with parameters:')
            for keyPair in sorted(args.items()):
                print(fmt_string % keyPair)

    @staticmethod
    def _div_ceil(dividend, divisor):
        return (dividend + (divisor - 1)) // divisor

    def _get_audio_len(self, inp):
        return self.get_shape(inp) if self.frame_splicing_factor == 1 else \
            self._div_ceil(self.get_shape(inp), self.frame_splicing_factor)

    def _remove_silence(self, inp):
        begin, length = self.get_nonsilent_region(inp)
        out = self.trim_silence(inp, self.to_float(begin), self.to_float(length))
        return out

    def _do_spectrogram_masking(self):
        return self.mask_params['time_num_regions'] > 0 or self.mask_params['freq_num_regions'] > 0 or \
               self.mask_params['both_num_regions'] > 0

    @staticmethod
    def _interleave_lists(*lists):
        """
        [*, **, ***], [1, 2, 3], [a, b, c] -> [*, 1, a, **, 2, b, ***, 3, c]
        Returns:
             iterator over interleaved list
        """
        assert all((len(lists[0]) == len(test_l) for test_l in lists)), "All lists have to have the same length"
        return itertools.chain(*zip(*lists))

    def _generate_cutouts(self):
        """
        Returns:
            Generates anchors and shapes of the cutout regions.
            Single call generates one batch of data.
            The output shall be passed to DALI's Erase operator
            anchors = [f0 t0 f1 t1 ...]
            shapes = [f0w t0h f1w t1h ...]
        """
        MAX_TIME_DIMENSION = 20 * 16000
        freq_anchors = np.random.random(self.mask_params['freq_num_regions'])
        time_anchors = np.random.random(self.mask_params['time_num_regions'])
        both_anchors_freq = np.random.random(self.mask_params['both_num_regions'])
        both_anchors_time = np.random.random(self.mask_params['both_num_regions'])
        anchors = []
        for anch in freq_anchors:
            anchors.extend([anch, 0])
        for anch in time_anchors:
            anchors.extend([0, anch])
        for t, f in zip(both_anchors_time, both_anchors_freq):
            anchors.extend([f, t])

        shapes = []
        shapes.extend(
            self._interleave_lists(
                np.random.randint(self.mask_params['freq_min'], self.mask_params['freq_max'] + 1,
                                  self.mask_params['freq_num_regions']),
                # XXX: Here, a time dimension of the spectrogram shall be passed.
                #      However, in DALI ArgumentInput can't come from GPU.
                #      So we leave the job for Erase (masking operator) to get it together.
                [int(MAX_TIME_DIMENSION)] * self.mask_params['freq_num_regions']
            )
        )
        shapes.extend(
            self._interleave_lists(
                [self.nfeatures] * self.mask_params['time_num_regions'],
                np.random.randint(self.mask_params['time_min'], self.mask_params['time_max'] + 1,
                                  self.mask_params['time_num_regions'])
            )
        )
        shapes.extend(
            self._interleave_lists(
                np.random.randint(self.mask_params['both_min_freq'], self.mask_params['both_max_freq'] + 1,
                                  self.mask_params['both_num_regions']),
                np.random.randint(self.mask_params['both_min_time'], self.mask_params['both_max_time'] + 1,
                                  self.mask_params['both_num_regions'])
            )
        )
        return anchors, shapes

    def _discrete_resample_coeffs_generator(self):
        """
        Generate resample coeffs from discrete set
        """
        yield np.random.choice([self.resample_range[0], 1.0, self.resample_range[1]],
                               size=self.batch_size).astype('float32')

    def _cutouts_generator(self):
        """
        Generator, that wraps cutouts creation in order to randomize inputs
        and allow passing them to DALI's ExternalSource operator
        """

        def tuples2list(tuples: list):
            """
            [(a, b), (c, d)] -> [[a, c], [b, d]]
            """
            return map(list, zip(*tuples))

        [anchors, shapes] = tuples2list([self._generate_cutouts() for _ in range(self.batch_size)])
        yield np.array(anchors, dtype=np.float32), np.array(shapes, dtype=np.float32)

    def define_graph(self):
        audio, label = self.read()
        if not self.train or self.speed_perturbation_coeffs is None:
            audio, sr = self.decode(audio)
        else:
            resample_coeffs = self.speed_perturbation_coeffs() * self.sample_rate
            audio, sr = self.decode(audio, sample_rate=resample_coeffs)

        if self.do_remove_silence:
            audio = self._remove_silence(audio)

        # Max duration drop is performed at DataLayer stage

        if self.preprocessing_device == "gpu":
            audio = audio.gpu()

        if self.dither_coeff != 0.:
            audio = audio + self.normal_distribution(audio) * self.dither_coeff

        audio = self.preemph(audio)

        audio = self.spectrogram(audio)
        audio = self.mel_fbank(audio)
        audio = self.log_features(audio)

        audio_len = self._get_audio_len(audio)

        audio = self.normalize(audio)
        audio = self.pad(audio)

        if self.train and self._do_spectrogram_masking():
            anchors, shapes = self.spectrogram_cutouts()
            audio = self.mask_spectrogram(audio, anchor=anchors, shape=shapes)

        # When modifying DALI pipeline returns, make sure you update `output_map` in DALIGenericIterator invocation
        return audio.gpu(), label.gpu(), audio_len.gpu()


class DaliTritonPipeline(DaliPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not kwargs['train_pipeline'], "Pipeline for Triton shall be a validation pipeline"
        if torch.distributed.is_initialized():
            raise RuntimeError(
                "You're creating Triton pipeline, using multi-process mode. Please use single-process mode.")
        self.read = ops.ExternalSource(name="DALI_INPUT_0", no_copy=True, device="cpu")


def serialize_dali_triton_pipeline(output_path: str, config_data: dict, config_features: dict):
    pipe = DaliTritonPipeline.from_config(train_pipeline=False, device_id=-1, batch_size=-1, file_root=None,
                                          file_list=None, config_data=config_data, config_features=config_features,
                                          do_resampling=False, num_cpu_threads=-1)
    pipe.serialize(filename=output_path)
