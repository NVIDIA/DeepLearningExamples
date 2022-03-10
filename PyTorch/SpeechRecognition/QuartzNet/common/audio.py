# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import random
import soundfile as sf

import librosa
import torch
import numpy as np

import sox


def audio_from_file(file_path, offset=0, duration=0, trim=False, target_sr=16000):
    audio = AudioSegment(file_path, target_sr=target_sr, int_values=False,
                         offset=offset, duration=duration, trim=trim)

    samples = torch.tensor(audio.samples, dtype=torch.float).cuda()
    num_samples = torch.tensor(samples.shape[0]).int().cuda()
    return (samples.unsqueeze(0), num_samples.unsqueeze(0))


class AudioSegment(object):
    """Monaural audio segment abstraction.

    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(self, filename, target_sr=None, int_values=False, offset=0,
                 duration=0, trim=False, trim_db=60):
        """Create audio segment from samples.

        Samples are converted to float32 internally, with int scaled to [-1, 1].
        Load a file supported by librosa and return as an AudioSegment.
        :param filename: path of file to load
        :param target_sr: the desired sample rate
        :param int_values: if true, load samples as 32-bit integers
        :param offset: offset in seconds when loading audio
        :param duration: duration in seconds when loading audio
        :return: numpy array of samples
        """
        with sf.SoundFile(filename, 'r') as f:
            dtype = 'int32' if int_values else 'float32'
            sample_rate = f.samplerate
            if offset > 0:
                f.seek(int(offset * sample_rate))
            if duration > 0:
                samples = f.read(int(duration * sample_rate), dtype=dtype)
            else:
                samples = f.read(dtype=dtype)
        samples = samples.transpose()

        samples = self._convert_samples_to_float32(samples)
        if target_sr is not None and target_sr != sample_rate:
            samples = librosa.resample(samples, orig_sr=sample_rate,
                                       target_sr=target_sr)
            sample_rate = target_sr
        if trim:
            samples, _ = librosa.effects.trim(samples, top_db=trim_db)
        self._samples = samples
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

    def __eq__(self, other):
        """Return whether two objects are equal."""
        if type(other) is not type(self):
            return False
        if self._sample_rate != other._sample_rate:
            return False
        if self._samples.shape != other._samples.shape:
            return False
        if np.any(self.samples != other._samples):
            return False
        return True

    def __ne__(self, other):
        """Return whether two objects are unequal."""
        return not self.__eq__(other)

    def __str__(self):
        """Return human-readable representation of segment."""
        return ("%s: num_samples=%d, sample_rate=%d, duration=%.2fsec, "
                        "rms=%.2fdB" % (type(self), self.num_samples, self.sample_rate,
                                                        self.duration, self.rms_db))

    @staticmethod
    def _convert_samples_to_float32(samples):
        """Convert sample type to float32.

        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= (1. / 2 ** (bits - 1))
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def num_samples(self):
        return self._samples.shape[0]

    @property
    def duration(self):
        return self._samples.shape[0] / float(self._sample_rate)

    @property
    def rms_db(self):
        mean_square = np.mean(self._samples ** 2)
        return 10 * np.log10(mean_square)

    def gain_db(self, gain):
        self._samples *= 10. ** (gain / 20.)

    def pad(self, pad_size, symmetric=False):
        """Add zero padding to the sample.

        The pad size is given in number of samples. If symmetric=True,
        `pad_size` will be added to both sides. If false, `pad_size` zeros
        will be added only to the end.
        """
        self._samples = np.pad(self._samples,
                               (pad_size if symmetric else 0, pad_size),
                               mode='constant')

    def subsegment(self, start_time=None, end_time=None):
        """Cut the AudioSegment between given boundaries.

        Note that this is an in-place transformation.
        :param start_time: Beginning of subsegment in seconds.
        :type start_time: float
        :param end_time: End of subsegment in seconds.
        :type end_time: float
        :raise ValueError: If start_time or end_time is incorrectly set, e.g. out
                                             of bounds in time.
        """
        start_time = 0.0 if start_time is None else start_time
        end_time = self.duration if end_time is None else end_time
        if start_time < 0.0:
            start_time = self.duration + start_time
        if end_time < 0.0:
            end_time = self.duration + end_time
        if start_time < 0.0:
            raise ValueError("The slice start position (%f s) is out of "
                             "bounds." % start_time)
        if end_time < 0.0:
            raise ValueError("The slice end position (%f s) is out of bounds." %
                             end_time)
        if start_time > end_time:
            raise ValueError("The slice start position (%f s) is later than "
                             "the end position (%f s)." % (start_time, end_time))
        if end_time > self.duration:
            raise ValueError("The slice end position (%f s) is out of bounds "
                             "(> %f s)" % (end_time, self.duration))
        start_sample = int(round(start_time * self._sample_rate))
        end_sample = int(round(end_time * self._sample_rate))
        self._samples = self._samples[start_sample:end_sample]


class Perturbation:
    def __init__(self, p=0.1, rng=None):
        self.p = p
        self._rng = random.Random() if rng is None else rng

    def maybe_apply(self, segment, sample_rate=None):
        if self._rng.random() < self.p:
            self(segment, sample_rate)


class SpeedPerturbation(Perturbation):
    def __init__(self, min_rate=0.85, max_rate=1.15, discrete=False, p=0.1, rng=None):
        super(SpeedPerturbation, self).__init__(p, rng)
        assert 0 < min_rate < max_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.discrete = discrete

    def __call__(self, data, sample_rate):
        if self.discrete:
            rate = np.random.choice([self.min_rate, None, self.max_rate])
        else:
            rate = self._rng.uniform(self.min_rate, self.max_rate)

        if rate is not None:
            data._samples = sox.Transformer().speed(factor=rate).build_array(
                input_array=data._samples, sample_rate_in=sample_rate)


class GainPerturbation(Perturbation):
    def __init__(self, min_gain_dbfs=-10, max_gain_dbfs=10, p=0.1, rng=None):
        super(GainPerturbation, self).__init__(p, rng)
        self._rng = random.Random() if rng is None else rng
        self._min_gain_dbfs = min_gain_dbfs
        self._max_gain_dbfs = max_gain_dbfs

    def __call__(self, data, sample_rate=None):
        del sample_rate
        gain = self._rng.uniform(self._min_gain_dbfs, self._max_gain_dbfs)
        data._samples = data._samples * (10. ** (gain / 20.))


class ShiftPerturbation(Perturbation):
    def __init__(self, min_shift_ms=-5.0, max_shift_ms=5.0, p=0.1, rng=None):
        super(ShiftPerturbation, self).__init__(p, rng)
        self._min_shift_ms = min_shift_ms
        self._max_shift_ms = max_shift_ms

    def __call__(self, data, sample_rate):
        shift_ms = self._rng.uniform(self._min_shift_ms, self._max_shift_ms)
        if abs(shift_ms) / 1000 > data.duration:
            # TODO: do something smarter than just ignore this condition
            return
        shift_samples = int(shift_ms * data.sample_rate // 1000)
        # print("DEBUG: shift:", shift_samples)
        if shift_samples < 0:
            data._samples[-shift_samples:] = data._samples[:shift_samples]
            data._samples[:-shift_samples] = 0
        elif shift_samples > 0:
            data._samples[:-shift_samples] = data._samples[shift_samples:]
            data._samples[-shift_samples:] = 0
