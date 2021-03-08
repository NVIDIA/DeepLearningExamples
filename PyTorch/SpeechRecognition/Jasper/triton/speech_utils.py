#!/usr/bin/python

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


import soundfile as sf
import math
from os import system
import numpy as np
import tritonclient.grpc, tritonclient.http
import tritonclient.grpc.model_config_pb2 as model_config
from tritonclient.utils import triton_to_np_dtype, np_to_triton_dtype
import grpc
import sys
import os
if "./triton" not in sys.path:
    sys.path.append(os.path.join(sys.path[0], "../"))
from common.text import _clean_text

WINDOWS_FNS = {"hanning": np.hanning, "hamming": np.hamming, "none": None}

triton_type_to_np_dtype = {
    'TYPE_BOOL': np.bool,
    'TYPE_INT8': np.int8,
    'TYPE_INT16': np.int16,
    'TYPE_INT32': np.int32,
    'TYPE_INT64': np.int64,
    'TYPE_UINT8': np.uint8,
    'TYPE_FP16': np.float16,
    'TYPE_FP32': np.float32,
    'TYPE_FP64': np.float64
}

model_dtype_to_np_dtype = {
    "BOOL": np.bool,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "FP16": np.float16,
    "FP32": np.float32,
    "FP64": np.float64,
    "BYTES": np.dtype(object)
}

def load_transcript(transcript_path):
    with open(transcript_path, 'r', encoding="utf-8") as transcript_file:
        transcript = transcript_file.read().replace('\n', '')
    return transcript

def parse_transcript(transcript, labels_map, blank_index):
    chars = [labels_map.get(x, blank_index) for x in list(transcript)]
    transcript = list(filter(lambda x: x != blank_index, chars))
    return transcript

def normalize_string(s, labels, table, **unused_kwargs):
    """
    Normalizes string. For example:
    'call me at 8:00 pm!' -> 'call me at eight zero pm'

    Args:
        s: string to normalize
        labels: labels used during model training.

    Returns:
            Normalized string
    """

    def good_token(token, labels):
        s = set(labels)
        for t in token:
            if not t in s:
                return False
        return True

    try:
        text = _clean_text(s, ["english_cleaners"], table).strip()
        return ''.join([t for t in text if good_token(t, labels=labels)])
    except:
        print("WARNING: Normalizing {} failed".format(s))
        return None

def ctc_decoder_predictions_tensor(prediction_cpu_tensor, batch_size, labels):
    """
    Takes output of greedy ctc decoder and performs ctc decoding algorithm to
    remove duplicates and special symbol. Returns prediction
    Args:
        tensor: model output tensor
        label: A list of labels
    Returns:
        prediction
    """
    blank_id = len(labels) - 1
    hypotheses = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    # iterate over batch
    prediction_cpu_tensor = prediction_cpu_tensor.reshape((batch_size, int(prediction_cpu_tensor.size/batch_size)))
    for ind in range(batch_size):
        prediction = prediction_cpu_tensor[ind].tolist()
        # CTC decoding procedure
        decoded_prediction = []
        previous = len(labels) - 1 # id of a blank symbol
        for p in prediction:
            if (p != previous or previous == blank_id) and p != blank_id:
                decoded_prediction.append(p)
            previous = p
        hypothesis = ''.join([labels_map[c] for c in decoded_prediction])
        hypotheses.append(hypothesis)
    return hypotheses

class SpeechClient(object):

    def __init__(self, url, protocol, model_name, model_version, batch_size,
                 model_platform=None, verbose=False,
                 mode="batch",
                 from_features=True):

        self.model_name = model_name
        self.model_version = model_version
        self.verbose = verbose
        self.batch_size = batch_size
        self.transpose_audio_features = False
        self.grpc_stub = None
        self.ctx = None
        self.correlation_id = 0
        self.first_run = True
        if mode == "streaming" or mode == "asynchronous":
            self.correlation_id = 1

        self.buffer = []

        if protocol == "grpc":
            # Create gRPC client for communicating with the server
            self.prtcl_client = tritonclient.grpc
        else:
            # Create HTTP client for communicating with the server
            self.prtcl_client = tritonclient.http

        self.triton_client = self.prtcl_client.InferenceServerClient(
            url=url, verbose=self.verbose)

        self.audio_signals_name, self.num_samples_name, self.transcripts_name, \
        self.audio_signals_type, self.num_samples_type, self.transcripts_type =  self.parse_model(# server_status,
            model_name,
            batch_size, model_platform, verbose)
        self.labels = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", "<BLANK>"]

    def postprocess(self, transcript_values, labels):

        res = []
        for transcript, filename in zip(transcript_values,
                                        labels):
            print('---')
            print('File: ', filename)
            t=ctc_decoder_predictions_tensor(transcript, self.batch_size, self.labels)
            print("Final transcript: ", t)
            print('---')
            res.append(t)
        return res


    def check_num_samples(self, num_samples, model_name):
        if num_samples['data_type'] != 'TYPE_UINT32' and num_samples['data_type'] != 'TYPE_INT32':
             raise Exception(
                    "expecting num_samples datatype to be TYPE_UINT32/TYPE_INT32, "
                    "model '" + model_name + "' output type is " +
                    model_config.DataType.Name(num_samples['data_type']))
        if len(num_samples['dims']) != 1:
            raise Exception("Expecting num_samples to have 1 dimension, "
                            "model '{}' num_samples has {}".format(
                                model_name,len(num_samples['dims'])))

    def parse_model(self, # server_status,
                    model_name, batch_size,
                    model_platform=None, verbose=False):
        """
        Check the configuration of the ensemble model
        """

        if self.prtcl_client is tritonclient.grpc:
            config = self.triton_client.get_model_config(model_name, as_json=True)
        else:
            config = self.triton_client.get_model_config(model_name)

        self.model_platform = model_platform

        # Inputs are:
        #   1) audio_signal: raw audio samples [num_samples]
        #   2) sample_rate: sample rate of audio
        #   3) num_samples: length of audio

        if len(config['input']) < 2:
            raise Exception(
                "expecting 2-3 inputs, got {}".format(len(config['input'])))

        # Outputs are:
        #   1) transcripts:        candidate transcripts

        if len(config['output']) != 1:
            raise Exception(
                "expecting 1 output, got {}".format(len(config['output'])))

        audio_signal = config['input'][0]

        if len(config['input']) > 1:
            num_samples = config['input'][1]
            self.check_num_samples(num_samples, model_name);

        transcripts = config['output'][0]

        expected_audio_signal_dim = 1

        # Model specifying maximum batch size of 0 indicates that batching
        # is not supported and so the input tensors do not expect an "N"
        # dimension (and 'batch_size' should be 1 so that only a single
        # image instance is inferred at a time).
        max_batch_size = config['max_batch_size']
        if max_batch_size == 0:
            if batch_size != 1:
                raise Exception(
                    "batching not supported for model '" + model_name + "'")
        else:  # max_batch_size > 0
            if batch_size > max_batch_size:
                raise Exception(
                    "expecting batch size <= {} for model {}".format(
                        max_batch_size, model_name))

        if len(audio_signal['dims']) != expected_audio_signal_dim:
            raise Exception("Expecting audio signal to have {} dimensions, "
                            "model '{}' audio_signal has {}".format(
                expected_audio_signal_dim,
                model_name,
                len(audio_signal.dims)))

        return (audio_signal['name'],
                num_samples['name'],
                transcripts['name'],
                triton_type_to_np_dtype[audio_signal['data_type']],
                triton_type_to_np_dtype[num_samples['data_type']],
                triton_type_to_np_dtype[transcripts['data_type']])


    def recognize(self, audio_signal, filenames):
        # Send requests of FLAGS.batch_size audio signals. If the number of
        # audios isn't an exact multiple of FLAGS.batch_size then just
        # start over with the first audio until the batch is filled.

        input_batch = []
        input_filenames = []
        max_num_samples_batch = 0

        for idx in range(self.batch_size):
            input_batch.append(audio_signal[idx].astype(
                self.audio_signals_type))
            input_filenames.append(filenames[idx])
            num_samples = audio_signal[idx].shape[0]

            if (num_samples > max_num_samples_batch):
                max_num_samples_batch = num_samples

        for idx in range(self.batch_size):
            num_samples = input_batch[idx].shape[0]

            mean = np.mean(input_batch[idx])
            std_var = np.std(input_batch[idx])
            gauss_noise = np.random.normal(
                mean,std_var,
                max_num_samples_batch-num_samples)

            input_batch[idx]= np.concatenate(
                (input_batch[idx], gauss_noise.astype(
                    self.audio_signals_type)))

        max_num_samples_batch = np.asarray([max_num_samples_batch],
                                           dtype=self.num_samples_type)

        num_samples_batch = [max_num_samples_batch]*self.batch_size

        # Send request
        print("Sending request to transcribe file(s):", ",".join(
            input_filenames))

        inputs = []

        input_batch = np.asarray(input_batch)
        num_samples_batch = np.asarray(num_samples_batch)

        inputs.append(self.prtcl_client.InferInput(self.audio_signals_name,
                                                   input_batch.shape,
                                                   np_to_triton_dtype(input_batch.dtype)))
        inputs.append(self.prtcl_client.InferInput(self.num_samples_name,
                                                   num_samples_batch.shape,
                                                   np_to_triton_dtype(num_samples_batch.dtype)))

        if self.prtcl_client is tritonclient.grpc:
            inputs[0].set_data_from_numpy(input_batch)
            inputs[1].set_data_from_numpy(num_samples_batch)
        else: # http
            inputs[0].set_data_from_numpy(input_batch, binary_data=True)
            inputs[1].set_data_from_numpy(num_samples_batch, binary_data=True)

        outputs = []
        if self.prtcl_client is tritonclient.grpc:
            outputs.append(self.prtcl_client.InferRequestedOutput(self.transcripts_name))
        else:
            outputs.append(self.prtcl_client.InferRequestedOutput(self.transcripts_name,
                                                                  binary_data=True))

        triton_result = self.triton_client.infer(self.model_name, inputs=inputs,
                                          outputs=outputs)
        transcripts = triton_result.as_numpy(self.transcripts_name)

        result = self.postprocess(transcripts, input_filenames)

        return result


def preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff*signal[:-1])


def normalize_signal(signal, gain=None):
    """
    Normalize float32 signal to [-1, 1] range
    """
    if gain is None:
        gain = 1.0/(np.max(np.abs(signal)) + 1e-5)
    return signal*gain



class AudioSegment(object):
    """Monaural audio segment abstraction.
    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(self, samples, sample_rate, target_sr=16000, trim=False,
                 trim_db=60):
        """Create audio segment from samples.
        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        samples = self._convert_samples_to_float32(samples)
        self._samples = samples
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

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

    @classmethod
    def from_file(cls, filename, target_sr=16000, int_values=False, offset=0,
                  duration=0, trim=False):
        """
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
        return cls(samples, sample_rate, target_sr=target_sr, trim=trim)

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate

# define our clear function
def clear_screen():
    _ = system('clear')
