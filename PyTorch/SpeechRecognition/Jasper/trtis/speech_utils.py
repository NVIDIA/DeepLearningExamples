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
from tensorrtserver.api import *
import tensorrtserver.api.model_config_pb2 as model_config
import grpc
from tensorrtserver.api import api_pb2
from tensorrtserver.api import grpc_service_pb2
from tensorrtserver.api import grpc_service_pb2_grpc
if "./trtis" not in sys.path:
    sys.path.append("./")
    sys.path.append("./trtis")
from parts.text import _clean_text

WINDOWS_FNS = {"hanning": np.hanning, "hamming": np.hamming, "none": None}


def model_dtype_to_np(model_dtype):
    if model_dtype == model_config.TYPE_BOOL:
        return np.bool
    elif model_dtype == model_config.TYPE_INT8:
        return np.int8
    elif model_dtype == model_config.TYPE_INT16:
        return np.int16
    elif model_dtype == model_config.TYPE_INT32:
        return np.int32
    elif model_dtype == model_config.TYPE_INT64:
        return np.int64
    elif model_dtype == model_config.TYPE_UINT8:
        return np.uint8
    elif model_dtype == model_config.TYPE_UINT16:
        return np.uint16
    elif model_dtype == model_config.TYPE_UINT32:
        return np.uint32
    elif model_dtype == model_config.TYPE_FP16:
        return np.float16
    elif model_dtype == model_config.TYPE_FP32:
        return np.float32
    elif model_dtype == model_config.TYPE_FP64:
        return np.float64
    elif model_dtype == model_config.TYPE_STRING:
        return np.dtype(object)
    return None

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

        self.ctx = InferContext(url, protocol, model_name, model_version,
                                verbose, self.correlation_id, False)
        server_ctx = ServerStatusContext(url, protocol, model_name,
                                         verbose)
        server_status = server_ctx.get_server_status()

        self.audio_signals_name, self.num_samples_name, self.transcripts_name, \
        self.audio_signals_type, self.num_samples_type, self.transcripts_type =  self.parse_model(server_status, model_name,
                                                                                                  batch_size, model_platform, verbose)
        self.labels = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", "<BLANK>"]

    def postprocess(self, results, labels):

        if len(results) != 1:
            raise Exception("expected 1 result, got {}".format(len(results)))

        transcript_values = results['TRANSCRIPT']
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
            

    def check_num_samples(self, num_samples):
        if num_samples.data_type != model_config.TYPE_UINT32 and num_samples.data_type != model_config.TYPE_INT32:
             raise Exception(
                    "expecting num_samples datatype to be TYPE_UINT32/TYPE_INT32, "
                    "model '" + model_name + "' output type is " +
                    model_config.DataType.Name(num_samples.data_type))
        if len(num_samples.dims) != 1:
            raise Exception("Expecting num_samples to have 1 dimension, "
                            "model '{}' num_samples has {}".format(
                                model_name,len(num_samples.dims)))

    def parse_model(self, server_status,
                    model_name, batch_size,
                    model_platform=None, verbose=False):
        """
        Check the configuration of the ensemble model
        """

        if model_name not in server_status.model_status:
            print("Server status:")
            print(server_status)
            raise Exception("unable to get status for '" + model_name + "'")

        status = server_status.model_status[model_name]
        config = status.config

        self.model_platform = model_platform

        # Inputs are:
        #   1) audio_signal: raw audio samples [num_samples]
        #   2) sample_rate: sample rate of audio
        #   3) num_samples: length of audio

        if len(config.input) < 2:
            raise Exception(
                "expecting 2-3 inputs, got {}".format(len(config.input)))

        # Outputs are:
        #   1) transcripts:        candidate transcripts

        if len(config.output) != 1:
            raise Exception(
                "expecting 1 output, got {}".format(len(config.output)))

        audio_signal = config.input[0]

        if len(config.input) > 1:
            num_samples = config.input[1]
            self.check_num_samples(num_samples);
            
        transcripts = config.output[0]

        expected_audio_signal_dim = 1
        expected_audio_signal_type = model_config.TYPE_FP32

        if audio_signal.data_type != expected_audio_signal_type:
            raise Exception("expecting audio_signal datatype to be " +
                            model_config.DataType.Name(
                                expected_audio_signal_type) +
                            "model '" + model_name + "' output type is " +
                            model_config.DataType.Name(audio_signal.data_type))


        # Model specifying maximum batch size of 0 indicates that batching
        # is not supported and so the input tensors do not expect an "N"
        # dimension (and 'batch_size' should be 1 so that only a single
        # image instance is inferred at a time).
        max_batch_size = config.max_batch_size
        if max_batch_size == 0:
            if batch_size != 1:
                raise Exception(
                    "batching not supported for model '" + model_name + "'")
        else:  # max_batch_size > 0
            if batch_size > max_batch_size:
                raise Exception(
                    "expecting batch size <= {} for model {}".format(
                        max_batch_size, model_name))

        if len(audio_signal.dims) != expected_audio_signal_dim:
            raise Exception("Expecting audio signal to have {} dimensions, "
                            "model '{}' audio_signal has {}".format(
                expected_audio_signal_dim,
                model_name,
                len(audio_signal.dims)))

        return (audio_signal.name, num_samples.name, transcripts.name, 
                model_dtype_to_np(audio_signal.data_type),
                model_dtype_to_np(num_samples.data_type),
                model_dtype_to_np(transcripts.data_type),
                )


    def update_audio_request(self, request, audio_generator):

        for audio_signal, sample_rate, start, end in audio_generator:
            # Delete the current inputs

            input_batch = [audio_signal.astype(self.audio_signals_type)]
            num_samples_batch = audio_signal.shape[0]
            num_samples_batch = [np.asarray([num_samples_batch],
                                            dtype=self.num_samples_type)]


            flags = InferRequestHeader.FLAG_NONE
            input_batch[0] = np.expand_dims(input_batch[0], axis=0)

            audio_bytes = input_batch[0].tobytes()
            num_samples_bytes = num_samples_batch[0].tobytes()

            request.meta_data.input[0].dims[0] = audio_signal.shape[0]
            request.meta_data.input[0].batch_byte_size = len(audio_bytes)

            request.meta_data.input[1].dims[0] = 1
            request.meta_data.input[1].batch_byte_size = len(num_samples_bytes)

            if start:
                request.meta_data.flags = flags | \
                                          InferRequestHeader.FLAG_SEQUENCE_START
            else:
                request.meta_data.flags = flags;

            # Send request with audio signal
            del request.raw_input[:]
            request.raw_input.extend([audio_bytes])
            request.raw_input.extend([num_samples_bytes])
            yield request

            # If end, send empty request to flush out remaining audio
            if end:
                request.meta_data.flags = flags | \
                                          InferRequestHeader.FLAG_SEQUENCE_END
                zero_bytes = np.zeros(shape=input_batch[0].shape,
                                      dtype=input_batch[0].dtype).tobytes()
                del request.raw_input[:]
                request.raw_input.extend([zero_bytes])
                request.raw_input.extend([num_samples_bytes])
                yield request

    def recognize(self, audio_signal, filenames):
        # Send requests of FLAGS.batch_size audio signals. If the number of
        # audios isn't an exact multiple of FLAGS.batch_size then just
        # start over with the first audio until the batch is filled.

        flags = InferRequestHeader.FLAG_NONE
        flags = flags | InferRequestHeader.FLAG_SEQUENCE_START

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
            print("num_samples : ", num_samples)
            #input_batch[idx] = np.pad(input_batch[idx],
            #                          ((0,
            #                            max_num_samples_batch -
            #                            num_samples)),
            #                          mode='constant')

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

        num_samples_batch = [max_num_samples_batch] * self.batch_size

        #print(num_samples_batch)
        #print(input_batch)
        #print(input_sample_rates)

        # Send request
        print("Sending request to transcribe file(s):", ",".join(
            input_filenames))

        if (self.model_platform == "obsolete_pyt"):
            result = self.ctx.run(
                {self.audio_signals_name: input_batch,
                 self.num_samples_name: num_samples_batch},
                {self.transcripts_name: InferContext.ResultFormat.RAW},
                self.batch_size, flags)
        else:
            result = self.ctx.run(
                {self.audio_signals_name: input_batch,
                 self.num_samples_name: num_samples_batch},
                {self.transcripts_name: InferContext.ResultFormat.RAW},
                self.batch_size, flags)

        res =self.postprocess(result, input_filenames)

        return res
 

def preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def normalize_signal(signal, gain=None):
    """
    Normalize float32 signal to [-1, 1] range
    """
    if gain is None:
        gain = 1.0 / (np.max(np.abs(signal)) + 1e-5)
    return signal * gain



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
