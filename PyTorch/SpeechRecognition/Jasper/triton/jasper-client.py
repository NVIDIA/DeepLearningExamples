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


import sys
import argparse
import numpy as np
import os
from speech_utils import AudioSegment, SpeechClient
import soundfile
import pyaudio as pa
import threading
import math
import time
import glob

FLAGS = None


# read audio chunk from a file
def get_audio_chunk_from_soundfile(sf, chunk_size, int_values):

    dtype = 'int32' if int_values else 'float32'
    audio_signal = sf.read(chunk_size, dtype=dtype)
    end = False
    # pad to chunk size
    if len(audio_signal) < chunk_size:
        end = True
        audio_signal = np.pad(audio_signal, (0, chunk_size-len(
            audio_signal)), mode='constant')
    return audio_signal, end


# generator that returns chunks of audio data from file
def audio_generator_from_file(input_filename, target_sr, int_values,
                              chunk_duration):

    sf = soundfile.SoundFile(input_filename, 'rb')
    chunk_size = int(chunk_duration*sf.samplerate)
    start = True
    end = False

    while not end:

        audio_signal, end = get_audio_chunk_from_soundfile(
            sf, chunk_size, int_values)

        audio_segment = AudioSegment(audio_signal, sf.samplerate, target_sr)

        yield audio_segment.samples, target_sr, start, end

        start = False

    sf.close()


# generator that returns chunks of audio data from file
class AudioGeneratorFromMicrophone:

    def __init__(self,input_device_id, target_sr, chunk_duration):

        self.recording_state = "init"
        self.target_sr  = target_sr
        self.chunk_duration = chunk_duration

        self.p = pa.PyAudio()

        device_info = self.p.get_host_api_info_by_index(0)
        num_devices = device_info.get('deviceCount')
        devices = {}
        for i in range(0, num_devices):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get(
                'maxInputChannels')) > 0:
                devices[i] = self.p.get_device_info_by_host_api_device_index(
                    0, i)

        if (len(devices) == 0):
            raise RuntimeError("Cannot find any valid input devices")

        if input_device_id is None or input_device_id not in \
            devices.keys():
            print("\nInput Devices:")
            for id, info in devices.items():
                print("{}: {}".format(id,info.get("name")))
            input_device_id = int(input("Enter device id to use: "))

        self.input_device_id = input_device_id


    def generate_audio(self):

        chunk_size = int(self.chunk_duration*self.target_sr)


        self. recording_state = "init"

        def keyboard_listener():
            input("Press Enter to start and end recording...")
            self.recording_state = "capture"
            print("Recording...")

            input("")
            self.recording_state = "release"

        listener = threading.Thread(target=keyboard_listener)
        listener.start()

        start = True
        end = False

        stream_initialized = False
        step = 0
        while self.recording_state != "release":
            try:
                if self.recording_state == "capture":

                    if not stream_initialized:
                        stream = self.p.open(
                            format=pa.paInt16,
                            channels=1,
                            rate=self.target_sr,
                            input=True,
                            input_device_index=self.input_device_id,
                            frames_per_buffer=chunk_size)
                        stream_initialized = True

                    # Read audio chunk from microphone
                    audio_signal = stream.read(chunk_size)
                    audio_signal = np.frombuffer(audio_signal,dtype=np.int16)
                    audio_segment = AudioSegment(audio_signal,
                                                              self.target_sr,
                                                              self.target_sr)

                    yield audio_segment.samples, self.target_sr, start, end

                    start = False
                    step += 1
            except Exception as e:
                print(e)
                break

        stream.close()
        self.p.terminate()

    def generate_audio_signal(self):


        #chunk_size = int(self.chunk_duration*self.target_sr)
        chunk_size = int(0.2*self.target_sr)
        self. recording_state = "init"

        def keyboard_listener():
            input("Press Enter to start and end recording...")
            self.recording_state = "capture"
            print("Recording...")

            input("")
            self.recording_state = "release"

        listener = threading.Thread(target=keyboard_listener)
        listener.start()

        audio_samples = []
        stream_initialized = False
        step = 0
        while self.recording_state != "release":
            try:
                if self.recording_state == "capture":

                    if not stream_initialized:
                        stream = self.p.open(
                            format=pa.paInt16,
                            channels=1,
                            rate=self.target_sr,
                            input=True,
                            input_device_index=self.input_device_id,
                            frames_per_buffer=chunk_size)
                        stream_initialized = True

                    # Read audio chunk from microphone
                    audio_signal = stream.read(chunk_size)
                    audio_signal = np.frombuffer(audio_signal,dtype=np.int16)
                    audio_segment = AudioSegment(audio_signal,
                                                              self.target_sr,
                                                              self.target_sr)

                    if step == 0:
                        audio_samples = audio_segment.samples
                    else:
                        audio_samples = np.concatenate((audio_samples,
                                                       audio_segment.samples))

                    start = False
                    step += 1
            except Exception as e:
                print(e)
                break

        stream.close()
        self.p.terminate()

        return audio_samples

# generator that returns chunks of audio features from file
def audio_features_generator(input_filename, speech_features_params,
                             target_sr, int_values, chunk_duration):

    sf = soundfile.SoundFile(input_filename, 'rb')

    chunk_size = int(chunk_duration*sf.samplerate)

    start = True
    end = False

    while not end:

        audio_signal, end = get_audio_chunk_from_soundfile(sf, chunk_size,
                                                       int_values)

        audio_segment = AudioSegment(audio_signal, sf.samplerate, target_sr)
        audio_features, features_length = get_speech_features(
          audio_segment.samples, target_sr, speech_features_params)

        yield audio_features, start, end

        start = False

    sf.close()


def audio_features_generator_with_buffer(input_filename,
                                         speech_features_params, target_sr,
                                         int_values, chunk_duration):

    sf = soundfile.SoundFile(input_filename, 'rb')

    chunk_size = int(chunk_duration*sf.samplerate)

    start = True
    end = False

    audio_signal = np.zeros(shape=3*chunk_size, dtype=np.float32)

    while not end:

        audio_signal[-chunk_size:], end = get_audio_chunk_from_soundfile(sf, chunk_size, int_values)

        audio_segment = AudioSegment(audio_signal, sf.samplerate, target_sr)
        audio_features, features_length = get_speech_features(
          audio_segment.samples, target_sr, speech_features_params)

        yield audio_features, start, end

        start = False
        audio_signal[:-chunk_size] = audio_signal[chunk_size:]


    sf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False,
                        default=False, help='Enable verbose output')
    parser.add_argument('--fixed_size', type=int, required=False,
                        default=0,
                        help="send fixed_size requests, pad or truncate")    
    parser.add_argument('--batch_size', type=int, required=False, default=1,
                        help='batch size')
    parser.add_argument('--model_platform', required=False,
                        default='trt',
                        help='Jasper model platform')
    parser.add_argument('-u', '--url', type=str, required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is '
                             'localhost:8000.')
    parser.add_argument('-i', '--protocol', type=str, required=False,
                        default='HTTP',
                        help='Protocol (HTTP/gRPC) used to communicate with '
                             'inference service. Default is HTTP.')
    parser.add_argument('--audio_filename', type=str, required=False,
                        default=None,
                        help='Input audio filename')
    parser.add_argument('--data_dir', type=str, required=False,
                        default=None,
                        help='data directory')
    parser.add_argument('--manifest_filename', type=str, required=False,
                        default=None,
                        help='relative manifest paths to --data_dir directory.')

    FLAGS = parser.parse_args()

    protocol = FLAGS.protocol.lower()

    valid_model_platforms = {"ts-trace","onnx", "tensorrt"}

    if FLAGS.model_platform not in valid_model_platforms:
        raise ValueError("Invalid model_platform {}. Valid choices are {"
                         "}".format(FLAGS.model_platform,
            valid_model_platforms))


    model_name = "jasper-" + FLAGS.model_platform + "-ensemble"

    speech_client = SpeechClient(
        FLAGS.url, protocol, model_name, 1,
        FLAGS.batch_size, model_platform=FLAGS.model_platform,
        verbose=FLAGS.verbose, mode="synchronous",
        from_features=False
    )
    
    filenames = []
    transcripts = []
    if FLAGS.audio_filename is not None:
        audio_file = os.path.join(FLAGS.data_dir, FLAGS.audio_filename)
        if os.path.isdir(audio_file):
            filenames = glob.glob(os.path.join(os.path.abspath(audio_file), "**", "*.wav"),
                                recursive=True)
        else:
            filenames = [audio_file]
    elif FLAGS.manifest_filename is not None:
        filter_speed=1.0
        data_dir=FLAGS.data_dir
        labels = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", "<BLANK>"]
        labels_map = dict([(labels[i], i) for i in range(len(labels))])
        blank_index = len(labels)-1
        table = None
        import string
        punctuation = string.punctuation
        punctuation = punctuation.replace("+", "")
        punctuation = punctuation.replace("&", "")
        table = str.maketrans(punctuation, " " * len(punctuation))

        import json
        if "./triton" not in sys.path:
            sys.path.append("./")
            sys.path.append("./triton")
        from speech_utils import normalize_string, parse_transcript
        FLAGS.manifest_filename = FLAGS.manifest_filename.split(',')
        for manifest in FLAGS.manifest_filename:
            manifest=os.path.join(data_dir, manifest)
            print(manifest)
            with open(manifest, "r", encoding="utf-8") as fh:
                a=json.load(fh)
                for data in a:
                    files_and_speeds = data['files']
                    audio_path = [x['fname'] for x in files_and_speeds if x['speed'] == filter_speed][0]
                    filenames.append(os.path.join(data_dir, audio_path))
                    transcript_text = data['transcript']
                    transcript_text = normalize_string(transcript_text, labels=labels, table=table)
                    transcripts.append(transcript_text) #parse_transcript(transcript_text, labels_map, blank_index)) # convert to vocab indices

        
        
    # Read the audio files
    # Group requests in batches
    audio_idx = 0
    last_request = False
    predictions = []
    while not last_request:
        batch_audio_samples = []
        batch_filenames = []
        
        for idx in range(FLAGS.batch_size):
            filename = filenames[audio_idx]
            print("Reading audio file: ", filename)
            audio = AudioSegment.from_file(
                filename,
                offset=0, duration=FLAGS.fixed_size).samples
            if FLAGS.fixed_size:
                audio = np.resize(audio, FLAGS.fixed_size)
                
            audio_idx = (audio_idx + 1) % len(filenames)
            if audio_idx == 0:
                last_request = True

            batch_audio_samples.append(audio)
            batch_filenames.append(filename)

        predictions += speech_client.recognize(
            batch_audio_samples,
            batch_filenames)

    if transcripts:
        predictions = [x for l in predictions for x in l ]
        from metrics import word_error_rate
        wer, scores, num_words = word_error_rate(predictions, transcripts)
        print(wer)
