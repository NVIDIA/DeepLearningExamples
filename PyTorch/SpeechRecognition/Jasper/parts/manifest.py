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

import json
import re
import string
import numpy as np
import os

from .text import _clean_text


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

class Manifest(object):
    def __init__(self, data_dir, manifest_paths, labels, blank_index, max_duration=None, pad_to_max=False,
                 min_duration=None, sort_by_duration=False, max_utts=0,
                 normalize=True, speed_perturbation=False, filter_speed=1.0):
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.blank_index = blank_index
        self.max_duration= max_duration
        ids = []
        duration = 0.0
        filtered_duration = 0.0

        # If removing punctuation, make a list of punctuation to remove
        table = None
        if normalize:
            # Punctuation to remove
            punctuation = string.punctuation
            punctuation = punctuation.replace("+", "")
            punctuation = punctuation.replace("&", "")
            ### We might also want to consider:
            ### @ -> at
            ### # -> number, pound, hashtag
            ### ~ -> tilde
            ### _ -> underscore
            ### % -> percent
            # If a punctuation symbol is inside our vocab, we do not remove from text
            for l in labels:
                punctuation = punctuation.replace(l, "")
            # Turn all punctuation to whitespace
            table = str.maketrans(punctuation, " " * len(punctuation))
        for manifest_path in manifest_paths:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                a=json.load(fh)
                for data in a:
                    files_and_speeds = data['files']

                    if pad_to_max:
                        if not speed_perturbation:
                            min_speed = filter_speed
                        else:
                            min_speed = min(x['speed'] for x in files_and_speeds)
                        max_duration = self.max_duration * min_speed

                    data['duration'] = data['original_duration']
                    if min_duration is not None and data['duration'] < min_duration:
                        filtered_duration += data['duration']
                        continue
                    if max_duration is not None and data['duration'] > max_duration:
                        filtered_duration += data['duration']
                        continue

                    # Prune and normalize according to transcript
                    transcript_text = data[
                        'transcript'] if "transcript" in data else self.load_transcript(
                        data['text_filepath'])
                    if normalize:
                        transcript_text = normalize_string(transcript_text, labels=labels,
                                                                                             table=table)
                    if not isinstance(transcript_text, str):
                        print(
                            "WARNING: Got transcript: {}. It is not a string. Dropping data point".format(
                                transcript_text))
                        filtered_duration += data['duration']
                        continue
                    data["transcript"] = self.parse_transcript(transcript_text) # convert to vocab indices

                    if speed_perturbation:
                        audio_paths = [x['fname'] for x in files_and_speeds]
                        data['audio_duration'] = [x['duration'] for x in files_and_speeds]
                    else:
                        audio_paths = [x['fname'] for x in files_and_speeds if x['speed'] == filter_speed]
                        data['audio_duration'] = [x['duration'] for x in files_and_speeds if x['speed'] == filter_speed]
                    data['audio_filepath'] = [os.path.join(data_dir, x) for x in audio_paths]
                    data.pop('files')
                    data.pop('original_duration')

                    ids.append(data)
                    duration += data['duration']

                    if max_utts > 0 and len(ids) >= max_utts:
                        print(
                            'Stopping parsing %s as max_utts=%d' % (manifest_path, max_utts))
                        break

        if sort_by_duration:
            ids = sorted(ids, key=lambda x: x['duration'])
        self._data = ids
        self._size = len(ids)
        self._duration = duration
        self._filtered_duration = filtered_duration

    def load_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding="utf-8") as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        return transcript

    def parse_transcript(self, transcript):
        chars = [self.labels_map.get(x, self.blank_index) for x in list(transcript)]
        transcript = list(filter(lambda x: x != self.blank_index, chars))
        return transcript

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(self._data)

    @property
    def duration(self):
        return self._duration

    @property
    def filtered_duration(self):
        return self._filtered_duration

    @property
    def data(self):
        return list(self._data)
