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

import torch
import torch.distributed as dist
import numpy as np
from common.helpers import print_once
from common.text import _clean_text, punctuation_map


def normalize_string(s, symbols, punct_map):
    """
    Normalizes string.
    Example:
        'call me at 8:00 pm!' -> 'call me at eight zero pm'
    """
    labels = set(symbols)
    try:
        text = _clean_text(s, ["english_cleaners"], punct_map).strip()
        return ''.join([tok for tok in text if all(t in labels for t in tok)])
    except Exception as e:
        print_once("WARNING: Normalizing failed: {s} {e}")


class DaliJasperIterator(object):
    """
    Returns batches of data for Jasper training:
    preprocessed_signal, preprocessed_signal_length, transcript, transcript_length

    This iterator is not meant to be the entry point to Dali processing pipeline.
    Use DataLoader instead.
    """

    def __init__(self, dali_pipelines, transcripts, symbols, batch_size, reader_name, train_iterator: bool):
        self.transcripts = transcripts
        self.symbols = symbols
        self.batch_size = batch_size
        from nvidia.dali.plugin.pytorch import DALIGenericIterator
        from nvidia.dali.plugin.base_iterator import LastBatchPolicy

        self.dali_it = DALIGenericIterator(
            dali_pipelines, ["audio", "label", "audio_shape"], reader_name=reader_name,
            dynamic_shape=True, auto_reset=True,
            last_batch_policy=(LastBatchPolicy.DROP if train_iterator else LastBatchPolicy.PARTIAL))

    @staticmethod
    def _str2list(s: str):
        """
        Returns list of floats, that represents given string.
        '0.' denotes separator
        '1.' denotes 'a'
        '27.' denotes "'"
        Assumes, that the string is lower case.
        """
        list = []
        for c in s:
            if c == "'":
                list.append(27.)
            else:
                list.append(max(0., ord(c) - 96.))
        return list

    @staticmethod
    def _pad_lists(lists: list, pad_val=0):
        """
        Pads lists, so that all have the same size.
        Returns list with actual sizes of corresponding input lists
        """
        max_length = 0
        sizes = []
        for li in lists:
            sizes.append(len(li))
            max_length = max_length if len(li) < max_length else len(li)
        for li in lists:
            li += [pad_val] * (max_length - len(li))
        return sizes

    def _gen_transcripts(self, labels, normalize_transcripts: bool = True):
        """
        Generate transcripts in format expected by NN
        """
        lists = [
            self._str2list(normalize_string(self.transcripts[lab.item()], self.symbols, punctuation_map(self.symbols)))
            for lab in labels
        ] if normalize_transcripts else [self._str2list(self.transcripts[lab.item()]) for lab in labels]
        sizes = self._pad_lists(lists)
        return torch.tensor(lists).cuda(), torch.tensor(sizes, dtype=torch.int32).cuda()

    def __next__(self):
        data = self.dali_it.__next__()
        transcripts, transcripts_lengths = self._gen_transcripts(data[0]["label"])
        return data[0]["audio"], data[0]["audio_shape"][:, 1], transcripts, transcripts_lengths

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


# TODO: refactor
class SyntheticDataIterator(object):
    def __init__(self, batch_size, nfeatures, feat_min=-5., feat_max=0., txt_min=0., txt_max=23., feat_lens_max=1760,
                 txt_lens_max=231, regenerate=False):
        """
        Args:
            batch_size
            nfeatures: number of features for melfbanks
            feat_min: minimum value in `feat` tensor, used for randomization
            feat_max: maximum value in `feat` tensor, used for randomization
            txt_min: minimum value in `txt` tensor, used for randomization
            txt_max: maximum value in `txt` tensor, used for randomization
            regenerate: If True, regenerate random tensors for every iterator step.
                        If False, generate them only at start.
        """
        self.batch_size = batch_size
        self.nfeatures = nfeatures
        self.feat_min = feat_min
        self.feat_max = feat_max
        self.feat_lens_max = feat_lens_max
        self.txt_min = txt_min
        self.txt_max = txt_max
        self.txt_lens_max = txt_lens_max
        self.regenerate = regenerate

        if not self.regenerate:
            self.feat, self.feat_lens, self.txt, self.txt_lens = self._generate_sample()

    def _generate_sample(self):
        feat = (self.feat_max - self.feat_min) * np.random.random_sample(
            (self.batch_size, self.nfeatures, self.feat_lens_max)) + self.feat_min
        feat_lens = np.random.randint(0, int(self.feat_lens_max) - 1, size=self.batch_size)
        txt = (self.txt_max - self.txt_min) * np.random.random_sample(
            (self.batch_size, self.txt_lens_max)) + self.txt_min
        txt_lens = np.random.randint(0, int(self.txt_lens_max) - 1, size=self.batch_size)
        return torch.Tensor(feat).cuda(), \
               torch.Tensor(feat_lens).cuda(), \
               torch.Tensor(txt).cuda(), \
               torch.Tensor(txt_lens).cuda()

    def __next__(self):
        if self.regenerate:
            return self._generate_sample()
        return self.feat, self.feat_lens, self.txt, self.txt_lens

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self
