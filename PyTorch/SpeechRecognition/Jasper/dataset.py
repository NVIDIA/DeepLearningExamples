# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains classes and functions related to data loading
"""
import torch
import numpy as np
import math
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist
from parts.manifest import Manifest
from parts.features import WaveformFeaturizer

class DistributedBucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None):
        """Distributed sampler that buckets samples with similar length to minimize padding,
          similar concept as pytorch BucketBatchSampler  https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.samplers.html#torchnlp.samplers.BucketBatchSampler

        Args:
            dataset: Dataset used for sampling.
            batch_size: data batch size
            num_replicas (optional): Number of processes participating in
                distributed training.
            rank (optional): Rank of the current process within num_replicas.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.batch_size = batch_size
        self.tile_size = batch_size * self.num_replicas
        self.num_buckets = 6
        self.bucket_size = self.round_up_to(math.ceil(self.dataset_size / self.num_buckets), self.tile_size)
        self.index_count = self.round_up_to(self.dataset_size, self.tile_size)
        self.num_samples = self.index_count // self.num_replicas

    def round_up_to(self, x, mod):
        return (x + mod - 1) // mod * mod

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = np.arange(self.index_count) % self.dataset_size
        for bucket in range(self.num_buckets):
            bucket_start = self.bucket_size * bucket
            bucket_end = min(bucket_start + self.bucket_size, self.index_count)
            indices[bucket_start:bucket_end] = indices[bucket_start:bucket_end][torch.randperm(bucket_end - bucket_start, generator=g)]

        tile_indices = torch.randperm(self.index_count // self.tile_size, generator=g)
        for tile_index in tile_indices:
            start_index = self.tile_size * tile_index + self.batch_size * self.rank
            end_index = start_index + self.batch_size
            yield indices[start_index:end_index]

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = [ x.cuda(non_blocking=True) for x in self.next_input]

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input
    def next(self):
        return self.__next__()
    def __iter__(self):
        return self

def seq_collate_fn(batch):
    """batches samples and returns as tensors
    Args:
    batch : list of samples
    Returns
    batches of tensors
    """
    batch_size = len(batch)
    def _find_max_len(lst, ind):
        max_len = -1
        for item in lst:
            if item[ind].size(0) > max_len:
                max_len = item[ind].size(0)
        return max_len
    max_audio_len = _find_max_len(batch, 0)
    max_transcript_len = _find_max_len(batch, 2)

    batched_audio_signal = torch.zeros(batch_size, max_audio_len)
    batched_transcript = torch.zeros(batch_size, max_transcript_len)
    audio_lengths = []
    transcript_lengths = []
    for ind, sample in enumerate(batch):
        batched_audio_signal[ind].narrow(0, 0, sample[0].size(0)).copy_(sample[0])
        audio_lengths.append(sample[1])
        batched_transcript[ind].narrow(0, 0, sample[2].size(0)).copy_(sample[2])
        transcript_lengths.append(sample[3])
    return batched_audio_signal, torch.stack(audio_lengths), batched_transcript, \
         torch.stack(transcript_lengths)

class AudioToTextDataLayer:
    """Data layer with data loader
    """
    def __init__(self, **kwargs):
        self._device = torch.device("cuda")

        featurizer_config = kwargs['featurizer_config']
        pad_to_max = kwargs.get('pad_to_max', False)
        perturb_config = kwargs.get('perturb_config', None)
        manifest_filepath = kwargs['manifest_filepath']
        dataset_dir = kwargs['dataset_dir']
        labels = kwargs['labels']
        batch_size = kwargs['batch_size']
        drop_last = kwargs.get('drop_last', False)
        shuffle = kwargs.get('shuffle', True)
        min_duration = featurizer_config.get('min_duration', 0.1)
        max_duration = featurizer_config.get('max_duration', None)
        normalize_transcripts = kwargs.get('normalize_transcripts', True)
        trim_silence = kwargs.get('trim_silence', False)
        multi_gpu = kwargs.get('multi_gpu', False)
        sampler_type = kwargs.get('sampler', 'default')
        speed_perturbation = featurizer_config.get('speed_perturbation', False)
        sort_by_duration=sampler_type == 'bucket'
        self._featurizer = WaveformFeaturizer.from_config(featurizer_config, perturbation_configs=perturb_config)
        self._dataset = AudioDataset(
            dataset_dir=dataset_dir,
            manifest_filepath=manifest_filepath,
            labels=labels, blank_index=len(labels),
            sort_by_duration=sort_by_duration,
            pad_to_max=pad_to_max,
            featurizer=self._featurizer, max_duration=max_duration,
            min_duration=min_duration, normalize=normalize_transcripts,
            trim=trim_silence, speed_perturbation=speed_perturbation)

        print('sort_by_duration', sort_by_duration)

        if not multi_gpu:
            self.sampler = None
            self._dataloader = torch.utils.data.DataLoader(
                dataset=self._dataset,
                batch_size=batch_size,
                collate_fn=lambda b: seq_collate_fn(b),
                drop_last=drop_last,
                shuffle=shuffle if self.sampler is None else False,
                num_workers=4,
                pin_memory=True,
                sampler=self.sampler
            )
        elif sampler_type == 'bucket':
            self.sampler = DistributedBucketBatchSampler(self._dataset, batch_size=batch_size)
            print("DDBucketSampler")
            self._dataloader = torch.utils.data.DataLoader(
                dataset=self._dataset,
                collate_fn=lambda b: seq_collate_fn(b),
                num_workers=4,
                pin_memory=True,
                batch_sampler=self.sampler
            )
        elif sampler_type == 'default':
            self.sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)
            print("DDSampler")
            self._dataloader = torch.utils.data.DataLoader(
                dataset=self._dataset,
                batch_size=batch_size,
                collate_fn=lambda b: seq_collate_fn(b),
                drop_last=drop_last,
                shuffle=shuffle if self.sampler is None else False,
                num_workers=4,
                pin_memory=True,
                sampler=self.sampler
            )
        else:
            raise RuntimeError("Sampler {} not supported".format(sampler_type))

    def __len__(self):
        return len(self._dataset)

    @property
    def data_iterator(self):
        return self._dataloader

class AudioDataset(Dataset):
    def __init__(self, dataset_dir, manifest_filepath, labels, featurizer, max_duration=None, pad_to_max=False,
                 min_duration=None, blank_index=0, max_utts=0, normalize=True, sort_by_duration=False,
                 trim=False, speed_perturbation=False):
        """Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations
        (in seconds). Each entry is a different audio sample.
        Args:
            dataset_dir: absolute path to dataset folder
            manifest_filepath: relative path from dataset folder to manifest json as described above. Can be coma-separated paths.
            labels: String containing all the possible characters to map to
            featurizer: Initialized featurizer class that converts paths of audio to feature tensors
            max_duration: If audio exceeds this length, do not include in dataset
            min_duration: If audio is less than this length, do not include in dataset
            pad_to_max: if specified input sequences into dnn model will be padded to max_duration
            blank_index: blank index for ctc loss / decoder
            max_utts: Limit number of utterances
            normalize: whether to normalize transcript text
            sort_by_duration: whether or not to sort sequences by increasing duration
            trim: if specified trims leading and trailing silence from an audio signal.
            speed_perturbation: specify if using data contains speed perburbation
        """
        m_paths = manifest_filepath.split(',')
        self.manifest = Manifest(dataset_dir, m_paths, labels, blank_index, pad_to_max=pad_to_max,
                             max_duration=max_duration,
                             sort_by_duration=sort_by_duration,
                             min_duration=min_duration, max_utts=max_utts,
                             normalize=normalize, speed_perturbation=speed_perturbation)
        self.featurizer = featurizer
        self.blank_index = blank_index
        self.trim = trim
        print(
            "Dataset loaded with {0:.2f} hours. Filtered {1:.2f} hours.".format(
            self.manifest.duration / 3600,
            self.manifest.filtered_duration / 3600))

    def __getitem__(self, index):
        sample = self.manifest[index]
        rn_indx = np.random.randint(len(sample['audio_filepath']))
        duration = sample['audio_duration'][rn_indx] if 'audio_duration' in sample else 0
        offset = sample['offset'] if 'offset' in sample else 0
        features = self.featurizer.process(sample['audio_filepath'][rn_indx],
                                           offset=offset, duration=duration,
                                           trim=self.trim)

        return features, torch.tensor(features.shape[0]).int(), \
               torch.tensor(sample["transcript"]), torch.tensor(
               len(sample["transcript"])).int()

    def __len__(self):
        return len(self.manifest)
