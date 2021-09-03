# Copyright (c) 2018, deepakn94, codyaustun, robieta. All rights reserved.
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
#
# -----------------------------------------------------------------------
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import torch
import os
from feature_spec import FeatureSpec
from neumf_constants import USER_CHANNEL_NAME, ITEM_CHANNEL_NAME, LABEL_CHANNEL_NAME, TEST_SAMPLES_PER_SERIES


class TorchTensorDataset:
    """ Warning! This dataset/loader uses torch.load. Torch.load implicitly uses pickle. Pickle is insecure.
    It is trivial to achieve arbitrary code execution using a prepared pickle payload. Only unpickle data you trust."""

    def __init__(self, feature_spec: FeatureSpec, mapping_name: str, args):
        self.local_rank = args.local_rank
        self.mapping_name = mapping_name
        self.features = dict()
        self.feature_spec = feature_spec
        self._load_features()

    def _load_features(self):
        chunks = self.feature_spec.source_spec[self.mapping_name]
        for chunk in chunks:
            assert chunk['type'] == 'torch_tensor', "Only torch_tensor files supported in this loader"
            files_list = chunk['files']
            assert len(files_list) == 1, "Only one file per chunk supported in this loader"
            file_relative_path = files_list[0]
            path_to_load = os.path.join(self.feature_spec.base_directory, file_relative_path)
            chunk_data = torch.load(path_to_load, map_location=torch.device('cuda:{}'.format(self.local_rank)))
            running_pos = 0
            for feature_name in chunk['features']:
                next_running_pos = running_pos + 1
                feature_data = chunk_data[:, running_pos:next_running_pos]
                # This is needed because slicing instead of indexing keeps the data 2-dimensional
                feature_data = feature_data.reshape(-1, 1)
                running_pos = next_running_pos
                self.features[feature_name] = feature_data


class TestDataLoader:
    def __init__(self, dataset: TorchTensorDataset, args):
        self.dataset = dataset
        self.feature_spec = dataset.feature_spec
        self.channel_spec = self.feature_spec.channel_spec
        self.samples_in_series = self.feature_spec.metadata[TEST_SAMPLES_PER_SERIES]
        self.raw_dataset_length = None  # First feature loaded sets this. Total length before splitting across cards
        self.data = dict()
        self.world_size = args.world_size
        self.local_rank = args.local_rank
        self.batch_size = args.valid_batch_size

        self._build_channel_dict()
        self._deduplication_augmentation()
        self._split_between_devices()
        self._split_into_batches()

    def _build_channel_dict(self):
        for channel_name, channel_features in self.channel_spec.items():
            channel_tensors = dict()
            for feature_name in channel_features:
                channel_tensors[feature_name] = self.dataset.features[feature_name]

                if not self.raw_dataset_length:
                    self.raw_dataset_length = channel_tensors[feature_name].shape[0]
                else:
                    assert self.raw_dataset_length == channel_tensors[feature_name].shape[0]

            self.data[channel_name] = channel_tensors

    def _deduplication_augmentation(self):
        # Augmentation
        # This adds a duplication mask tensor.
        # This is here to exactly replicate the MLPerf training regime. Moving this deduplication to the candidate item
        # generation stage increases the real diversity of the candidates, which makes the ranking task harder
        # and results in a drop in HR@10 of approx 0.01. This has been deemed unacceptable (May 2021).

        # We need the duplication mask to determine if a given item should be skipped during ranking
        # If an item with label 1 is duplicated in the sampled ones, we need to be careful to not mark the one with
        # label 1 as a duplicate. If an item appears repeatedly only with label 1, no duplicates are marked.

        # To easily compute candidates, we sort the items. This will impact the distribution of examples between
        # devices, but should not influence the numerics or performance meaningfully.
        # We need to assure that the positive item, which we don't want to mark as a duplicate, appears first.
        # We do this by adding labels as a secondary factor

        # Reshape the tensors to have items for a given user in a single row
        user_feature_name = self.channel_spec[USER_CHANNEL_NAME][0]
        item_feature_name = self.channel_spec[ITEM_CHANNEL_NAME][0]
        label_feature_name = self.channel_spec[LABEL_CHANNEL_NAME][0]
        self.ignore_mask_channel_name = 'mask_ch'
        self.ignore_mask_feature_name = 'mask'

        items = self.data[ITEM_CHANNEL_NAME][item_feature_name].view(-1, self.samples_in_series)
        users = self.data[USER_CHANNEL_NAME][user_feature_name].view(-1, self.samples_in_series)
        labels = self.data[LABEL_CHANNEL_NAME][label_feature_name].view(-1, self.samples_in_series)

        sorting_weights = items.float() - labels.float() * 0.5
        _, indices = torch.sort(sorting_weights)
        # The gather reorders according to the indices decided by the sort above
        sorted_items = torch.gather(items, 1, indices)
        sorted_labels = torch.gather(labels, 1, indices)
        sorted_users = torch.gather(users, 1, indices)

        dup_mask = sorted_items[:, 0:-1] == sorted_items[:, 1:]  # This says if a given item is equal to the next one
        dup_mask = dup_mask.type(torch.bool)
        # The first item for a given user can never be a duplicate:
        dup_mask = torch.cat((torch.zeros_like(dup_mask[:, 0:1]), dup_mask), dim=1)

        # Reshape them back
        self.data[ITEM_CHANNEL_NAME][item_feature_name] = sorted_items.view(-1, 1)
        self.data[USER_CHANNEL_NAME][user_feature_name] = sorted_users.view(-1, 1)
        self.data[LABEL_CHANNEL_NAME][label_feature_name] = sorted_labels.view(-1, 1)
        self.data[self.ignore_mask_channel_name] = dict()
        self.data[self.ignore_mask_channel_name][self.ignore_mask_feature_name] = dup_mask.view(-1, 1)

    def _split_between_devices(self):
        if self.world_size > 1:
            # DO NOT REPLACE WITH torch.chunk (number of returned chunks can silently be lower than requested).
            # It would break compatibility with small datasets.
            num_test_cases = self.raw_dataset_length / self.samples_in_series
            smaller_batch = (int(num_test_cases // self.world_size)) * self.samples_in_series
            bigger_batch = smaller_batch + self.samples_in_series
            remainder = int(num_test_cases % self.world_size)
            samples_per_card = [bigger_batch] * remainder + [smaller_batch] * (self.world_size - remainder)
            for channel_name, channel_dict in self.data.items():
                for feature_name, feature_tensor in channel_dict.items():
                    channel_dict[feature_name] = \
                        channel_dict[feature_name].split(samples_per_card)[self.local_rank]

    def _split_into_batches(self):
        self.batches = None
        # This is the structure of each batch, waiting to be copied and filled in with data
        for channel_name, channel_dict in self.data.items():
            for feature_name, feature_tensor in channel_dict.items():
                feature_batches = feature_tensor.view(-1).split(self.batch_size)
                if not self.batches:
                    self.batches = list(
                        {channel_name: dict() for channel_name in self.data.keys()} for _ in feature_batches)
                for pos, feature_batch_data in enumerate(feature_batches):
                    self.batches[pos][channel_name][feature_name] = feature_batch_data

    def get_epoch_data(self):
        return self.batches

    def get_ignore_mask(self):
        return self.data[self.ignore_mask_channel_name][self.ignore_mask_feature_name]


class TrainDataloader:
    def __init__(self, dataset: TorchTensorDataset, args):
        self.dataset = dataset
        self.local_rank = args.local_rank
        if args.distributed:
            self.local_batch = args.batch_size // args.world_size
        else:
            self.local_batch = args.batch_size

        self.feature_spec = dataset.feature_spec
        self.channel_spec = self.feature_spec.channel_spec
        self.negative_samples = args.negative_samples

        self.data = dict()
        self.raw_dataset_length = None  # first feature loaded sets this
        self._build_channel_dict()
        self.length_after_augmentation = self.raw_dataset_length * (self.negative_samples + 1)
        samples_per_worker = self.length_after_augmentation / args.world_size
        self.samples_begin = int(samples_per_worker * args.local_rank)
        self.samples_end = int(samples_per_worker * (args.local_rank + 1))

    def _build_channel_dict(self):
        for channel_name, channel_features in self.channel_spec.items():
            channel_tensors = dict()
            for feature_name in channel_features:
                channel_tensors[feature_name] = self.dataset.features[feature_name]
                if not self.raw_dataset_length:
                    self.raw_dataset_length = channel_tensors[feature_name].shape[0]
                else:
                    assert self.raw_dataset_length == channel_tensors[feature_name].shape[0]
            self.data[channel_name] = channel_tensors

    def get_epoch_data(self):

        # Augment, appending args.negative_samples times the original set, now with random items end negative labels
        augmented_data = {channel_name: dict() for channel_name in self.data.keys()}
        user_feature_name = self.channel_spec[USER_CHANNEL_NAME][0]
        item_feature_name = self.channel_spec[ITEM_CHANNEL_NAME][0]
        label_feature_name = self.channel_spec[LABEL_CHANNEL_NAME][0]

        # USER
        user_tensor = self.data[USER_CHANNEL_NAME][user_feature_name]
        neg_users = user_tensor.repeat(self.negative_samples, 1)
        augmented_users = torch.cat((user_tensor, neg_users))
        augmented_data[USER_CHANNEL_NAME][user_feature_name] = augmented_users
        del neg_users

        # ITEM
        item_tensor = self.data[ITEM_CHANNEL_NAME][item_feature_name]
        neg_items = torch.empty_like(item_tensor).repeat(self.negative_samples, 1) \
            .random_(0, self.feature_spec.feature_spec[item_feature_name]['cardinality'])
        augmented_items = torch.cat((item_tensor, neg_items))
        augmented_data[ITEM_CHANNEL_NAME][item_feature_name] = augmented_items
        del neg_items

        # LABEL
        label_tensor = self.data[LABEL_CHANNEL_NAME][label_feature_name]
        neg_label = torch.zeros_like(label_tensor, dtype=torch.float32).repeat(self.negative_samples, 1)
        augmented_labels = torch.cat((label_tensor, neg_label))
        del neg_label
        augmented_data[LABEL_CHANNEL_NAME][label_feature_name] = augmented_labels

        # Labels are not shuffled between cards.
        # This replicates previous behaviour.
        epoch_indices = torch.randperm(self.samples_end - self.samples_begin, device='cuda:{}'.format(self.local_rank))
        epoch_indices += self.samples_begin

        batches = None
        for channel_name, channel_dict in augmented_data.items():
            for feature_name, feature_tensor in channel_dict.items():
                # the last batch will almost certainly be smaller, drop it
                # Warning: may not work if there's only one
                feature_batches = feature_tensor.view(-1)[epoch_indices].split(self.local_batch)[:-1]
                if not batches:
                    batches = list({channel_name: dict() for channel_name in self.data.keys()} for _ in feature_batches)
                for pos, feature_batch_data in enumerate(feature_batches):
                    batches[pos][channel_name][feature_name] = feature_batch_data

        return batches
