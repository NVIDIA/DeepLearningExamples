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

import os
import math
import numpy as np
import torch.distributed as dist
from .iterator import DaliJasperIterator, SyntheticDataIterator
from .pipeline import DaliPipeline
from common.helpers import print_once


def _parse_json(json_path: str, start_label=0, predicate=lambda json: True):
    """
    Parses json file to the format required by DALI
    Args:
        json_path: path to json file
        start_label: the label, starting from which DALI will assign consecutive int numbers to every transcript
        predicate: function, that accepts a sample descriptor (i.e. json dictionary) as an argument.
                   If the predicate for a given sample returns True, it will be included in the dataset.

    Returns:
        output_files: dictionary, that maps file name to label assigned by DALI
        transcripts: dictionary, that maps label assigned by DALI to the transcript
    """
    import json
    global cnt
    with open(json_path) as f:
        librispeech_json = json.load(f)
    output_files = {}
    transcripts = {}
    curr_label = start_label
    for original_sample in librispeech_json:
        if not predicate(original_sample):
            continue
        transcripts[curr_label] = original_sample['transcript']
        output_files[original_sample['files'][-1]['fname']] = curr_label
        curr_label += 1
    return output_files, transcripts


def _dict_to_file(dict: dict, filename: str):
    with open(filename, "w") as f:
        for key, value in dict.items():
            f.write("{} {}\n".format(key, value))


class DaliDataLoader:
    """
    DataLoader is the main entry point to the data preprocessing pipeline.
    To use, create an object and then just iterate over `data_iterator`.
    DataLoader will do the rest for you.
    Example:
        data_layer = DataLoader(DaliTrainPipeline, path, json, bs, ngpu)
        data_it = data_layer.data_iterator
        for data in data_it:
            print(data)  # Here's your preprocessed data

    Args:
        device_type: Which device to use for preprocessing. Choose: "cpu", "gpu"
        pipeline_type: Choose: "train", "val", "synth"
    """

    def __init__(self, gpu_id, dataset_path: str, config_data: dict, config_features: dict, json_names: list,
                 symbols: list, batch_size: int, pipeline_type: str, grad_accumulation_steps: int = 1,
                 synth_iters_per_epoch: int = 544, device_type: str = "gpu"):
        import torch
        self.batch_size = batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.drop_last = (pipeline_type == 'train')
        self.device_type = device_type
        pipeline_type = self._parse_pipeline_type(pipeline_type)
        if pipeline_type == "synth":
            self._dali_data_iterator = self._init_synth_iterator(self.batch_size, config_features['nfilt'],
                                                                 iters_per_epoch=synth_iters_per_epoch,
                                                                 ngpus=torch.distributed.get_world_size())
        else:
            self._dali_data_iterator = self._init_iterator(gpu_id=gpu_id, dataset_path=dataset_path,
                                                           config_data=config_data,
                                                           config_features=config_features,
                                                           json_names=json_names, symbols=symbols,
                                                           train_pipeline=pipeline_type == "train")

    def _init_iterator(self, gpu_id, dataset_path, config_data, config_features, json_names: list, symbols: list,
                       train_pipeline: bool):
        """
        Returns data iterator. Data underneath this operator is preprocessed within Dali
        """

        def hash_list_of_strings(li):
            return str(abs(hash(''.join(li))))

        output_files, transcripts = {}, {}
        max_duration = config_data['max_duration']
        for jname in json_names:
            of, tr = _parse_json(jname if jname[0] == '/' else os.path.join(dataset_path, jname), len(output_files),
                                 predicate=lambda json: json['original_duration'] <= max_duration)
            output_files.update(of)
            transcripts.update(tr)
        file_list_path = os.path.join("/tmp", "jasper_dali.file_list." + hash_list_of_strings(json_names))
        _dict_to_file(output_files, file_list_path)
        self.dataset_size = len(output_files)
        print_once(f"Dataset read by DALI. Number of samples: {self.dataset_size}")

        pipeline = DaliPipeline.from_config(config_data=config_data, config_features=config_features, device_id=gpu_id,
                                            file_root=dataset_path, file_list=file_list_path,
                                            device_type=self.device_type, batch_size=self.batch_size,
                                            train_pipeline=train_pipeline)

        return DaliJasperIterator([pipeline], transcripts=transcripts, symbols=symbols, batch_size=self.batch_size,
                                  reader_name="file_reader", train_iterator=train_pipeline)

    def _init_synth_iterator(self, batch_size, nfeatures, iters_per_epoch, ngpus):
        self.dataset_size = ngpus * iters_per_epoch * batch_size
        return SyntheticDataIterator(batch_size, nfeatures, regenerate=True)

    @staticmethod
    def _parse_pipeline_type(pipeline_type):
        pipe = pipeline_type.lower()
        assert pipe in ("train", "val", "synth"), 'Invalid pipeline type (choices: "train", "val", "synth").'
        return pipe

    def _shard_size(self):
        """
        Total number of samples handled by a single GPU in a single epoch.
        """
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if self.drop_last:
            divisor = world_size * self.batch_size * self.grad_accumulation_steps
            return self.dataset_size // divisor * divisor // world_size
        else:
            return int(math.ceil(self.dataset_size / world_size))

    def __len__(self):
        """
        Number of batches handled by each GPU.
        """
        if self.drop_last:
            assert self._shard_size() % self.batch_size == 0, f'{self._shard_size()} {self.batch_size}'

        return int(math.ceil(self._shard_size() / self.batch_size))

    def data_iterator(self):
        return self._dali_data_iterator

    def __iter__(self):
        return self._dali_data_iterator
