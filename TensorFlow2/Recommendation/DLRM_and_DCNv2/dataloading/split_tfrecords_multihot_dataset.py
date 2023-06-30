# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
# author: Tomasz Grel (tgrel@nvidia.com)


import tensorflow as tf
import os
import glob
import json
import numpy as np

import tqdm

def serialize_composite(rt):
    components = tf.nest.flatten(rt, expand_composites=True)
    tensor = tf.stack([tf.io.serialize_tensor(t) for t in components])
    return tf.io.serialize_tensor(tensor)


def deserialize_composite(serialized, type_spec):
    data = tf.io.parse_tensor(serialized, tf.string)
    component_specs = tf.nest.flatten(type_spec, expand_composites=True)
    components = [tf.io.parse_tensor(data[i], out_type=spec.dtype)
                   for i, spec in enumerate(component_specs)]
    return tf.nest.pack_sequence_as(type_spec, components, expand_composites=True)


def length_filename(dataset_dir):
    return f'{dataset_dir}/length.json'


class PrebatchStreamWriter:
    def __init__(self, dst_dir, dtype, feature_name='data', multihot=False, batches_per_file=1):
        self.dst_dir = dst_dir
        os.makedirs(dst_dir, exist_ok=True)
        self.dtype = dtype
        self.feature_name = feature_name
        self.multihot = multihot
        self.batches_per_file = batches_per_file
        self.writer = None
        self._file_idx = -1
        self._batches_saved = 0

    def _new_file(self):
        if self.writer:
            self.writer.close()

        self._file_idx += 1
        self.writer = tf.io.TFRecordWriter(os.path.join(self.dst_dir, f'data_{self._file_idx}.tfrecords'))

    def save(self, prebatch):
        if self._batches_saved % self.batches_per_file == 0:
            self._new_file()

        if self.multihot:
            serialized = serialize_composite(tf.cast(prebatch, self.dtype)).numpy()
        else:
            if isinstance(prebatch, tf.RaggedTensor):
                prebatch = prebatch.to_tensor()

            serialized = tf.io.serialize_tensor(tf.cast(prebatch, dtype=self.dtype)).numpy()

        features = tf.train.Features(feature={
            self.feature_name: tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized]))
        })

        example = tf.train.Example(features=features)
        self.writer.write(example.SerializeToString())
        self._batches_saved += 1

    def close(self):
        self.writer.close()


def create_writer(dst_dir, dtype, feature_name='data', multihot=False,
                  format='tfrecords', num_features=1, batches_per_file=1):
    if format == 'tfrecords':
        writer = PrebatchStreamWriter(dst_dir=dst_dir, dtype=dtype, multihot=multihot, batches_per_file=batches_per_file)
        metadata = dict(format=format, dtype=dtype.name, multihot=multihot,
                        feature_name=feature_name,num_features=num_features, batches_per_file=batches_per_file)

        with open(os.path.join(dst_dir, 'format.json'), 'w') as f:
            json.dump(metadata, f)
        return writer
    else:
        raise ValueError(f'Unknown feature format: {format}')


def create_reader(src_dir, batch_size, world_size=1, rank=0, data_parallel=True):
    with open(os.path.join(src_dir, 'format.json')) as f:
        metadata = json.load(f)

    if metadata['format'] == 'tfrecords':
        reader = SingleFeatureTFRecordsFileReader(dst_dir=src_dir, batch_size=batch_size,
                                                  dtype=tf.dtypes.as_dtype(metadata['dtype']),
                                                  multihot=metadata['multihot'],
                                                  feature_name=metadata['feature_name'],
                                                  num_features=metadata['num_features'],
                                                  world_size=world_size, rank=rank, data_parallel=data_parallel)

        return reader
    else:
        raise ValueError(f'Unknown feature format: {metadata["format"]}')


class SingleFeatureTFRecordsFileReader:
    def __init__(self, dst_dir, batch_size, dtype, rank=0, world_size=1,
                 num_features=1, feature_name='data', multihot=False,
                 data_parallel=True, parallel_calls=4):
        self.filenames = glob.glob(os.path.join(dst_dir, 'data_*.tfrecords'))
        self.feature_name = feature_name
        self.multihot = multihot
        self.batch_size = batch_size
        self.num_features = num_features
        self.dtype = dtype
        self.feature_description = {self.feature_name: tf.io.FixedLenFeature([], tf.string, default_value='')}
        self.data_parallel = data_parallel
        self.parallel_calls = parallel_calls

        self.rank = rank
        self.world_size = world_size

        if self.data_parallel:
            local_batch_size = int(self.batch_size / world_size)
            batch_sizes_per_gpu = [local_batch_size] * world_size
            indices = tuple(np.cumsum([0] + list(batch_sizes_per_gpu)))
            self.dp_begin_idx = indices[rank]
            self.dp_end_idx = indices[rank + 1]

    def __len__(self):
        pass

    def _data_parallel_split(self, x):
        return x[self.dp_begin_idx:self.dp_end_idx, ...]

    def _parse_function(self, proto):
        parsed = tf.io.parse_single_example(proto, self.feature_description)

        if self.multihot:
            rt_spec = tf.RaggedTensorSpec(dtype=tf.int32, shape=[self.batch_size, None],
                                          row_splits_dtype=tf.int32, ragged_rank=1)
            tensor = parsed[self.feature_name]
            tensor = deserialize_composite(serialized=tensor, type_spec=rt_spec)
        else:
            tensor = tf.io.parse_tensor(parsed[self.feature_name], out_type=self.dtype)
            tensor = tf.reshape(tensor, shape=[self.batch_size, self.num_features])

        if self.data_parallel:
            tensor = self._data_parallel_split(tensor)

        return tensor

    def op(self):
        num_parallel_reads = 8
        dataset = tf.data.TFRecordDataset(self.filenames, num_parallel_reads=num_parallel_reads)
        dataset = dataset.map(self._parse_function, num_parallel_calls=self.parallel_calls, deterministic=True)
        dataset = dataset.prefetch(buffer_size=1)
        dataset = dataset.repeat()
        return dataset


class SplitTFRecordsDataset:
    def __init__(self, dataset_dir, feature_ids, num_numerical, batch_size, world_size, rank):
        self.dataset_dir = dataset_dir
        self.feature_ids = feature_ids
        self.num_numerical = num_numerical
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank

        self.numerical_reader = create_reader(src_dir=os.path.join(dataset_dir, 'numerical'),
                                              world_size=world_size, rank=rank, batch_size=batch_size,
                                              data_parallel=True)

        self.label_reader = create_reader(src_dir=os.path.join(dataset_dir, 'label'),
                                          world_size=world_size, rank=rank, data_parallel=True,
                                          batch_size=batch_size)

        self.categorical_readers = []
        for feature_id in feature_ids:
            reader = create_reader(src_dir=os.path.join(dataset_dir, f'categorical_{feature_id}'),
                                   batch_size=batch_size, data_parallel=False)
            self.categorical_readers.append(reader)

        filename = length_filename(self.dataset_dir)
        with open(filename) as f:
            self.length = json.load(f)

    def __len__(self):
        return self.length

    def op(self):
        categorical_tf_datasets = tuple(d.op() for d in self.categorical_readers)
        features_datasets = (self.numerical_reader.op(), categorical_tf_datasets)
        structure_to_zip = (features_datasets, self.label_reader.op())
        dataset = tf.data.Dataset.zip(structure_to_zip)
        return dataset

    @staticmethod
    def generate(src_train, src_test, feature_spec, dst_dir, dst_feature_spec, prebatch_size, max_batches_train, max_batches_test):
        local_table_sizes = feature_spec.get_categorical_sizes()
        names = feature_spec.get_categorical_feature_names()
        local_table_hotness = [feature_spec.feature_spec[name].get('hotness', 1) for name in names]

        os.makedirs(dst_dir, exist_ok=True)
        num_files = 1

        feature_spec.to_yaml(output_path=os.path.join(dst_dir, dst_feature_spec))
        sources = [(src_train, 'train', max_batches_train), (src_test, 'test', max_batches_test)]

        for src, dst_suffix, max_batches in sources:
            num_batches = min(len(src), max_batches)
            if num_batches % num_files != 0:
                raise ValueError('The length of the dataset must be evenly divided by the number of TFRecords files')

            dst_subdir = os.path.join(dst_dir, dst_suffix)
            numerical_writer = create_writer(dst_dir=os.path.join(dst_subdir, 'numerical'), dtype=tf.float16,
                                             num_features=feature_spec.get_number_of_numerical_features(),
                                             batches_per_file=num_batches // num_files)

            label_writer = create_writer(dst_dir=os.path.join(dst_subdir, 'label'), dtype=tf.int8,
                                         batches_per_file=num_batches // num_files)

            categorical_writers = []
            for i, (hotness, cardinality) in enumerate(zip(local_table_hotness, local_table_sizes)):
                # TODO: possibly optimize the dtype by using cardinality here
                writer = create_writer(dst_dir=os.path.join(dst_subdir, f'categorical_{i}'), dtype=tf.int32,
                                       multihot=hotness > 1,
                                       batches_per_file=num_batches // num_files)
                categorical_writers.append(writer)

            with open(length_filename(dst_subdir), 'w') as f:
                json.dump(num_batches, f)

            for batch_idx, batch in tqdm.tqdm(enumerate(src.op()),
                                              total=max_batches,
                                              desc=f'Generating the {dst_suffix} data'):

                print('writing batch: ', batch_idx)
                if batch_idx == max_batches:
                    break
                print(batch_idx)
                (numerical, categorical), label = batch
                if label.shape[0] != prebatch_size:
                    raise ValueError(f'Source dataset batch size ({label.shape[0]}) '
                                     f'different from the prebatch size ({prebatch_size}). Unsupported.')
                numerical_writer.save(numerical)
                label_writer.save(label)
                for writer, feature in zip(categorical_writers, categorical):
                    writer.save(feature)

            numerical_writer.close()
            label_writer.close()
            for writer in categorical_writers:
                writer.close()
