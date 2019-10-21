# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import print_function

import os
import math

import tensorflow as tf

__all__ = ['count_steps', 'list_filenames_in_dataset', 'parse_tfrecords_dataset']


def count_steps(iter_unit, num_samples, num_iter, global_batch_size):
    
    num_samples, num_iter = map(float, (num_samples, num_iter))
    if iter_unit not in ["batch", "epoch"]:
        raise ValueError("Invalid `iter_unit` value: %s" % iter_unit)

    if iter_unit == 'epoch':
        num_steps = (num_samples // global_batch_size) * num_iter
        num_epochs = num_iter
        num_decay_steps = num_steps

    else:
        num_steps = num_iter
        num_epochs = math.ceil(num_steps / (num_samples // global_batch_size))
        num_decay_steps = 90 * num_samples // global_batch_size

    return num_steps, num_epochs, num_decay_steps


def list_filenames_in_dataset(data_dir, mode, count=True):

    if mode not in ["train", 'validation']:
        raise ValueError("Unknown mode received: %s" % mode)

    if not os.path.exists(data_dir):
        raise FileNotFoundError("The data directory: `%s` can't be found" % data_dir)

    filename_pattern = os.path.join(data_dir, '%s-*' % mode)

    file_list = sorted(tf.gfile.Glob(filename_pattern))
    num_samples = 0 
    
    if count:
        def count_records(tf_record_filename):
            count = 0
            for _ in tf.python_io.tf_record_iterator(tf_record_filename):
                count += 1
            return count

        n_files = len(file_list)
        num_samples = (count_records(file_list[0]) * (n_files - 1) + count_records(file_list[-1]))

    return file_list, num_samples


def parse_tfrecords_dataset(data_dir, mode, iter_unit, num_iter, global_batch_size):

    if data_dir is not None:
        filenames, num_samples = list_filenames_in_dataset(data_dir=data_dir, mode=mode)
    else:
        num_samples = 256000
        filenames = []

    num_steps, num_epochs, num_decay_steps = count_steps(
        iter_unit=iter_unit, num_samples=num_samples, num_iter=num_iter, global_batch_size=global_batch_size
    )

    return filenames, num_samples, num_steps, num_epochs, num_decay_steps


def parse_inference_input(to_predict):
    
    filenames = []
    
    image_formats = ['.jpg', '.jpeg', '.JPEG', '.JPG', '.png', '.PNG']
    
    if os.path.isdir(to_predict):
        filenames = [f for f in os.listdir(to_predict) 
                     if os.path.isfile(os.path.join(to_predict, f)) 
                     and os.path.splitext(f)[1] in image_formats]
        
    elif os.path.isfile(to_predict):
        filenames.append(to_predict)
      
    return filenames

def parse_dali_idx_dataset(data_idx_dir, mode):
    
    if data_idx_dir is not None:
        filenames, _ = list_filenames_in_dataset(data_dir=data_idx_dir, mode=mode, count=False)
        
    return filenames
