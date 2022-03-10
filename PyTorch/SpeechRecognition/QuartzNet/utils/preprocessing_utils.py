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

#!/usr/bin/env python
import os
import multiprocessing
import functools

import sox


from tqdm import tqdm

def preprocess(data, input_dir, dest_dir, target_sr=None, speed=None,
               overwrite=True):
    speed = speed or []
    speed.append(1)
    speed = list(set(speed))  # Make uniqe

    input_fname = os.path.join(input_dir,
                               data['input_relpath'],
                               data['input_fname'])
    input_sr = sox.file_info.sample_rate(input_fname)
    target_sr = target_sr or input_sr

    os.makedirs(os.path.join(dest_dir, data['input_relpath']), exist_ok=True)

    output_dict = {}
    output_dict['transcript'] = data['transcript'].lower().strip()
    output_dict['files'] = []

    fname = os.path.splitext(data['input_fname'])[0]
    for s in speed:
        output_fname = fname + '{}.wav'.format('' if s==1 else '-{}'.format(s))
        output_fpath = os.path.join(dest_dir,
                                    data['input_relpath'],
                                    output_fname)

        if not os.path.exists(output_fpath) or overwrite:
            cbn = sox.Transformer().speed(factor=s).convert(target_sr)
            cbn.build(input_fname, output_fpath)

        file_info = sox.file_info.info(output_fpath)
        file_info['fname'] = os.path.join(os.path.basename(dest_dir),
                                          data['input_relpath'],
                                          output_fname)
        file_info['speed'] = s
        output_dict['files'].append(file_info)

        if s == 1:
            file_info = sox.file_info.info(output_fpath)
            output_dict['original_duration'] = file_info['duration']
            output_dict['original_num_samples'] = file_info['num_samples']

    return output_dict


def parallel_preprocess(dataset, input_dir, dest_dir, target_sr, speed, overwrite, parallel):
    with multiprocessing.Pool(parallel) as p:
        func = functools.partial(preprocess,
            input_dir=input_dir, dest_dir=dest_dir,
            target_sr=target_sr, speed=speed, overwrite=overwrite)
        dataset = list(tqdm(p.imap(func, dataset), total=len(dataset)))
        return dataset
