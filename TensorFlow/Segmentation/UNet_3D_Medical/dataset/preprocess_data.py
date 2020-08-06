# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import os
import argparse
from random import shuffle
import numpy as np

import nibabel as nib
import tensorflow as tf


PARSER = argparse.ArgumentParser()

PARSER.add_argument('--input_dir', '-i',
                    type=str, help='path to the input directory with data')

PARSER.add_argument('--output_dir', '-o',
                    type=str, help='path to the output directory where tfrecord files will be stored')

PARSER.add_argument('--verbose', '-v', dest='verbose', action='store_true', default=False)

PARSER.add_argument('--vol_per_file', default=4, dest='vol_per_file',
                    type=int, help='how many volumes to pack into a single tfrecord file')

PARSER.add_argument('--single_data_dir', dest='single_data_dir', action='store_true', default=False)


def load_features(path):
    data = np.zeros((240, 240, 155, 4), dtype=np.uint8)
    name = os.path.basename(path)
    for i, modality in enumerate(["_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz", "_flair.nii.gz"]):
        vol = load_single_nifti(os.path.join(path, name+modality)).astype(np.float32)
        vol[vol > 0.85 * vol.max()] = 0.85 * vol.max()
        vol = 255 * vol / vol.max()
        data[..., i] = vol.astype(np.uint8)

    return data


def load_segmentation(path):
    path = os.path.join(path, os.path.basename(path)) + "_seg.nii.gz"
    return load_single_nifti(path).astype(np.uint8)


def load_single_nifti(path):
    data = nib.load(path).get_fdata().astype(np.int16)
    return np.transpose(data, (1, 0, 2))


def write_to_file(features_list, labels_list, foreground_mean_list, foreground_std_list, output_dir, count):
    output_filename = os.path.join(output_dir, "volume-{}.tfrecord".format(count))
    filelist = list(zip(np.array(features_list),
                        np.array(labels_list),
                        np.array(foreground_mean_list),
                        np.array(foreground_std_list)))
    np_to_tfrecords(filelist, output_filename)


def np_to_tfrecords(filelist, output_filename):
    writer = tf.io.TFRecordWriter(output_filename)

    for idx in range(len(filelist)):
        X = filelist[idx][0].flatten().tostring()
        Y = filelist[idx][1].flatten().tostring()
        mean = filelist[idx][2].astype(np.float32).flatten()
        stdev = filelist[idx][3].astype(np.float32).flatten()

        d_feature = {}
        d_feature['X'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[X]))
        d_feature['Y'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[Y]))
        d_feature['mean'] = tf.train.Feature(float_list=tf.train.FloatList(value=mean))
        d_feature['stdev'] = tf.train.Feature(float_list=tf.train.FloatList(value=stdev))

        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()


def main():
    # parse arguments
    params = PARSER.parse_args()
    input_dir = params.input_dir
    output_dir = params.output_dir
    os.makedirs(params.output_dir, exist_ok=True)

    patient_list = []
    if params.single_data_dir:
        patient_list.extend([os.path.join(input_dir, folder) for folder in os.listdir(input_dir)])
    else:
        assert "HGG" in os.listdir(input_dir) and "LGG" in os.listdir(input_dir),\
            "Data directory has to contain folders named HGG and LGG. " \
            "If you have a single folder with patient's data please set --single_data_dir flag"
        path_hgg = os.path.join(input_dir, "HGG")
        path_lgg = os.path.join(input_dir, "LGG")
        patient_list.extend([os.path.join(path_hgg, folder) for folder in os.listdir(path_hgg)])
        patient_list.extend([os.path.join(path_lgg, folder) for folder in os.listdir(path_lgg)])
    shuffle(patient_list)

    features_list = []
    labels_list = []
    foreground_mean_list = []
    foreground_std_list = []
    count = 0

    total_tfrecord_files = len(patient_list) // params.vol_per_file + (1 if len(patient_list) % params.vol_per_file
                                                                       else 0)
    for i, folder in enumerate(patient_list):

        # Calculate mean and stdev only for foreground voxels
        features = load_features(folder)
        foreground = features > 0
        fg_mean = np.array([(features[..., i][foreground[..., i]]).mean() for i in range(features.shape[-1])])
        fg_std = np.array([(features[..., i][foreground[..., i]]).std() for i in range(features.shape[-1])])

        # BraTS labels are 0,1,2,4 -> switching to 0,1,2,3
        labels = load_segmentation(folder)
        labels[labels == 4] = 3

        features_list.append(features)
        labels_list.append(labels)
        foreground_mean_list.append(fg_mean)
        foreground_std_list.append(fg_std)

        if (i+1) % params.vol_per_file == 0:
            write_to_file(features_list, labels_list, foreground_mean_list, foreground_std_list, output_dir, count)

            # Clear lists
            features_list = []
            labels_list = []
            foreground_mean_list = []
            foreground_std_list = []
            count += 1

            if params.verbose:
                print("{}/{} tfrecord files created".format(count, total_tfrecord_files))

    # create one more file if there are any remaining unpacked volumes
    if features_list:
        write_to_file(features_list, labels_list, foreground_mean_list, foreground_std_list, output_dir, count)
        count += 1
        if params.verbose:
            print("{}/{} tfrecord files created".format(count, total_tfrecord_files))


if __name__ == '__main__':
    main()

