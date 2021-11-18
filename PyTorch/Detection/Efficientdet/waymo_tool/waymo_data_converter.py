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

import logging
import argparse
import glob
import tensorflow as tf
import math
import numpy as np
import itertools
import ipdb
import os
import h5py
import cv2
import sys
import json
import matplotlib.pyplot as plt
from collections import Counter
from google.cloud import storage

tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

args = None

def hash(m, n, t):
    return int(int(m)*10000000 + int(n)*100 + int(t))


def parse_args():
    global args, seg_id
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["training", "validation"], default="validation")
    parser.add_argument("--tf-dir", default="/workspace/data/waymo_tfrecords_val")
    parser.add_argument("--out-dir", default="/workspace/data/waymo_coco_format_val")
    parser.add_argument("--seg-min", default=0, type=int)
    parser.add_argument("--seg-max", default=1, type=int)
    parser.add_argument("--log-file", default="waymo-converter")
    args = parser.parse_args()

    # set starting seg id
    seg_id = args.seg_min
    return args


def setup_logging(args):
    logging.basicConfig(filename="/results/{}.log".format(args.log_file),
                        # filemode="w",
                        format="%(asctime)s:%(levelname)s:%(message)s",
                        datefmt="%m/%d/%Y %I:%M:%S %p",
                        level=logging.DEBUG)
    logging.info('Logging setup done!')


def create_dirs(args):
    # create intermediate and out directories
    os.makedirs(args.tf_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    args.images_dir = os.path.join(args.out_dir, "images")
    args.annotations_dir = os.path.join(args.out_dir, "annotations")
    os.makedirs(args.images_dir, exist_ok=True)
    os.makedirs(args.annotations_dir, exist_ok=True)
    logging.info("Created images and annotations directories: {} {}".format(
        args.images_dir, args.annotations_dir))


# set global frame and annotations id
seg_id = 0
frame_id = 0
annotation_id = 0
images_content = []
annotations_content = []

info = {
    u'description': u'COCO 2014 Dataset',
    u'url': u'http://cocodataset.org',
    u'version': u'1.0',
    u'year': 2014,
    u'contributor': u'COCO Consortium',
    u'date_created': u'2017/09/01'
}

licenses = [{
    u'url': u'http://creativecommons.org/licenses/by-nc-sa/2.0/',
    u'id': 1,
    u'name': u'Attribution-NonCommercial-ShareAlike License'
}, {
    u'url': u'http://creativecommons.org/licenses/by-nc/2.0/',
    u'id': 2,
    u'name': u'Attribution-NonCommercial License'
}, {
    u'url': u'http://creativecommons.org/licenses/by-nc-nd/2.0/',
    u'id': 3,
    u'name': u'Attribution-NonCommercial-NoDerivs License'
}, {
    u'url': u'http://creativecommons.org/licenses/by/2.0/',
    u'id': 4,
    u'name': u'Attribution License'
}, {
    u'url': u'http://creativecommons.org/licenses/by-sa/2.0/',
    u'id': 5,
    u'name': u'Attribution-ShareAlike License'
}, {
    u'url': u'http://creativecommons.org/licenses/by-nd/2.0/',
    u'id': 6,
    u'name': u'Attribution-NoDerivs License'
}, {
    u'url': u'http://flickr.com/commons/usage/',
    u'id': 7,
    u'name': u'No known copyright restrictions'
}, {
    u'url': u'http://www.usa.gov/copyright.shtml',
    u'id': 8,
    u'name': u'United States Government Work'
}]

#dataset-specific
category = [{
    u'supercategory': u'object',
    u'id': 1,
    u'name': u'vehicle'
}, {
    u'supercategory': u'object',
    u'id': 2,
    u'name': u'pedestrian'
}, {
    u'supercategory': u'object',
    u'id': 3,
    u'name': u'cyclist'
}]


# Function to convert Waymo TFrecord to COCO format
def convert(tfrecord):
    global frame_id, seg_id, annotation_id, images_content, annotations_content
    try:
        dataset = tf.data.TFRecordDataset(tfrecord, compression_type='')
        num_frames = 0
        images = []
        annotations = []
        all_labels = []
        # try:
        for data in dataset:
            frame_id += 1
            num_frames += 1
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            image_id = 1
            #           iterate across images in frame - front, side, etc.,
            for index, camera_image in enumerate(frame.images):

                output_image = tf.image.decode_jpeg(camera_image.image).numpy()

                #               iterate across labels in frame - front, side, etc.,
                for camera_labels in frame.camera_labels:
                    # Ignore camera labels that do not correspond to this camera.
                    if camera_labels.name != camera_image.name:
                        continue
                    for image_labels in camera_labels.labels:
                        #Since label 3 doesn't exist
                        if image_labels.type == 4:
                            image_labels.type = 3
                        annotations.append({
                            "image_id":
                            hash(seg_id, frame_id, image_id),
                            "area":
                            image_labels.box.width * image_labels.box.length,
                            "bbox": [
                                image_labels.box.center_x -
                                image_labels.box.length / 2.,
                                image_labels.box.center_y -
                                image_labels.box.width / 2.,
                                image_labels.box.length, image_labels.box.width
                            ],
                            "category_id":
                            image_labels.type,
                            "iscrowd":
                            0,
                            "id":
                            annotation_id
                        })
                        all_labels.append(image_labels.type)
                        annotation_id += 1

                h, w, c = output_image.shape
                plt.imsave("{}/{}_{}_{}.jpg".format(args.images_dir, seg_id, frame_id, image_id),
                        output_image,
                        cmap=None)

                images.append({
                    u'license': 1,
                    u'file_name': "{}_{}_{}.jpg".format(seg_id, frame_id, image_id),
                    u'waymo_url': None,
                    u'height': h,
                    u'width': w,
                    u'date_captured': u'2013-11-14 16:28:13',
                    u'flickr_url': None,
                    u'id': hash(seg_id, frame_id, image_id)
                })
                image_id += 1
        logging.info("Converted {} frames in {}".format(num_frames, tfrecord))
        images_content += images
        annotations_content += annotations
        logging.info("# images: {} # annotations: {}".format(
            len(images), len(annotations)))
        logging.info("# Label spread: {}".format(Counter(all_labels)))
    except:
        logging.info("Corrupted record {}".format(tfrecord))


# combine annotations, images data per segment into one annotations.json file
def combine():
    global images_content, annotations_content
    all_data = {
        "info": info,
        "images": images_content,
        "licenses": licenses,
        "annotations": annotations_content,
        "categories": category
    }
    with open("{}/annotations-{}-{}.json".format(args.annotations_dir, args.seg_min, args.seg_max), 'w') as outfile:
        json.dump(all_data, outfile)

# download waymo data
def download_and_convert(args):
    global seg_id, frame_id
    if args.dataset == "training":
        num_segs = 32
    if args.dataset == "validation":
        num_segs = 8
    logging.info("Number of segments in dataset: {}".format(num_segs))
    logging.info("Segments to process: {} to {}".format(args.seg_min, args.seg_max))

    logging.info("Creating google storage client to access waymo bucket")
    storage_client = storage.Client(project=None)
    bucket_name = "waymo_open_dataset_v_1_2_0"
    bucket = storage_client.bucket(bucket_name)

    while seg_id < args.seg_max:
        # copy from bucket
        frame_id = 0
        source_blob_name = '{dataset}/{dataset}_{:04}.tar'.format(
            seg_id, dataset=args.dataset)
        try:
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(os.path.join(args.tf_dir, "{}_{:04}.tar".format(args.dataset, seg_id)))
        except AssertionError as err:
            logging.exception(
                "Failed to download segment {}. Make sure GOOGLE_APPLICATION_CREDENTIALS is set and you have access to gs://waymo_open_dataset_v_1_2_0"
                .format(seg_id))
            sys.exit()
        logging.info("Extracting tfrecords from segment: {}_{:04}".format(args.dataset, seg_id))
        os.system("cd {}; tar -xvf {}_{:04}.tar".format(args.tf_dir, args.dataset, seg_id))
        tfrecords = glob.glob("{}/*.tfrecord".format(args.tf_dir))

        # extract data from each record
        for record_id, record in enumerate(tfrecords):
            if "with_camera_labels" in record:
                logging.info("Processing record # {}: {}".format(record_id, record))
                convert(record)
            else:
                logging.info("Skipping record # {}: {}".format(record_id, record))
            logging.info("Deleting record # {}: {}...".format(record_id, record))
            os.remove(record)
        logging.info("Processed {} records".format(len(tfrecords)))
        os.remove("{}/{}_{:04}.tar".format(args.tf_dir, args.dataset, seg_id))
        os.remove("{}/LICENSE".format(args.tf_dir))
        seg_id += 1
    # write annotations.json
    combine()


if __name__ == "__main__":

    # trigger download and conversion of Waymo data
    print("Usage: python waymo_data_converter.py --dataset <validation/training> --tf-dir <empty scratch pad dir> --out-dir <empty coco format output dir> --seg-min <0 or any starting seg id> --seg-max <32 - train, 8 - validation or any ending seg id> --log-file <name of log file which will be written to /results>")
    args = parse_args()
    setup_logging(args)
    create_dirs(args)
    logging.info("Running on dataset: {} \ntf records dir: {} \ncoco format out dir: {}".format(args.dataset, args.tf_dir, args.out_dir))
    download_and_convert(args)
