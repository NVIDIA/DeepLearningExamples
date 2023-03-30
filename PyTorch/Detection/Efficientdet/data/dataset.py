""" COCO dataset (quick and dirty)

Hacked together by Ross Wightman
"""

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import os
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from effdet.anchors import Anchors, AnchorLabeler

class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``

    """

    def __init__(self, root, ann_file, config, transform=None):
        super(CocoDetection, self).__init__()
        if isinstance(root, (str, bytes)):
            root = os.path.expanduser(root)
        self.root = root
        self.transform = transform
        self.yxyx = True   # expected for TF model, most PT are xyxy
        self.include_masks = False
        self.include_bboxes_ignore = False
        self.has_annotations = 'image_info' not in ann_file
        self.coco = None
        self.cat_ids = []
        self.cat_to_label = dict()
        self.img_ids = []
        self.img_ids_invalid = []
        self.img_infos = []
        self._load_annotations(ann_file)
        self.anchors = Anchors(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)
        self.anchor_labeler = AnchorLabeler(self.anchors, config.num_classes, match_threshold=0.5)

    def _load_annotations(self, ann_file):
        assert self.coco is None
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        img_ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for img_id in sorted(self.coco.imgs.keys()):
            info = self.coco.loadImgs([img_id])[0]
            valid_annotation = not self.has_annotations or img_id in img_ids_with_ann
            if valid_annotation and min(info['width'], info['height']) >= 32:
                self.img_ids.append(img_id)
                self.img_infos.append(info)
            else:
                self.img_ids_invalid.append(img_id)

    def _parse_img_ann(self, img_id, img_info):
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        bboxes = []
        bboxes_ignore = []
        cls = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if self.include_masks and ann['area'] <= 0:
                continue
            if w < 1 or h < 1:
                continue

            # To subtract 1 or not, TF doesn't appear to do this so will keep it out for now.
            if self.yxyx:
                #bbox = [y1, x1, y1 + h - 1, x1 + w - 1]
                bbox = [y1, x1, y1 + h, x1 + w]
            else:
                #bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
                bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                if self.include_bboxes_ignore:
                    bboxes_ignore.append(bbox)
            else:
                bboxes.append(bbox)
                cls.append(self.cat_to_label[ann['category_id']] if self.cat_to_label else ann['category_id'])

        if bboxes:
            bboxes = np.array(bboxes, dtype=np.float32)
            cls = np.array(cls, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.array([], dtype=np.int64)

        if self.include_bboxes_ignore:
            if bboxes_ignore:
                bboxes_ignore = np.array(bboxes_ignore, dtype=np.float32)
            else:
                bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(img_id=img_id, bbox=bboxes, cls=cls, img_size=(img_info['width'], img_info['height']))

        if self.include_bboxes_ignore:
            ann['bbox_ignore'] = bboxes_ignore

        return ann

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """
        img_id = self.img_ids[index]
        img_info = self.img_infos[index]
        if self.has_annotations:
            ann = self._parse_img_ann(img_id, img_info)
        else:
            ann = dict(img_id=img_id, img_size=(img_info['width'], img_info['height']))

        path = img_info['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img, ann = self.transform(img, ann)

        cls_targets, box_targets, num_positives = self.anchor_labeler.label_anchors(
            ann['bbox'], ann['cls'])
        ann.pop('bbox')
        ann.pop('cls')
        ann['num_positives'] = num_positives
        ann.update(cls_targets)
        ann.update(box_targets)

        return img, ann

    def __len__(self):
        return len(self.img_ids)
