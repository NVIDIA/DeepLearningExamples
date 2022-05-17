# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import os
import torch
import torchvision
import torch.multiprocessing as mp

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def build_target(self, anno, img_size, pin_memory=False):
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.tensor(boxes, dtype=torch.float32, pin_memory=pin_memory).reshape(-1, 4) # guard against no boxes
        target = BoxList(boxes, img_size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes, dtype=torch.float32, pin_memory=pin_memory)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img_size, pin_memory=pin_memory)
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)
        return target

    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.get_raw_img_info(idx), torchvision.io.image.ImageReadMode.RGB)
        target = self.get_target(idx)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def get_raw_img_info(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return os.path.join(self.root, path)

    def get_target(self, index, pin_memory=False):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)
        img_size = (self.coco.imgs[img_id]["width"], self.coco.imgs[img_id]["height"])
        return self.build_target(anno, img_size, pin_memory=pin_memory)


class HybridDataLoader(object):
    def __init__(self, cfg, is_train, batch_size, batch_sampler, dataset, collator, transforms, size_divisible):
        assert(dataset._transforms is None), "dataset._transforms must be None when hybrid dataloader is selected"
        self.batch_size = batch_size
        self.length = len(batch_sampler)
        self.batch_sampler = iter(batch_sampler)
        self.dataset = dataset
        self.transforms = transforms
        self.size_divisible = size_divisible

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        images, targets, idxs = [], [], []
        for idx in next(self.batch_sampler):
            raw_image = torchvision.io.read_image(self.dataset.get_raw_img_info(idx), torchvision.io.image.ImageReadMode.RGB).pin_memory().to(device='cuda', non_blocking=True)
            raw_target = self.dataset.get_target(idx, pin_memory=True)
            image, target = self.transforms(raw_image, raw_target)
            images.append( image )
            targets.append( target )
            idxs.append( idx )
        images = to_image_list(images, self.size_divisible)
        return images, targets, idxs