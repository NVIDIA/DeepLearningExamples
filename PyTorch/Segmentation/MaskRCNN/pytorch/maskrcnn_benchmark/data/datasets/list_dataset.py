# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Simple dataset class that wraps a list of path names
"""

from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList


class ListDataset(object):
    def __init__(self, image_lists, transforms=None):
        self.image_lists = image_lists
        self.transforms = transforms

    def __getitem__(self, item):
        img = Image.open(self.image_lists[item]).convert("RGB")

        # dummy target
        w, h = img.size
        target = BoxList([[0, 0, w, h]], img.size, mode="xyxy")

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_lists)

    def get_img_info(self, item):
        """
        Return the image dimensions for the image, without
        loading and pre-processing it
        """
        pass
