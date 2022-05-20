# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np


def generate_foreground_bounding_box(img):
    """
    Generate the spatial bounding box of foreground in the image with start-end positions.
    Foreground is defined by positive intensity across channels.
    The output format of the coordinates is:
        [1st_spatial_dim_start, 2nd_spatial_dim_start, ..., Nth_spatial_dim_start],
        [1st_spatial_dim_end, 2nd_spatial_dim_end, ..., Nth_spatial_dim_end]
    The bounding boxes edges are aligned with the input image edges.
    This function returns [0, 0, ...], [0, 0, ...] if there's no positive intensity.
    Args:
        img: source image to generate bounding box from.
    """
    data = np.any(img > 0, axis=0)
    ndim = len(data.shape)

    if not data.any():
        return [0] * ndim, [0] * ndim
    else:
        indices = np.where(data)
        box_start = [ax.min() for ax in indices]
        box_end = [ax.max() + 1 for ax in indices]
        return box_start, box_end


def spatial_crop(img, box_start, box_end):
    slices = [slice(s, e) for s, e in zip(box_start, box_end)]
    sd = min(len(slices), len(img.shape[1:]))
    slices = [slice(None)] + slices[:sd]
    return img[tuple(slices)]


def crop_foreground(image, label=None):
    box_start, box_end = generate_foreground_bounding_box(image)
    box_start = np.asarray(box_start, dtype=np.int16)
    box_end = np.asarray(box_end, dtype=np.int16)
    image_cropped = spatial_crop(image, box_start, box_end)
    label_cropped = spatial_crop(label, box_start, box_end) if label is not None else None
    return image_cropped, label_cropped, (box_start, box_end)


def _normalize(img, nonzero, eps=1e-7):
    slices = (img != 0) if nonzero else np.ones(img.shape, dtype=bool)
    if not np.any(slices):
        return img

    sub = np.mean(img[slices])
    div = np.std(img[slices])
    if div == 0.0:
        div = eps
    img[slices] = (img[slices] - sub) / div
    return img


def normalize_intensity(img, nonzero=True, channel_wise=True):
    if channel_wise:
        for i, d in enumerate(img):
            img[i] = _normalize(d, nonzero=nonzero)
    else:
        img = _normalize(img, nonzero=nonzero)
    return img.astype(np.float32)
