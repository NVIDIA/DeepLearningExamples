#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

"""Parameters used to build Mask-RCNN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import Namespace


class _Namespace(Namespace):
    def values(self):
        return self.__dict__


def default_config():
    return _Namespace(**dict(
        # input pre-processing parameters
        image_size=(832, 1344),
        augment_input_data=True,
        gt_mask_size=112,

        # dataset specific parameters
        num_classes=91,
        # num_classes=81,
        skip_crowd_during_training=True,
        use_category=True,

        # Region Proposal Network
        rpn_positive_overlap=0.7,
        rpn_negative_overlap=0.3,
        rpn_batch_size_per_im=256,
        rpn_fg_fraction=0.5,
        rpn_min_size=0.,

        # Proposal layer.
        batch_size_per_im=512,
        fg_fraction=0.25,
        fg_thresh=0.5,
        bg_thresh_hi=0.5,
        bg_thresh_lo=0.,

        # Faster-RCNN heads.
        fast_rcnn_mlp_head_dim=1024,
        bbox_reg_weights=(10., 10., 5., 5.),

        # Mask-RCNN heads.
        include_mask=True,  # whether or not to include mask branch.   # ===== Not existing in MLPerf ===== #
        mrcnn_resolution=28,

        # training
        train_rpn_pre_nms_topn=2000,
        train_rpn_post_nms_topn=1000,
        train_rpn_nms_threshold=0.7,

        # evaluation
        test_detections_per_image=100,
        test_nms=0.5,
        test_rpn_pre_nms_topn=1000,
        test_rpn_post_nms_topn=1000,
        test_rpn_nms_thresh=0.7,

        # model architecture
        min_level=2,
        max_level=6,
        num_scales=1,
        aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
        anchor_scale=8.0,

        # localization loss
        rpn_box_loss_weight=1.0,
        fast_rcnn_box_loss_weight=1.0,
        mrcnn_weight_loss_mask=1.0,

        # ---------- Training configurations ----------

        # Skips loading variables from the resnet checkpoint. It is used for
        # skipping nonexistent variables from the constructed graph. The list
        # of loaded variables is constructed from the scope 'resnetX', where 'X'
        # is depth of the resnet model. Supports regular expression.
        skip_checkpoint_variables='^NO_SKIP$',

        # ---------- Eval configurations ----------
        # Visualizes images and detection boxes on TensorBoard.
        visualize_images_summary=False,
    ))
