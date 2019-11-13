# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

from ..utils import cat

from maskrcnn_benchmark import _C as C

class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder=None,
        fpn_post_nms_top_n=None,
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = objectness.device
        N, A, H, W = objectness.shape

        num_anchors = A * H * W

        # If inputs are on GPU, use a faster path
        use_fast_cuda_path = (objectness.is_cuda and box_regression.is_cuda)
        # Encompasses box decode, clip_to_image and remove_small_boxes calls
        if use_fast_cuda_path:
            objectness = objectness.reshape(N, -1) # Now [N, AHW]
            objectness = objectness.sigmoid()

            pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
            objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

            # Get all image shapes, and cat them together
            image_shapes = [box.size for box in anchors]
            image_shapes_cat = torch.tensor([box.size for box in anchors], device=objectness.device).float()

            # Get a single tensor for all anchors
            concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)

            # Note: Take all anchors, we'll index accordingly inside the kernel
            # only take the anchors corresponding to the topk boxes
            concat_anchors = concat_anchors.reshape(N, -1, 4) # [batch_idx, topk_idx]

            # Return pre-nms boxes, associated scores and keep flag
            # Encompasses:
            # 1. Box decode
            # 2. Box clipping
            # 3. Box filtering
            # At the end we need to keep only the proposals & scores flagged
            # Note: topk_idx, objectness are sorted => proposals, objectness, keep are also
            # sorted -- this is important later
            proposals, objectness, keep = C.GeneratePreNMSUprightBoxes(
                                    N,
                                    A,
                                    H,
                                    W,
                                    topk_idx,
                                    objectness.float(),    # Need to cast these as kernel doesn't support fp16
                                    box_regression.float(),
                                    concat_anchors,
                                    image_shapes_cat,
                                    pre_nms_top_n,
                                    self.min_size,
                                    self.box_coder.bbox_xform_clip,
                                    True)


            # view as [N, pre_nms_top_n, 4]
            proposals = proposals.view(N, -1, 4)
            objectness = objectness.view(N, -1)
        else:
            # reverse the reshape from before ready for permutation
            objectness = objectness.reshape(N, A, H, W)
            objectness = objectness.permute(0, 2, 3, 1).reshape(N, -1)
            objectness = objectness.sigmoid()

            pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
            objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

            # put in the same format as anchors
            box_regression = box_regression.view(N, -1, 4, H, W).permute(0, 3, 4, 1, 2)
            box_regression = box_regression.reshape(N, -1, 4)


            batch_idx = torch.arange(N, device=device)[:, None]
            box_regression = box_regression[batch_idx, topk_idx]

            image_shapes = [box.size for box in anchors]
            concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
            concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]

            proposals = self.box_coder.decode(
                box_regression.view(-1, 4), concat_anchors.view(-1, 4)
            )

            proposals = proposals.view(N, -1, 4)

        # handle non-fast path without changing the loop
        if not use_fast_cuda_path:
            keep = [None for _ in range(N)]

        result = []
        keep = keep.to(torch.bool)
        for proposal, score, im_shape, k in zip(proposals, objectness, image_shapes, keep):
            if use_fast_cuda_path:
                # Note: Want k to be applied per-image instead of all-at-once in batched code earlier
                #       clip_to_image and remove_small_boxes already done in single kernel
                p = proposal.masked_select(k[:, None]).view(-1, 4)
                score = score.masked_select(k)
                boxlist = BoxList(p, im_shape, mode="xyxy")
            else:
                boxlist = BoxList(proposal, im_shape, mode="xyxy")
                boxlist = boxlist.clip_to_image(remove_empty=False)
                boxlist = remove_small_boxes(boxlist, self.min_size)
            boxlist.add_field("objectness", score)
            boxlist = boxlist_nms(
                boxlist,
                self.nms_thresh,
                max_proposals=self.post_nms_top_n,
                score_field="objectness",
            )
            result.append(boxlist)
        return result

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # TODO resolve this difference and make it consistent. It should be per image,
        # and not per batch
        if self.training:
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.bool)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST

    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = RPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
    )
    return box_selector
