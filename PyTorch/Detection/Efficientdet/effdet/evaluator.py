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

import torch
import torch.distributed as dist
import abc
import json
from .distributed import synchronize, is_main_process, all_gather_container
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
import numpy as np
import itertools

def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table


class Evaluator:

    def __init__(self):
        pass

    @abc.abstractmethod
    def add_predictions(self, output, target):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass


class COCOEvaluator(Evaluator):

    def __init__(self, coco_api, distributed=False, waymo=False):
        super().__init__()
        self.coco_api = coco_api
        self.distributed = distributed
        self.distributed_device = None
        self.img_ids = []
        self.predictions = []
        self.waymo = waymo

    def reset(self):
        self.img_ids = []
        self.predictions = []

    def add_predictions(self, detections, target):
        if self.distributed:
            if self.distributed_device is None:
                # cache for use later to broadcast end metric
                self.distributed_device = detections.device
            synchronize()
            detections = all_gather_container(detections)
            #target = all_gather_container(target)
            sample_ids = all_gather_container(target['img_id'])
            if not is_main_process():
                return
        else:
            sample_ids = target['img_id']

        detections = detections.cpu()
        sample_ids = sample_ids.cpu()
        for index, sample in enumerate(detections):
            image_id = int(sample_ids[index])
            for det in sample:
                score = float(det[4])
                if score < .001:  # stop when below this threshold, scores in descending order
                    break
                coco_det = dict(
                    image_id=image_id,
                    bbox=det[0:4].tolist(),
                    score=score,
                    category_id=int(det[5]))
                self.img_ids.append(image_id)
                self.predictions.append(coco_det)

    def evaluate(self):
        if not self.distributed or dist.get_rank() == 0:
            assert len(self.predictions)
            json.dump(self.predictions, open('./temp.json', 'w'), indent=4)
            results = self.coco_api.loadRes('./temp.json')
            coco_eval = COCOeval(self.coco_api, results, 'bbox')
            coco_eval.params.imgIds = self.img_ids  # score only ids we've used
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            metric = coco_eval.stats[0]  # mAP 0.5-0.95
            if self.waymo:
                results = self._derive_coco_results(coco_eval, iou_type="bbox", class_names=['Vehicle', 'Pedestrian', 'Cyclist'])
            if self.distributed:
                dist.broadcast(torch.tensor(metric, device=self.distributed_device), 0)
        else:
            metric = torch.tensor(0, device=self.distributed_device)
            dist.broadcast(metric, 0)
            metric = metric.item()
        self.reset()
        return metric


    def save_predictions(self, file_path):
        if not self.distributed or dist.get_rank() == 0:
            assert len(self.predictions)
            json.dump(self.predictions, open(file_path, 'w'), indent=4)

    
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            print("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        print(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            print("Note that some metrics cannot be computed.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        print("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})

        # get index for threshold closest to coco api iouThrs
        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                        (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        # Per category waymo eval
        waymo_results_per_category = []
        # For waymo evaluation, we find AP at specific IoUs for each object
        # Vehicle @ IoU 0.7, Pedestrian/Cyclist @ IoU 0.5
        # IoU thresholds defined in coco api:
        # iouThrs = np.array([0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95])
        thresholds = [.7, .5, .5]
        threshold_ids = [_get_thr_ind(coco_eval, thr) for thr in thresholds]
        mean_precision = np.array([])
        for idx, name in enumerate(class_names):
            # get precision at specific iouThr
            precision = precisions[threshold_ids[idx], :, idx, 0, -1]
            # precision for a specific category and specific iou threshold
            precision = precision[precision > -1]
            mean_precision = np.append(mean_precision, precision)
            ap = np.mean(precision) if precision.size else float("nan")
            waymo_results_per_category.append(("{}".format(name), float(ap * 100)))
        # compute mAP (Waymo evaluation format
        # AP (all categories) 
        # L2 (easy + hard detections) 
        # ALL_NS (all categories except stop signs))
        ap = np.mean(mean_precision) if mean_precision.size else float("nan")
        waymo_results_per_category = [("L2_ALL_NS", float(ap * 100))] + waymo_results_per_category

        # tabulate waymo evaluation results
        results_flatten = list(itertools.chain(*waymo_results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::len(results_flatten)] for i in range(len(results_flatten))])
        headers = [("category", "mAP")] + \
            [("category", "AP @ IoU {}".format(coco_eval.params.iouThrs[threshold_ids[i]]))
            for i in range(len(threshold_ids))]                                     
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=list(itertools.chain(*headers)),
            numalign="left",
        )
        print("Waymo Evaluation: {} AP: \n".format(iou_type) + table)
        results.update({"WaymoAP" + name: ap for name, ap in waymo_results_per_category})

        return results



class FastMapEvalluator(Evaluator):

    def __init__(self, distributed=False):
        super().__init__()
        self.distributed = distributed
        self.predictions = []

    def add_predictions(self, output, target):
        pass

    def evaluate(self):
        pass