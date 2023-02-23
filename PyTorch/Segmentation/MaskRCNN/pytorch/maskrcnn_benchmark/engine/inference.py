# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import datetime
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, all_gather, synchronize, synchronized_timestamp


def compute_on_dataset(model, data_loader, device, steps=-1):
    model.eval()
    results_dict = {}
    latency = []
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        #Break earlier for inference on partial dataset
        if steps > -1 and i >= steps:
            break
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            batch_start = time.perf_counter()
            output = model(images)
            latency.append(time.perf_counter() - batch_start)
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict, latency


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        skip_eval=False,
        dllogger=None,
        steps=-1,
        profile=False,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    dataset = data_loader.dataset
    dllogger.log(step="PARAMETER", data={"eval_dataset_name": dataset_name, "eval_num_samples":len(dataset)})
    start_time = synchronized_timestamp()
    with torch.autograd.profiler.emit_nvtx(enabled=profile):
        predictions, latency = compute_on_dataset(model, data_loader, device, steps=steps)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    latency_avg = sum(latency) / len(latency)
    latency.sort()
    def _latency_avg(n):
        return sum(latency[:n]) / n 
    latency_90 = _latency_avg(int(len(latency)*0.9))
    latency_95 = _latency_avg(int(len(latency)*0.95))
    latency_99 = _latency_avg(int(len(latency)*0.99))
    len_dataset = len(dataset) if steps is -1 else steps
    total_time_str = str(datetime.timedelta(seconds=total_time))
    dllogger.log(step=tuple(), data={"e2e_infer_time": total_time, "inference_perf_fps": len_dataset / total_time})
    stats = {'latency_avg' : latency_avg, 'latency_90': latency_90,
            'latency_95' : latency_95, 'latency_99': latency_99,}
    dllogger.log(step=tuple(), data=stats)
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info(
    "Total inference time: {} ({} s / img per device, on {} devices)".format(
        total_time_str, total_time * num_devices / len_dataset, num_devices
        )
    )


    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    if skip_eval:
        dllogger.log(step="PARAMETER", data={"skip_eval":True, "predictions_saved_path":os.path.join(output_folder, "predictions.pth")})
        return
        
    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
