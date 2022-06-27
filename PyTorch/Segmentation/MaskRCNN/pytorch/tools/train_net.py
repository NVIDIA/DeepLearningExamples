# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import logging
import functools

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.engine.tester import test
from maskrcnn_benchmark.utils.logger import format_step
#from dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
#import dllogger as DLLogger
import dllogger
import torch.utils.tensorboard as tbx

from maskrcnn_benchmark.utils.logger import format_step

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp

try:
    from apex.parallel import DistributedDataParallel as DDP
    use_apex_ddp = True
except ImportError:
    print('Use APEX for better performance')
    use_apex_ddp = False

def test_and_exchange_map(tester, model, distributed):
    results = tester(model=model, distributed=distributed)

    # main process only
    if is_main_process():
        # Note: one indirection due to possibility of multiple test datasets, we only care about the first
        #       tester returns (parsed results, raw results). In our case, don't care about the latter
        map_results, raw_results = results[0]
        bbox_map = map_results.results["bbox"]['AP']
        segm_map = map_results.results["segm"]['AP']
    else:
        bbox_map = 0.
        segm_map = 0.

    if distributed:
        map_tensor = torch.tensor([bbox_map, segm_map], dtype=torch.float32, device=torch.device("cuda"))
        torch.distributed.broadcast(map_tensor, 0)
        bbox_map = map_tensor[0].item()
        segm_map = map_tensor[1].item()

    return bbox_map, segm_map

def mlperf_test_early_exit(iteration, iters_per_epoch, tester, model, distributed, min_bbox_map, min_segm_map):
    if iteration > 0 and iteration % iters_per_epoch == 0:
        epoch = iteration // iters_per_epoch

        dllogger.log(step="PARAMETER", data={"eval_start": True})

        bbox_map, segm_map = test_and_exchange_map(tester, model, distributed)

        # necessary for correctness
        model.train()
        dllogger.log(step=(iteration, epoch, ), data={"BBOX_mAP": bbox_map, "MASK_mAP": segm_map})
        # terminating condition
        if bbox_map >= min_bbox_map and segm_map >= min_segm_map:
            dllogger.log(step="PARAMETER", data={"target_accuracy_reached": True})
            return True

    return False


def train(cfg, local_rank, distributed, fp16, dllogger):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    use_amp = False
    if fp16:
        use_amp = True
    else:
        use_amp = cfg.DTYPE == "float16"

    if distributed:
        if cfg.USE_TORCH_DDP or not use_apex_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
            )
        else:
            model = DDP(model, delay_allreduce=True)

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader, iters_per_epoch = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    # set the callback function to evaluate and potentially
    # early exit each epoch
    if cfg.PER_EPOCH_EVAL:
        per_iter_callback_fn = functools.partial(
                mlperf_test_early_exit,
                iters_per_epoch=iters_per_epoch,
                tester=functools.partial(test, cfg=cfg, dllogger=dllogger),
                model=model,
                distributed=distributed,
                min_bbox_map=cfg.MIN_BBOX_MAP,
                min_segm_map=cfg.MIN_MASK_MAP)
    else:
        per_iter_callback_fn = None

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        use_amp,
        cfg,
        dllogger,
        per_iter_end_callback_fn=per_iter_callback_fn,
        nhwc=cfg.NHWC
    )

    return model, iters_per_epoch

def test_model(cfg, model, distributed, iters_per_epoch, dllogger):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    results = []
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        result = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            dllogger=dllogger,
        )
        synchronize()
        results.append(result)
    if is_main_process(): 
        map_results, raw_results = results[0]
        bbox_map = map_results.results["bbox"]['AP']
        segm_map = map_results.results["segm"]['AP']
        dllogger.log(step=(cfg.SOLVER.MAX_ITER, cfg.SOLVER.MAX_ITER / iters_per_epoch,), data={"BBOX_mAP": bbox_map, "MASK_mAP": segm_map})
        dllogger.log(step=tuple(), data={"BBOX_mAP": bbox_map, "MASK_mAP": segm_map})

def main():

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument("--max_steps", type=int, default=0, help="Override number of training steps in the config")
    parser.add_argument("--skip-test", dest="skip_test", help="Do not test the final model",
                        action="store_true",)
    parser.add_argument("--fp16", help="Mixed precision training", action="store_true")
    parser.add_argument("--amp", help="Mixed precision training", action="store_true")
    parser.add_argument('--skip_checkpoint', default=False, action='store_true', help="Whether to save checkpoints")
    parser.add_argument("--json-summary", help="Out file for DLLogger", default="dllogger.out",
                        type=str,
                        )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    args.fp16 = args.fp16 or args.amp
    
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Redundant option - Override config parameter with command line input
    if args.max_steps > 0:
        cfg.SOLVER.MAX_ITER = args.max_steps

    if args.skip_checkpoint:
        cfg.SAVE_CHECKPOINT = False
        
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    if is_main_process():
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                filename=args.json_summary),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    else:
        dllogger.init(backends=[])

    dllogger.metadata("BBOX_mAP", {"unit": None})
    dllogger.metadata("MASK_mAP", {"unit": None})
    dllogger.metadata("e2e_train_time", {"unit": "s"})
    dllogger.metadata("train_perf_fps", {"unit": "images/s"})

    dllogger.log(step="PARAMETER", data={"gpu_count":num_gpus})
    # dllogger.log(step="PARAMETER", data={"environment_info": collect_env_info()})
    dllogger.log(step="PARAMETER", data={"config_file": args.config_file})

    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()

    dllogger.log(step="PARAMETER", data={"config":cfg})
    
    if args.fp16:
        fp16 = True
    else:
        fp16 = False

    model, iters_per_epoch = train(cfg, args.local_rank, args.distributed, fp16, dllogger)

    if not args.skip_test:
        if not cfg.PER_EPOCH_EVAL:
            test_model(cfg, model, args.distributed, iters_per_epoch, dllogger)


if __name__ == "__main__":
    main()
    dllogger.log(step=tuple(), data={})
    dllogger.flush()
