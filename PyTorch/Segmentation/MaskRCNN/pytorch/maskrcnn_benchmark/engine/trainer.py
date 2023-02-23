# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, synchronized_timestamp
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


class Prefetcher:
    def __init__(self, data_loader, device):
        self.data_loader = iter(data_loader)
        self.device = device
        self.images = None
        self.targets = None
        self.loader_stream = torch.cuda.Stream()
        self.done = False

    def __iter__(self):
        return self

    def prefetch(self):
        try:
            with torch.cuda.stream(self.loader_stream):
                self.images, self.targets, _ = next(self.data_loader)
                self.images = self.images.to(self.device)
                self.targets = [target.to(self.device, non_blocking=True) for target in self.targets]
        except StopIteration:
            self.images, self.targets = None, None
            self.done = True

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.loader_stream)
        if self.images is None and not self.done:
            self.prefetch()
        if self.done:
            raise StopIteration()
        else:
            images, targets = self.images, self.targets
            self.images, self.targets = None, None
            return images, targets


def do_train(
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
    per_iter_end_callback_fn=None,
    nhwc=False
):
    dllogger.log(step="PARAMETER", data={"train_start": True})
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    prefetcher = Prefetcher(data_loader, device)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = synchronized_timestamp()
    end = start_training_time
    if use_amp:
        scaler = torch.cuda.amp.GradScaler(init_scale=8192.0)
    for iteration, (images, targets) in enumerate(prefetcher, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        if nhwc:
            images = images.to_nhwc()
            model = model.to(memory_format=torch.channels_last)
        targets = [target.to(device) for target in targets]

        if use_amp:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
        else:
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)


        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        if use_amp:        
            scaler.scale(losses).backward()
        else:
            losses.backward()

        def _take_step():
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        if not cfg.SOLVER.ACCUMULATE_GRAD:
            _take_step()
        else:
            if (iteration + 1) % cfg.SOLVER.ACCUMULATE_STEPS == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.div_(cfg.SOLVER.ACCUMULATE_STEPS)
                _take_step()
            
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            log_data = {"eta":eta_string, "learning_rate":optimizer.param_groups[0]["lr"],
                        "memory": torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 }
            log_data.update(meters.get_dict())
            dllogger.log(step=(iteration,), data=log_data)

        if cfg.SAVE_CHECKPOINT:
            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)

        # per-epoch work (testing)
        if per_iter_end_callback_fn is not None:
            early_exit = per_iter_end_callback_fn(iteration=iteration)
            if early_exit:
                break

    total_training_time = synchronized_timestamp() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    dllogger.log(step=tuple(), data={"e2e_train_time": total_training_time,
                                                   "train_perf_fps": max_iter * cfg.SOLVER.IMS_PER_BATCH / total_training_time})
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info(
    "Total training time: {} ({:.4f} s / it)".format(
        total_time_str, total_training_time / (max_iter)
        )
    )

