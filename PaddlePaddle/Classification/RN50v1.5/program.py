# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

import logging
import time
from profile import Profiler

import dllogger
import models
import numpy as np
from lr_scheduler import build_lr_scheduler
from optimizer import build_optimizer
from utils.misc import AverageMeter
from utils.mode import Mode, RunScope
from utils.utility import get_num_trainers

import paddle
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.distributed.fleet import DistributedStrategy
from paddle.distributed.fleet.meta_optimizers.common import CollectiveHelper
from paddle.incubate import asp as sparsity


def create_feeds(image_shape):
    """
    Create feeds mapping for the inputs of Pragrm execution.

    Args:
        image_shape(list[int]): Model input shape, such as [4, 224, 224].
    Returns:
        feeds(dict): A dict to map variables'name to their values.
                     key (string): Name of variable to feed.
                     Value (tuple): paddle.static.data.
    """
    feeds = {}
    feeds['data'] = paddle.static.data(
        name="data", shape=[None] + image_shape, dtype="float32"
    )
    feeds['label'] = paddle.static.data(
        name="label", shape=[None, 1], dtype="int64"
    )

    return feeds


def create_fetchs(out, feeds, class_num, label_smoothing=0, mode=Mode.TRAIN):
    """
    Create fetchs to obtain specific outputs from Pragrm execution (included loss and measures).

    Args:
        out(variable): The model output variable.
        feeds(dict): A dict of mapping variables'name to their values
                     (The input of Program execution).
        class_num(int): The number of classes.
        label_smoothing(float, optional): Epsilon of label smoothing. Default: 0.
        mode(utils.Mode, optional): Train or eval mode. Default: Mode.TRAIN
    Returns:
        fetchs(dict): A dict of outputs from Program execution (included loss and measures).
                      key (string): Name of variable to fetch.
                      Value (tuple): (variable, AverageMeter).
    """
    fetchs = {}
    target = paddle.reshape(feeds['label'], [-1, 1])

    if mode == Mode.TRAIN:
        if label_smoothing == 0:
            loss = F.cross_entropy(out, target)
        else:
            label_one_hot = F.one_hot(target, class_num)
            soft_target = F.label_smooth(label_one_hot, epsilon=label_smoothing)
            soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
            log_softmax = -F.log_softmax(out, axis=-1)
            loss = paddle.sum(log_softmax * soft_target, axis=-1)
    else:
        loss = F.cross_entropy(out, target)
        label = paddle.argmax(out, axis=-1, dtype='int32')
        fetchs['label'] = (label, None)

    loss = loss.mean()

    fetchs['loss'] = (loss, AverageMeter('loss', '7.4f', need_avg=True))

    acc_top1 = paddle.metric.accuracy(input=out, label=target, k=1)
    acc_top5 = paddle.metric.accuracy(input=out, label=target, k=5)
    metric_dict = {}
    metric_dict["top1"] = acc_top1
    metric_dict["top5"] = acc_top5

    for key in metric_dict:
        if mode != Mode.TRAIN and paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(
                metric_dict[key], op=paddle.distributed.ReduceOp.SUM
            )
            metric_dict[key] = (
                metric_dict[key] / paddle.distributed.get_world_size()
            )

        fetchs[key] = (
            metric_dict[key],
            AverageMeter(key, '7.4f', need_avg=True),
        )

    return fetchs


def create_strategy(args, is_train=True):
    """
    Create paddle.static.BuildStrategy and paddle.static.ExecutionStrategy with arguments.

    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        is_train(bool, optional): Indicate the prupose of strategy is for training
                                  of not. Default is True.
    Returns:
        build_strategy(paddle.static.BuildStrategy): A instance of BuildStrategy.
        exec_strategy(paddle.static.ExecutionStrategy): A instance of ExecutionStrategy.
    """
    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    exec_strategy.num_threads = 1
    exec_strategy.num_iteration_per_drop_scope = (
        10000 if args.amp and args.use_pure_fp16 else 10
    )

    paddle.set_flags(
        {
            'FLAGS_cudnn_exhaustive_search': True,
            'FLAGS_conv_workspace_size_limit': 4096,
        }
    )

    if not is_train:
        build_strategy.fix_op_run_order = True

    if args.amp:
        build_strategy.fuse_bn_act_ops = True
        build_strategy.fuse_elewise_add_act_ops = True
        build_strategy.fuse_bn_add_act_ops = True
        build_strategy.enable_addto = True
        if args.fuse_resunit and is_train:
            build_strategy.fuse_resunit = True

    return build_strategy, exec_strategy


def dist_optimizer(args, optimizer):
    """
    Create a distributed optimizer based on a given optimizer.

    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        optimizer(paddle.optimizer): A normal optimizer.
    Returns:
        optimizer(fleet.distributed_optimizer): A distributed optimizer.
    """
    build_strategy, exec_strategy = create_strategy(args)

    dist_strategy = DistributedStrategy()
    dist_strategy.execution_strategy = exec_strategy
    dist_strategy.build_strategy = build_strategy

    dist_strategy.fuse_all_reduce_ops = True
    all_reduce_size = 16
    dist_strategy.fuse_grad_size_in_MB = all_reduce_size
    dist_strategy.nccl_comm_num = 1
    dist_strategy.sync_nccl_allreduce = True

    if args.amp:
        dist_strategy.cudnn_batchnorm_spatial_persistent = True
        dist_strategy.amp = True
        dist_strategy.amp_configs = {
            "init_loss_scaling": args.scale_loss,
            "use_dynamic_loss_scaling": args.use_dynamic_loss_scaling,
            "use_pure_fp16": args.use_pure_fp16,
        }

    dist_strategy.asp = args.asp
    dist_strategy.qat = args.qat

    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)

    return optimizer


def build(args, main_prog, startup_prog, step_each_epoch, is_train=True):
    """
    Build a executable paddle.static.Program via following four steps:
        1. Create feeds.
        2. Create a model.
        3. Create fetchs.
        4. Create an optimizer if is_train==True.

    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        main_prog(paddle.static.Program):The main program.
        startup_prog(paddle.static.Program):The startup program.
        step_each_epoch(int): The number of steps in each epoch.
        is_train(bool, optional): Whether the main programe created is for training. Default: True.
    Returns:
        fetchs(dict): A dict of outputs from Program execution (included loss and measures).
        lr_scheduler(paddle.optimizer.lr.LRScheduler): A learning rate scheduler.
        feeds(dict): A dict to map variables'name to their values.
        optimizer(Optimizer): An optimizer with distributed/AMP/ASP strategy.
    """
    with paddle.static.program_guard(main_prog, startup_prog):
        with paddle.utils.unique_name.guard():
            mode = Mode.TRAIN if is_train else Mode.EVAL
            feeds = create_feeds(args.image_shape)

            model_name = args.model_arch_name
            class_num = args.num_of_class
            input_image_channel = args.image_channel
            data_format = args.data_layout
            use_pure_fp16 = args.use_pure_fp16
            bn_weight_decay = args.bn_weight_decay
            model = models.__dict__[model_name](
                class_num=class_num,
                input_image_channel=input_image_channel,
                data_format=data_format,
                use_pure_fp16=use_pure_fp16,
                bn_weight_decay=bn_weight_decay,
            )
            out = model(feeds["data"])

            fetchs = create_fetchs(
                out, feeds, class_num, args.label_smoothing, mode=mode
            )

            if args.asp:
                sparsity.set_excluded_layers(main_program=main_prog, param_names=[model.fc.weight.name])

            lr_scheduler = None
            optimizer = None
            if is_train:
                lr_scheduler = build_lr_scheduler(args, step_each_epoch)
                optimizer = build_optimizer(args, lr_scheduler)

                optimizer = dist_optimizer(args, optimizer)
                optimizer.minimize(fetchs['loss'][0], startup_prog)

    # This is a workaround to "Communicator of ring id 0 has not been initialized.".
    # Since Paddle's design, the initialization would be done inside train program,
    # eval_only need to manually call initialization.
    if (
        args.run_scope == RunScope.EVAL_ONLY
        and paddle.distributed.get_world_size() > 1
    ):
        collective_helper = CollectiveHelper(
            role_maker=fleet.PaddleCloudRoleMaker(is_collective=True)
        )
        collective_helper.update_startup_program(startup_prog)

    return fetchs, lr_scheduler, feeds, optimizer


def compile_prog(args, program, loss_name=None, is_train=True):
    """
    Compile the given program, which would fuse computing ops or optimize memory footprint
    based building strategy in config.

    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        program(paddle.static.Program): The main program to be compiled.
        loss_name(str, optional): The name of loss variable. Default: None.
        is_train(bool, optional): Indicate the prupose of strategy is for
                                  training of not. Default is True.
    Returns:
        compiled_program(paddle.static.CompiledProgram): A compiled program.
    """
    build_strategy, exec_strategy = create_strategy(args, is_train)

    compiled_program = paddle.static.CompiledProgram(
        program, build_strategy=build_strategy
    )

    return compiled_program


def run(
    args,
    dataloader,
    exe,
    program,
    fetchs,
    epoch,
    mode=Mode.TRAIN,
    lr_scheduler=None,
):
    """
    Execute program.

    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        dataloader(nvidia.dali.plugin.paddle.DALIGenericIterator):
                Iteratable output of NVIDIA DALI pipeline,
                please refer to dali_dataloader in dali.py for details.
        exe(paddle.static.Executor): A executor to run program.
        program(paddle.static.Program): The program to be executed.
        fetchs(dict): A dict of outputs from Program execution (included loss and measures).
        epoch(int): Current epoch id to run.
        mode(utils.Mode, optional): Train or eval mode. Default: Mode.TRAIN.
        lr_scheduler(paddle.optimizer.lr.LRScheduler, optional): A learning rate scheduler.
                                                                 Default: None.
    Returns:
        metrics (dict): A dictionary to collect values of metrics.
    """
    num_trainers = get_num_trainers()
    fetch_list = [f[0] for f in fetchs.values()]
    metric_dict = {"lr": AverageMeter('lr', 'f', postfix=",", need_avg=False)}

    for k in fetchs:
        if fetchs[k][1] is not None:
            metric_dict[k] = fetchs[k][1]

    metric_dict["batch_time"] = AverageMeter('batch_time', '.5f', postfix=" s,")
    metric_dict["data_time"] = AverageMeter('data_time', '.5f', postfix=" s,")
    metric_dict["compute_time"] = AverageMeter(
        'compute_time', '.5f', postfix=" s,"
    )

    for m in metric_dict.values():
        m.reset()

    profiler = Profiler()
    tic = time.perf_counter()

    idx = 0
    batch_size = None
    latency = []

    total_benchmark_steps = args.benchmark_steps + args.benchmark_warmup_steps

    dataloader.reset()
    while True:
        # profiler.profile_setup return True only when
        # profile is enable and idx == stop steps
        if profiler.profile_setup(idx):
            break

        idx += 1
        try:
            batch = next(dataloader)
        except StopIteration:
            # Reset dataloader when run benchmark to fill required steps.
            if args.benchmark and (idx < total_benchmark_steps):
                dataloader.reset()
                # Reset tic timestamp to ignore exception handling time.
                tic = time.perf_counter()
                continue
            break
        except RuntimeError:
            logging.warning(
                "Except RuntimeError when reading data from dataloader, try to read once again..."
            )
            continue

        reader_toc = time.perf_counter()
        metric_dict['data_time'].update(reader_toc - tic)

        batch_size = batch[0]["data"].shape()[0]
        feed_dict = batch[0]

        with profiler.profile_tag(
            idx, "Training" if mode == Mode.TRAIN else "Evaluation"
        ):
            results = exe.run(
                program=program, feed=feed_dict, fetch_list=fetch_list
            )

        for name, m in zip(fetchs.keys(), results):
            if name in metric_dict:
                metric_dict[name].update(np.mean(m), batch_size)
        metric_dict["compute_time"].update(time.perf_counter() - reader_toc)
        metric_dict["batch_time"].update(time.perf_counter() - tic)
        if mode == Mode.TRAIN:
            metric_dict['lr'].update(lr_scheduler.get_lr())

        if lr_scheduler is not None:
            with profiler.profile_tag(idx, "LR Step"):
                lr_scheduler.step()

        tic = time.perf_counter()

        if idx % args.print_interval == 0:
            log_msg = {}
            log_msg['loss'] = metric_dict['loss'].val.item()
            log_msg['top1'] = metric_dict['top1'].val.item()
            log_msg['top5'] = metric_dict['top5'].val.item()
            log_msg['data_time'] = metric_dict['data_time'].val
            log_msg['compute_time'] = metric_dict['compute_time'].val
            log_msg['batch_time'] = metric_dict['batch_time'].val
            log_msg['ips'] = (
                batch_size * num_trainers / metric_dict['batch_time'].val
            )
            if mode == Mode.TRAIN:
                log_msg['lr'] = metric_dict['lr'].val
            log_info((epoch, idx), log_msg, mode)

        if args.benchmark:
            latency.append(metric_dict['batch_time'].val)
            # Ignore the warmup iters
            if idx == args.benchmark_warmup_steps:
                metric_dict["compute_time"].reset()
                metric_dict["data_time"].reset()
                metric_dict["batch_time"].reset()
                latency.clear()
                logging.info("Begin benchmark at step %d", idx + 1)

            if idx == total_benchmark_steps:
                benchmark_data = {}
                benchmark_data['ips'] = (
                    batch_size * num_trainers / metric_dict['batch_time'].avg
                )
                if mode == mode.EVAL:
                    latency = np.array(latency) * 1000
                    quantile = np.quantile(latency, [0.9, 0.95, 0.99])

                    benchmark_data['latency_avg'] = np.mean(latency)
                    benchmark_data['latency_p90'] = quantile[0]
                    benchmark_data['latency_p95'] = quantile[1]
                    benchmark_data['latency_p99'] = quantile[2]

                logging.info("End benchmark at epoch step %d", idx)
                return benchmark_data

    epoch_data = {}
    epoch_data['loss'] = metric_dict['loss'].avg.item()
    epoch_data['epoch_time'] = metric_dict['batch_time'].total
    epoch_data['ips'] = (
        batch_size
        * num_trainers
        * metric_dict["batch_time"].count
        / metric_dict["batch_time"].sum
    )
    if mode == Mode.EVAL:
        epoch_data['top1'] = metric_dict['top1'].avg.item()
        epoch_data['top5'] = metric_dict['top5'].avg.item()
    log_info((epoch,), epoch_data, mode)

    return epoch_data


def log_info(step, metrics, mode):
    """
    Log metrics with step and mode information.

    Args:
        step(tuple): Step, coulbe (epoch-id, iter-id). Use tuple() for summary.
        metrics(dict): A dictionary collected values of metrics.
        mode(utils.Mode): Train or eval mode.
    """
    prefix = 'train' if mode == Mode.TRAIN else 'val'
    dllogger_iter_data = {}
    for key in metrics:
        dllogger_iter_data[f"{prefix}.{key}"] = metrics[key]
    dllogger.log(step=step, data=dllogger_iter_data)
