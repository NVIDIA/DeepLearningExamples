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

from concurrent.futures import ThreadPoolExecutor
import os
import time
import logging
import shutil
import numpy as np
import paddle
import paddle.distributed.fleet as fleet
from modeling import BertForPretraining, BertConfig
from loss import BertPretrainingCriterion
from utils.save_load import save_model
from utils.utility import get_num_trainers, get_trainer_id
from lr_scheduler import build_lr_scheduler
from optimizer import build_optimizer
from pretraining_dataset import create_pretraining_dataset, select_dataset_file_for_each_worker, WorkerInitObj
import dllogger


def create_strategy(use_distributed_fused_lamb=False):
    """
    Create paddle.static.BuildStrategy and paddle.static.ExecutionStrategy with arguments.

    Args:
        use_distributed_fused_lamb(bool, optional): Whether to use distributed fused lamb.
    Returns:
        build_strategy(paddle.static.BuildStrategy): A instance of BuildStrategy.
        exec_strategy(paddle.static.ExecutionStrategy): A instance of ExecutionStrategy.
    """
    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    build_strategy.enable_addto = True

    if use_distributed_fused_lamb:
        build_strategy.fuse_all_reduce_ops = False
        build_strategy.reduce_strategy = paddle.static.BuildStrategy.ReduceStrategy._NoReduce
    else:
        build_strategy.fuse_all_reduce_ops = True
        build_strategy.reduce_strategy = paddle.static.BuildStrategy.ReduceStrategy.AllReduce

    exec_strategy.num_threads = 1
    exec_strategy.num_iteration_per_drop_scope = 10000

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
    use_distributed_fused_lamb = True if args.optimizer == 'DistributedFusedLamb' else False
    build_strategy, exec_strategy = create_strategy(use_distributed_fused_lamb)
    dist_strategy = fleet.DistributedStrategy()

    if use_distributed_fused_lamb:
        dist_strategy.gradient_scale_configs = {'scale_strategy': 'sum'}

    dist_strategy.execution_strategy = exec_strategy
    dist_strategy.build_strategy = build_strategy

    if use_distributed_fused_lamb:
        dist_strategy.fuse_all_reduce_ops = False
    else:
        dist_strategy.fuse_all_reduce_ops = True

    dist_strategy.fuse_grad_size_in_MB = 0
    if args.amp:
        dist_strategy.amp = True
        custom_white_list = ['softmax', 'layer_norm', 'gelu']
        custom_black_list = ['lookup_table',
                             'lookup_table_v2'] if args.use_pure_fp16 else None
        dist_strategy.amp_configs = {
            'custom_white_list': custom_white_list,
            'custom_black_list': custom_black_list,
            'init_loss_scaling': args.scale_loss,
            'use_dynamic_loss_scaling': True,
            'incr_every_n_steps': 2000,
            'decr_every_n_nan_or_inf': 1,
            'incr_ratio': 2.0,
            'decr_ratio': 0.5,
            'use_pure_fp16': args.use_pure_fp16,
            'use_fp16_guard': args.use_pure_fp16
        }
    if not use_distributed_fused_lamb and args.gradient_merge_steps > 1:
        dist_strategy.gradient_merge = True
        dist_strategy.gradient_merge_configs = {
            'k_steps': args.gradient_merge_steps
        }

    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
    return optimizer


def build(args, main_prog, startup_prog, feeds, is_train=True):
    """
    Build a executable paddle.static.Program via following 3 steps:
        1. Create model.
        2. Create loss.
        3. Create optimizer if is_train==True.

    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        main_prog(paddle.static.Program):The main program.
        startup_prog(paddle.static.Program):The startup program.
        feeds(dict): A dict of mapping variables' names to their values
        is_train(bool, optional): Whether the main programe created is for training. Default: True.
    Returns:
        model(paddle.nn.Layer): An instance of BERT Model defined in modeling.py.
        lr_scheduler(paddle.optimizer.lr.LRScheduler): A learning rate scheduler.
        optimizer(Optimizer): An optimizer with distributed/AMP strategy.
        loss(variable): The output variable of loss function.
    """

    with paddle.static.program_guard(main_prog, startup_prog):
        with paddle.utils.unique_name.guard():
            [
                input_ids, segment_ids, input_mask, masked_lm_positions,
                masked_lm_labels, next_sentence_labels, masked_lm_scale
            ] = feeds
            bert_config = BertConfig.from_json_file(args.config_file)
            if bert_config.vocab_size % 8 != 0:
                bert_config.vocab_size += 8 - (bert_config.vocab_size % 8)
            model = BertForPretraining(bert_config)
            criterion = BertPretrainingCriterion(bert_config.vocab_size)
            prediction_scores, seq_relationship_score = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                masked_positions=masked_lm_positions)
            loss = criterion(prediction_scores, seq_relationship_score,
                             masked_lm_labels, next_sentence_labels,
                             masked_lm_scale)

            lr_scheduler = None
            optimizer = None
            if is_train:
                lr_scheduler = build_lr_scheduler(args)
                optimizer = build_optimizer(args, lr_scheduler)
                optimizer = dist_optimizer(args, optimizer)
                optimizer.minimize(loss)
        return model, lr_scheduler, optimizer, loss


def run(exe, program, args, lr_scheduler, loss, feeds, progress=None):
    """
    Execute program.

    Args:
        exe(paddle.static.Executor): A executor to run program.
        program(paddle.static.Program): The program to be executed.
        args(Namespace): Arguments obtained from ArgumentParser.
        lr_scheduler(paddle.optimizer.lr.LRScheduler): A learning rate scheduler.
                                                                 Default: None.
        loss(variable): The output variable of loss function.
        feeds(dict): A dict of mapping variables' names to their values
        progress(dict, optional): A dict to record the training progress of checkpoint.
    Returns:
        global_step(int): Final step id of this run.
        loss_return(float): Final loss of this run.
        train_time_raw(float): Time to train of this run.
    """
    pool = ThreadPoolExecutor(1)

    num_trainers = get_num_trainers()
    trainer_id = get_trainer_id()

    worker_init = WorkerInitObj(args.seed + trainer_id)

    batch_size_per_gpu = args.batch_size
    log_steps = args.log_freq
    save_steps = args.num_steps_per_checkpoint
    gradient_merge_steps = args.gradient_merge_steps

    most_recent_ckpts_paths = []
    last_step = args.last_step_of_checkpoint
    train_iter = 0
    epoch = 0
    resume_from_ckpt = False
    if progress is None:
        progress = dict()
    else:
        resume_from_ckpt = True
        last_step = progress.get('global_step', 0)
        epoch = progress.get('epoch', 0)

    global_step = 0 + last_step
    logging.info(f"Training will start at the {last_step+1}th step")

    max_steps = args.max_steps
    if args.steps_this_run is not None:
        if args.steps_this_run + last_step > max_steps:
            logging.info(
                f"Only {max_steps - last_step} steps will be performed in this run due to the limit of --max-steps."
            )
        else:
            max_steps = args.steps_this_run + last_step
            logging.warning(
                f"{args.steps_this_run} steps will be performed in this run.")
    total_samples = 0
    step_start = time.time()
    raw_train_start = None

    while True:
        input_dir = args.input_dir
        if not resume_from_ckpt or progress.get('files', None) is None:
            files = [
                os.path.join(input_dir, f) for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f)) and "training" in
                f
            ]
            files.sort()
            np.random.shuffle(files)
            f_start_id = 0
        else:
            f_start_id = progress['f_id']
            files = progress['files']
        resume_from_ckpt = False

        # Select one file for each worker and create the DataLoader for the file
        data_file = select_dataset_file_for_each_worker(
            files, f_start_id, num_trainers, trainer_id)
        train_data_loader = create_pretraining_dataset(
            args, data_file, feeds, worker_init, paddle.static.cuda_places())

        for f_id in range(f_start_id + 1, len(files)):
            data_file = select_dataset_file_for_each_worker(
                files, f_id, num_trainers, trainer_id)
            dataset_future = pool.submit(create_pretraining_dataset, args,
                                         data_file, feeds, worker_init,
                                         paddle.static.cuda_places())

            if raw_train_start is None:
                raw_train_start = time.time()

            for batch in train_data_loader:
                train_iter += 1
                loss_return = exe.run(program, feed=batch, fetch_list=[loss])
                total_samples += batch_size_per_gpu

                lr = lr_scheduler.get_lr()
                if train_iter % gradient_merge_steps == 0:
                    global_step += 1
                    lr_scheduler.step()

                if train_iter % (log_steps * gradient_merge_steps) == 0:
                    step_cost = time.time() - step_start
                    dllogger_it_data = {
                        'loss': loss_return[0].item(),
                        'learning_rate': lr,
                        'step_cost': step_cost,
                        'step_samples': total_samples,
                        'seqs_per_sec': total_samples / step_cost,
                    }
                    dllogger.log((epoch, global_step), data=dllogger_it_data)
                    total_samples = 0
                    step_start = time.time()

                if args.benchmark and train_iter == (
                        args.benchmark_warmup_steps * gradient_merge_steps):
                    raw_train_start = time.time()

                if train_iter % (save_steps * gradient_merge_steps
                                 ) == 0 or global_step >= max_steps:
                    if trainer_id == 0:
                        model_path = os.path.join(
                            args.output_dir, args.bert_model, "phase1"
                            if args.phase1 else "phase2", f"{global_step}")
                        progress = {
                            'files': files,
                            'epoch': epoch,
                            'global_step': global_step,
                            'f_id': f_id,
                            'phase': 1 if args.phase1 else 2,
                        }
                        save_model(program, model_path, args.model_prefix,
                                   progress)
                        most_recent_ckpts_paths.append(model_path)
                        if len(most_recent_ckpts_paths) > 3:
                            ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                            shutil.rmtree(ckpt_to_be_removed)
                if (global_step >= max_steps) or (
                        args.benchmark and global_step >=
                        args.benchmark_steps + args.benchmark_warmup_steps):
                    train_time_raw = time.time() - raw_train_start
                    del train_data_loader
                    return global_step, loss_return[0].item(), train_time_raw
            del train_data_loader
            train_data_loader = dataset_future.result(timeout=None)
        epoch += 1
