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
import os

from dali import build_dataloader
from utils.affinity import set_cpu_affinity
from utils.config import parse_args, print_args
from utils.logger import setup_dllogger
from utils.mode import Mode, RunScope
from utils.save_load import init_program, save_model

import paddle
import program
from paddle.distributed import fleet
from paddle.static.amp.fp16_lists import AutoMixedPrecisionLists
from paddle.static.amp.fp16_utils import cast_model_to_fp16
from paddle.incubate import asp as sparsity
from paddle.static.quantization.quanter import quant_aware


class MetricSummary:
    def __init__(self):
        super().__init__()
        self.metric_dict = None

    def update(self, new_metrics):
        if not self.is_updated:
            self.metric_dict = {}

        for key in new_metrics:
            if key in self.metric_dict:
                # top1, top5 and ips are "larger is better"
                if key in ['top1', 'top5', 'ips']:
                    self.metric_dict[key] = (
                        new_metrics[key]
                        if new_metrics[key] > self.metric_dict[key]
                        else self.metric_dict[key]
                    )
                # Others are "Smaller is better"
                else:
                    self.metric_dict[key] = (
                        new_metrics[key]
                        if new_metrics[key] < self.metric_dict[key]
                        else self.metric_dict[key]
                    )
            else:
                self.metric_dict[key] = new_metrics[key]

    @property
    def is_updated(self):
        return self.metric_dict is not None


def main(args):
    """
    A enterpoint to train and evaluate a ResNet50 model, which contains six steps.
        1. Parse arguments from command line.
        2. Initialize distributed training related setting, including CPU affinity.
        3. Build dataloader via DALI.
        4. Create training and evaluating Paddle.static.Program.
        5. Load checkpoint or pretrained model if given.
        6. Run program (train and evaluate with datasets, then save model if necessary).
    """
    setup_dllogger(args.report_file)
    if args.show_config:
        print_args(args)

    fleet.init(is_collective=True)
    if args.enable_cpu_affinity:
        set_cpu_affinity()

    device = paddle.set_device('gpu')
    startup_prog = paddle.static.Program()

    train_dataloader = None
    train_prog = None
    optimizer = None
    if args.run_scope in [RunScope.TRAIN_EVAL, RunScope.TRAIN_ONLY]:
        train_dataloader = build_dataloader(args, Mode.TRAIN)
        train_step_each_epoch = len(train_dataloader)
        train_prog = paddle.static.Program()

        train_fetchs, lr_scheduler, _, optimizer = program.build(
            args,
            train_prog,
            startup_prog,
            step_each_epoch=train_step_each_epoch,
            is_train=True,
        )

    eval_dataloader = None
    eval_prog = None
    if args.run_scope in [RunScope.TRAIN_EVAL, RunScope.EVAL_ONLY]:
        eval_dataloader = build_dataloader(args, Mode.EVAL)
        eval_step_each_epoch = len(eval_dataloader)
        eval_prog = paddle.static.Program()

        eval_fetchs, _, eval_feeds, _ = program.build(
            args,
            eval_prog,
            startup_prog,
            step_each_epoch=eval_step_each_epoch,
            is_train=False,
        )
        # clone to prune some content which is irrelevant in eval_prog
        eval_prog = eval_prog.clone(for_test=True)

    exe = paddle.static.Executor(device)
    exe.run(startup_prog)

    init_program(
        args,
        exe=exe,
        program=train_prog if train_prog is not None else eval_prog,
    )

    if args.amp:
        if args.run_scope == RunScope.EVAL_ONLY:
            cast_model_to_fp16(
                eval_prog,
                AutoMixedPrecisionLists(),
                use_fp16_guard=False,
                level='O1',
            )
        else:
            optimizer.amp_init(
                device,
                scope=paddle.static.global_scope(),
                test_program=eval_prog,
                use_fp16_test=True,
            )

    if args.asp and args.prune_model:
        logging.info("Pruning model to 2:4 sparse pattern...")
        sparsity.prune_model(train_prog, mask_algo=args.mask_algo)
        logging.info("Pruning model done.")

    if args.qat:
        if args.run_scope == RunScope.EVAL_ONLY:
            eval_prog = quant_aware(eval_prog, device, for_test=True, return_program=True)
        else:
            optimizer.qat_init(
                device,
                test_program=eval_prog)

    if eval_prog is not None:
        eval_prog = program.compile_prog(args, eval_prog, is_train=False)

    train_summary = MetricSummary()
    eval_summary = MetricSummary()
    for epoch_id in range(args.start_epoch, args.epochs):
        # Training
        if train_prog is not None:
            metric_summary = program.run(
                args,
                train_dataloader,
                exe,
                train_prog,
                train_fetchs,
                epoch_id,
                Mode.TRAIN,
                lr_scheduler,
            )
            train_summary.update(metric_summary)

            # Save a checkpoint
            if epoch_id % args.save_interval == 0:
                model_path = os.path.join(args.checkpoint_dir, args.model_arch_name)
                save_model(train_prog, model_path, epoch_id, args.model_prefix)

        # Evaluation
        if (eval_prog is not None) and (epoch_id % args.eval_interval == 0):
            metric_summary = program.run(
                args,
                eval_dataloader,
                exe,
                eval_prog,
                eval_fetchs,
                epoch_id,
                Mode.EVAL,
            )
            eval_summary.update(metric_summary)

    if train_summary.is_updated:
        program.log_info((), train_summary.metric_dict, Mode.TRAIN)
    if eval_summary.is_updated:
        program.log_info((), eval_summary.metric_dict, Mode.EVAL)

    if eval_prog is not None:
        model_path = os.path.join(args.inference_dir, args.model_arch_name)
        paddle.static.save_inference_model(model_path, [eval_feeds['data']], [eval_fetchs['label'][0]], exe, program=eval_prog)


if __name__ == '__main__':
    paddle.enable_static()
    main(parse_args())
