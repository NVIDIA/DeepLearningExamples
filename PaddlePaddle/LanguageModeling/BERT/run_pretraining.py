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

import os
import time
import logging
import paddle
import paddle.distributed.fleet as fleet

from utils.config import parse_args, print_args
from utils.save_load import init_program
from utils.logger import setup_loggers
from utils.affinity import set_cpu_affinity
from utils.utility import set_seed, get_trainer_id, get_num_trainers
import program
import dllogger
from lddl.paddle import get_bert_pretrain_data_loader


def main():
    """
    An enterpoint to train a BERT model, which contains five steps.
        1. Parse arguments from command line.
        2. Initialize distributed training related setting, including CPU affinity.
        3. Create training Paddle.static.Program.
        4. Load checkpoint or pretrained model if given.
        5. Run program (train with datasets and save model if necessary).
    """
    now = time.time()
    args = parse_args()
    setup_loggers(args.report_file)

    if args.show_config:
        print_args(args)

    device = paddle.set_device('gpu')
    fleet.init(is_collective=True)
    if args.enable_cpu_affinity:
        set_cpu_affinity()

    # Create the random seed for the worker
    set_seed(args.seed + get_trainer_id())

    dllogger.log(step="PARAMETER", data={"SEED": args.seed})
    dllogger.log(step="PARAMETER", data={"train_start": True})
    dllogger.log(step="PARAMETER",
                 data={"batch_size_per_gpu": args.batch_size})
    dllogger.log(step="PARAMETER", data={"learning_rate": args.learning_rate})

    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    model, lr_scheduler, optimizer, loss, feeds = program.build(
        args, main_program, startup_program)

    exe = paddle.static.Executor(device)
    exe.run(startup_program)

    progress = init_program(args, program=main_program, exe=exe, model=model)
    train_dataloader = get_bert_pretrain_data_loader(
        args.input_dir,
        vocab_file=args.vocab_file,
        data_loader_kwargs={
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'persistent_workers': True,
            'feed_list': feeds
        },
        base_seed=args.seed,
        log_dir=None if args.output_dir is None else
        os.path.join(args.output_dir, 'lddl_log'),
        log_level=logging.WARNING,
        start_epoch=0 if progress is None else progress.get("epoch", 0),
        sequence_length_alignment=64)

    if args.amp:
        optimizer.amp_init(device)

    global_step, actual_steps_this_run, final_loss, train_time_raw = program.run(
        exe, main_program, args, lr_scheduler, loss, train_dataloader,
        progress)

    if get_trainer_id() == 0:
        e2e_time = time.time() - now
        if args.benchmark:
            training_perf = args.batch_size * args.gradient_merge_steps * (
                actual_steps_this_run - args.benchmark_warmup_steps
            ) * get_num_trainers() / train_time_raw
        else:
            training_perf = args.batch_size * args.gradient_merge_steps * actual_steps_this_run * get_num_trainers(
            ) / train_time_raw
        dllogger.log(step=tuple(),
                     data={
                         "e2e_train_time": e2e_time,
                         "training_sequences_per_second": training_perf,
                         "final_loss": final_loss,
                         "raw_train_time": train_time_raw
                     })


if __name__ == "__main__":
    paddle.enable_static()
    main()
