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
import json
import time
import collections
import sys
import subprocess

import numpy as np
import paddle

import paddle.distributed.fleet as fleet
from paddle.fluid.contrib.mixed_precision.fp16_utils import rewrite_program
from paddle.fluid.contrib.mixed_precision.fp16_lists import AutoMixedPrecisionLists

from modeling import BertForQuestionAnswering, BertConfig
from tokenizer import BertTokenizer
from squad_utils import get_answers
from loss import CrossEntropyLossForSQuAD
from squad_dataset import SQuAD, create_squad_data_holder
from utils.collate import Pad, Stack, Tuple
from utils.utility import get_num_trainers, get_trainer_id, set_seed
from utils.logger import setup_loggers
from utils.affinity import set_cpu_affinity
from utils.save_load import mkdir_if_not_exist, init_program, save_model
from utils.config import print_args, parse_args
from utils.task import Task
from optimizer import AdamW
from lr_scheduler import Poly
from program import dist_optimizer
import dllogger


def evaluate(args, exe, logits, dev_program, data_loader):
    RawResult = collections.namedtuple(
        "RawResult", ["unique_id", "start_logits", "end_logits"])
    all_results = []
    infer_start = time.time()
    tic_eval = time.time()
    tic_benchmark_begin = 0
    tic_benchmark_end = 0

    dllogger.log(step="PARAMETER", data={"eval_start": True})
    for step, batch in enumerate(data_loader):
        start_logits_tensor, end_logits_tensor = exe.run(dev_program,
                                                         feed=batch,
                                                         fetch_list=[*logits])

        if args.benchmark and step == args.benchmark_warmup_steps:
            tic_benchmark_begin = time.time()
        if args.benchmark and step == args.benchmark_warmup_steps + args.benchmark_steps:
            tic_benchmark_end = time.time()

        unique_ids = np.array(batch[0]['unique_id'])
        for idx in range(unique_ids.shape[0]):
            if len(all_results) % 1000 == 0 and len(all_results):
                dllogger.log(step="PARAMETER",
                             data={
                                 "sample_number": len(all_results),
                                 "time_per_1000": time.time() - tic_eval
                             })
                tic_eval = time.time()
            unique_id = int(unique_ids[idx])
            start_logits = [float(x) for x in start_logits_tensor[idx]]
            end_logits = [float(x) for x in end_logits_tensor[idx]]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))
    if args.benchmark:
        time_to_benchmark = tic_benchmark_end - tic_benchmark_begin
        dllogger.log(step=tuple(),
                     data={
                         "inference_sequences_per_second":
                         args.predict_batch_size * args.benchmark_steps /
                         time_to_benchmark
                     })
        return
    else:
        time_to_infer = time.time() - infer_start
        dllogger.log(step=tuple(),
                     data={
                         "e2e_inference_time": time_to_infer,
                         "inference_sequences_per_second":
                         len(data_loader.dataset.features) / time_to_infer
                     })

    output_dir = os.path.join(args.output_dir, args.bert_model, "squad")
    mkdir_if_not_exist(output_dir)
    output_prediction_file = os.path.join(output_dir, "predictions.json")
    output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")

    answers, nbest_answers = get_answers(args, data_loader.dataset.examples,
                                         data_loader.dataset.features,
                                         all_results)
    with open(output_prediction_file, "w") as f:
        f.write(json.dumps(answers, indent=4) + "\n")
    with open(output_nbest_file, "w") as f:
        f.write(json.dumps(nbest_answers, indent=4) + "\n")

    if args.do_eval:
        eval_out = subprocess.check_output([
            sys.executable, args.eval_script, args.predict_file,
            output_prediction_file
        ])
        scores = str(eval_out).strip()
        exact_match = float(scores.split(":")[1].split(",")[0])
        f1 = float(scores.split(":")[2].split("}")[0])
        dllogger.log(step=tuple(), data={"exact_match": exact_match, "F1": f1})


def main(args):
    setup_loggers(args.report_file)
    if args.show_config:
        print_args(args)

    trainer_id = get_trainer_id()
    num_trainers = get_num_trainers()

    # Set the paddle execute enviroment
    fleet.init(is_collective=True)
    if args.enable_cpu_affinity:
        set_cpu_affinity()

    place = paddle.set_device('gpu')
    set_seed(args.seed)
    dllogger.log(step="PARAMETER", data={"SEED": args.seed})

    # Create the main_program for the training and dev_program for the validation
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    tokenizer = BertTokenizer(
        vocab_file=args.vocab_file,
        do_lower_case=args.do_lower_case,
        max_len=512)

    with paddle.static.program_guard(main_program, startup_program):
        input_ids, segment_ids, start_positions, end_positions, unique_id = create_squad_data_holder(
        )

    if args.do_train:
        train_dataset = SQuAD(
            tokenizer=tokenizer,
            doc_stride=args.doc_stride,
            path=args.train_file,
            version_2_with_negative=args.version_2_with_negative,
            max_query_length=args.max_query_length,
            max_seq_length=args.max_seq_length,
            mode="train")

        train_batch_sampler = paddle.io.DistributedBatchSampler(
            train_dataset, batch_size=args.train_batch_size, shuffle=True)

        train_batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # segment
            Stack(),  # unique_id
            Stack(dtype="int64"),  # start_pos
            Stack(dtype="int64")  # end_pos
        ): [data for i, data in enumerate(fn(samples)) if i != 2]

        train_data_loader = paddle.io.DataLoader(
            dataset=train_dataset,
            feed_list=[
                input_ids, segment_ids, start_positions, end_positions
            ],
            batch_sampler=train_batch_sampler,
            collate_fn=train_batchify_fn,
            num_workers=0,
            return_list=False)

    with paddle.static.program_guard(main_program, startup_program):
        bert_config = BertConfig.from_json_file(args.config_file)
        bert_config.fuse_mha = args.fuse_mha
        if bert_config.vocab_size % 8 != 0:
            bert_config.vocab_size += 8 - (bert_config.vocab_size % 8)


        model = BertForQuestionAnswering(bert_config)
        criterion = CrossEntropyLossForSQuAD()
        logits = model(input_ids=input_ids, token_type_ids=segment_ids)
        if args.do_predict:
            dev_program = main_program.clone(for_test=True)

        if args.do_train:
            loss = criterion(logits, (start_positions, end_positions))
            num_train_steps = len(train_data_loader) * args.epochs
            if args.max_steps is not None and args.max_steps > 0:
                num_train_steps = min(num_train_steps, args.max_steps)
            lr_scheduler = Poly(
                learning_rate=args.learning_rate, num_steps=num_train_steps)()
            optimizer = AdamW(args, learning_rate=lr_scheduler)()
            optimizer = dist_optimizer(args, optimizer)
            optimizer.minimize(loss)

    exe = paddle.static.Executor(place)
    exe.run(startup_program)
    init_program(
        args, program=main_program, exe=exe, model=model, task=Task.squad)

    if args.do_train:
        dllogger.log(step="PARAMETER", data={"train_start": True})
        dllogger.log(step="PARAMETER",
                     data={
                         "training_samples":
                         len(train_data_loader.dataset.examples)
                     })
        dllogger.log(step="PARAMETER",
                     data={
                         "training_features":
                         len(train_data_loader.dataset.features)
                     })
        dllogger.log(step="PARAMETER",
                     data={"train_batch_size": args.train_batch_size})
        dllogger.log(step="PARAMETER", data={"steps": num_train_steps})

        global_step = 0
        tic_benchmark_begin = 0
        tic_benchmark_end = 0
        tic_train_begin = time.time()
        for epoch in range(args.epochs):
            for batch in train_data_loader:
                if global_step >= num_train_steps:
                    break
                if args.benchmark and global_step >= args.benchmark_warmup_steps + args.benchmark_steps:
                    break

                loss_return = exe.run(main_program,
                                      feed=batch,
                                      fetch_list=[loss])

                lr = lr_scheduler.get_lr()
                lr_scheduler.step()

                global_step += 1

                if args.benchmark and global_step == args.benchmark_warmup_steps:
                    tic_benchmark_begin = time.time()
                if args.benchmark and global_step == args.benchmark_warmup_steps + args.benchmark_steps:
                    tic_benchmark_end = time.time()

                if global_step % args.log_freq == 0:
                    dllogger_it_data = {
                        'loss': loss_return[0].item(),
                        'learning_rate': lr
                    }
                    dllogger.log((epoch, global_step), data=dllogger_it_data)

        if not args.benchmark:
            time_to_train = time.time() - tic_train_begin
            dllogger.log(step=tuple(),
                         data={
                             "e2e_train_time": time_to_train,
                             "training_sequences_per_second":
                             args.train_batch_size * num_train_steps *
                             num_trainers / time_to_train
                         })
        else:
            time_to_benchmark = tic_benchmark_end - tic_benchmark_begin
            dllogger.log(step=tuple(),
                         data={
                             "training_sequences_per_second":
                             args.train_batch_size * args.benchmark_steps *
                             num_trainers / time_to_benchmark
                         })
        if trainer_id == 0:
            model_path = os.path.join(args.output_dir, args.bert_model,
                                      "squad")
            save_model(main_program, model_path, args.model_prefix)

    if args.do_predict and trainer_id == 0:
        dev_dataset = SQuAD(
            tokenizer=tokenizer,
            doc_stride=args.doc_stride,
            path=args.predict_file,
            version_2_with_negative=args.version_2_with_negative,
            max_query_length=args.max_query_length,
            max_seq_length=args.max_seq_length,
            mode="dev")

        dev_batch_sampler = paddle.io.BatchSampler(
            dev_dataset, batch_size=args.predict_batch_size, shuffle=False)

        dev_batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # segment
            Stack()  # unique_id
        ): fn(samples)

        dev_data_loader = paddle.io.DataLoader(
            dataset=dev_dataset,
            feed_list=[input_ids, segment_ids, unique_id],
            batch_sampler=dev_batch_sampler,
            collate_fn=dev_batchify_fn,
            num_workers=0,
            return_list=False)

        dllogger.log(step="PARAMETER", data={"predict_start": True})
        dllogger.log(
            step="PARAMETER",
            data={"eval_samples": len(dev_data_loader.dataset.examples)})
        dllogger.log(
            step="PARAMETER",
            data={"eval_features": len(dev_data_loader.dataset.features)})
        dllogger.log(step="PARAMETER",
                     data={"predict_batch_size": args.predict_batch_size})
        if args.amp:
            amp_lists = AutoMixedPrecisionLists(
                custom_white_list=['softmax', 'layer_norm', 'gelu'])
            rewrite_program(dev_program, amp_lists=amp_lists)
        evaluate(args, exe, logits, dev_program, dev_data_loader)


if __name__ == "__main__":
    paddle.enable_static()
    main(parse_args(Task.squad))
