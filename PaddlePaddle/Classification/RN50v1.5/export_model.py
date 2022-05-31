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
import logging
import paddle
import program
from dali import build_dataloader
from utils.mode import Mode
from utils.save_load import init_ckpt
from utils.logger import setup_dllogger
from utils.config import parse_args, print_args


def main(args):
    '''
    Export saved model params to paddle inference model
    '''
    setup_dllogger(args.trt_export_log_path)
    if args.show_config:
        print_args(args)

    eval_dataloader = build_dataloader(args, Mode.EVAL)

    startup_prog = paddle.static.Program()
    eval_prog = paddle.static.Program()

    eval_fetchs, _, eval_feeds, _ = program.build(
        args,
        eval_prog,
        startup_prog,
        step_each_epoch=len(eval_dataloader),
        is_train=False)
    eval_prog = eval_prog.clone(for_test=True)

    device = paddle.set_device('gpu')
    exe = paddle.static.Executor(device)
    exe.run(startup_prog)

    path_to_ckpt = args.from_checkpoint

    if path_to_ckpt is None:
        logging.warning(
            'The --from-checkpoint is not set, model weights will not be initialize.'
        )
    else:
        init_ckpt(path_to_ckpt, eval_prog, exe)
        logging.info('Checkpoint path is %s', path_to_ckpt)

    save_inference_dir = args.trt_inference_dir
    paddle.static.save_inference_model(
        path_prefix=os.path.join(save_inference_dir, args.model_arch_name),
        feed_vars=[eval_feeds['data']],
        fetch_vars=[eval_fetchs['label'][0]],
        executor=exe,
        program=eval_prog)

    logging.info('Successully export inference model to %s',
                 save_inference_dir)


if __name__ == '__main__':
    paddle.enable_static()
    main(parse_args(including_trt=True))
