# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

from typing import List

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from se3_transformer.runtime import gpu_affinity
from se3_transformer.runtime.arguments import PARSER
from se3_transformer.runtime.callbacks import BaseCallback
from se3_transformer.runtime.loggers import DLLogger
from se3_transformer.runtime.utils import to_cuda, get_local_rank


@torch.inference_mode()
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             callbacks: List[BaseCallback],
             args):
    model.eval()
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), unit='batch', desc=f'Evaluation',
                         leave=False, disable=(args.silent or get_local_rank() != 0)):
        *input, target = to_cuda(batch)

        for callback in callbacks:
            callback.on_batch_start()

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(*input)

            for callback in callbacks:
                callback.on_validation_step(input, target, pred)


if __name__ == '__main__':
    from se3_transformer.runtime.callbacks import QM9MetricCallback, PerformanceCallback
    from se3_transformer.runtime.utils import init_distributed, seed_everything
    from se3_transformer.model import SE3TransformerPooled, Fiber
    from se3_transformer.data_loading import QM9DataModule
    import torch.distributed as dist
    import logging
    import sys

    is_distributed = init_distributed()
    local_rank = get_local_rank()
    args = PARSER.parse_args()

    logging.getLogger().setLevel(logging.CRITICAL if local_rank != 0 or args.silent else logging.INFO)

    logging.info('====== SE(3)-Transformer ======')
    logging.info('|  Inference on the test set  |')
    logging.info('===============================')

    if not args.benchmark and args.load_ckpt_path is None:
        logging.error('No load_ckpt_path provided, you need to provide a saved model to evaluate')
        sys.exit(1)

    if args.benchmark:
        logging.info('Running benchmark mode with one warmup pass')

    if args.seed is not None:
        seed_everything(args.seed)

    major_cc, minor_cc = torch.cuda.get_device_capability()

    logger = DLLogger(args.log_dir, filename=args.dllogger_name)
    datamodule = QM9DataModule(**vars(args))
    model = SE3TransformerPooled(
        fiber_in=Fiber({0: datamodule.NODE_FEATURE_DIM}),
        fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
        fiber_edge=Fiber({0: datamodule.EDGE_FEATURE_DIM}),
        output_dim=1,
        tensor_cores=(args.amp and major_cc >= 7) or major_cc >= 8,  # use Tensor Cores more effectively
        **vars(args)
    )
    callbacks = [QM9MetricCallback(logger, targets_std=datamodule.targets_std, prefix='test')]

    model.to(device=torch.cuda.current_device())
    if args.load_ckpt_path is not None:
        checkpoint = torch.load(str(args.load_ckpt_path), map_location={'cuda:0': f'cuda:{local_rank}'})
        model.load_state_dict(checkpoint['state_dict'])

    if is_distributed:
        nproc_per_node = torch.cuda.device_count()
        affinity = gpu_affinity.set_affinity(local_rank, nproc_per_node)
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    test_dataloader = datamodule.test_dataloader() if not args.benchmark else datamodule.train_dataloader()
    evaluate(model,
             test_dataloader,
             callbacks,
             args)

    for callback in callbacks:
        callback.on_validation_end()

    if args.benchmark:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        callbacks = [PerformanceCallback(logger, args.batch_size * world_size, warmup_epochs=1, mode='inference')]
        for _ in range(6):
            evaluate(model,
                     test_dataloader,
                     callbacks,
                     args)
            callbacks[0].on_epoch_end()

        callbacks[0].on_fit_end()
