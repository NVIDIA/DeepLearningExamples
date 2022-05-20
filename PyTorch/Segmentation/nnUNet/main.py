# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from data_loading.data_module import DataModule
from nnunet.nn_unet import NNUnet
from utils.args import get_main_args
from utils.logger import LoggingCallback
from utils.utils import make_empty_dir, set_cuda_devices, set_granularity, verify_ckpt_path

if __name__ == "__main__":
    args = get_main_args()
    set_granularity()  # Increase maximum fetch granularity of L2 to 128 bytes
    set_cuda_devices(args)
    seed_everything(args.seed)
    data_module = DataModule(args)
    data_module.setup()
    ckpt_path = verify_ckpt_path(args)

    model = NNUnet(args)
    callbacks = [RichProgressBar(), ModelSummary(max_depth=2)]
    logger = False
    if args.benchmark:
        batch_size = args.batch_size if args.exec_mode == "train" else args.val_batch_size
        filnename = args.logname if args.logname is not None else "perf.json"
        callbacks.append(
            LoggingCallback(
                log_dir=args.results,
                filnename=filnename,
                global_batch_size=batch_size * args.gpus * args.nodes,
                mode=args.exec_mode,
                warmup=args.warmup,
                dim=args.dim,
            )
        )
    elif args.exec_mode == "train":
        if args.tb_logs:
            logger = TensorBoardLogger(
                save_dir=f"{args.results}/tb_logs",
                name=f"task={args.task}_dim={args.dim}_fold={args.fold}_precision={16 if args.amp else 32}",
                default_hp_metric=False,
                version=0,
            )
        if args.save_ckpt:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=f"{args.ckpt_store_dir}/checkpoints",
                    filename="{epoch}-{dice:.2f}",
                    monitor="dice",
                    mode="max",
                    save_last=True,
                )
            )

    trainer = Trainer(
        logger=logger,
        default_root_dir=args.results,
        benchmark=True,
        deterministic=False,
        max_epochs=args.epochs,
        precision=16 if args.amp else 32,
        gradient_clip_val=args.gradient_clip_val,
        enable_checkpointing=args.save_ckpt,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=args.gpus,
        num_nodes=args.nodes,
        strategy="ddp" if args.gpus > 1 else None,
        limit_train_batches=1.0 if args.train_batches == 0 else args.train_batches,
        limit_val_batches=1.0 if args.test_batches == 0 else args.test_batches,
        limit_test_batches=1.0 if args.test_batches == 0 else args.test_batches,
    )

    if args.benchmark:
        if args.exec_mode == "train":
            trainer.fit(model, train_dataloaders=data_module.train_dataloader())
        else:
            # warmup
            trainer.test(model, dataloaders=data_module.test_dataloader(), verbose=False)
            # benchmark run
            model.start_benchmark = 1
            trainer.test(model, dataloaders=data_module.test_dataloader(), verbose=False)
    elif args.exec_mode == "train":
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    elif args.exec_mode == "evaluate":
        trainer.validate(model, val_dataloaders=data_module.val_dataloader())
    elif args.exec_mode == "predict":
        if args.save_preds:
            ckpt_name = "_".join(args.ckpt_path.split("/")[-1].split(".")[:-1])
            dir_name = f"predictions_{ckpt_name}"
            dir_name += f"_task={model.args.task}_fold={model.args.fold}"
            if args.tta:
                dir_name += "_tta"
            save_dir = os.path.join(args.results, dir_name)
            model.save_dir = save_dir
            make_empty_dir(save_dir)
        model.args = args
        trainer.test(model, test_dataloaders=data_module.test_dataloader(), ckpt_path=ckpt_path)
