# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import ctypes
import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, early_stopping

from data_loading.data_module import DataModule
from nnunet.nn_unet import NNUnet
from utils.args import get_main_args
from utils.gpu_affinity import set_affinity
from utils.logger import LoggingCallback
from utils.utils import make_empty_dir, set_cuda_devices, verify_ckpt_path

if __name__ == "__main__":
    args = get_main_args()

    if args.affinity != "disabled":
        set_affinity(int(os.getenv("LOCAL_RANK", "0")), args.gpus, mode=args.affinity)

    # Limit number of CPU threads
    os.environ["OMP_NUM_THREADS"] = "1"
    # Set device limit on the current device cudaLimitMaxL2FetchGranularity = 0x05
    _libcudart = ctypes.CDLL("libcudart.so")
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128

    set_cuda_devices(args)
    seed_everything(args.seed)
    data_module = DataModule(args)
    data_module.prepare_data()
    data_module.setup()
    ckpt_path = verify_ckpt_path(args)

    callbacks = None
    model_ckpt = None
    if args.benchmark:
        model = NNUnet(args)
        batch_size = args.batch_size if args.exec_mode == "train" else args.val_batch_size
        filnename = args.logname if args.logname is not None else "perf1.json"
        callbacks = [
            LoggingCallback(
                log_dir=args.results,
                filnename=filnename,
                global_batch_size=batch_size * args.gpus,
                mode=args.exec_mode,
                warmup=args.warmup,
                dim=args.dim,
            )
        ]
    elif args.exec_mode == "train":
        model = NNUnet(args)
        early_stopping = EarlyStopping(monitor="dice_mean", patience=args.patience, verbose=True, mode="max")
        callbacks = [early_stopping]
        if args.save_ckpt:
            model_ckpt = ModelCheckpoint(
                filename="{epoch}-{dice_mean:.2f}", monitor="dice_mean", mode="max", save_last=True
            )
            callbacks.append(model_ckpt)
    else:  # Evaluation or inference
        if ckpt_path is not None:
            model = NNUnet.load_from_checkpoint(ckpt_path)
        else:
            model = NNUnet(args)

    trainer = Trainer(
        logger=False,
        gpus=args.gpus,
        precision=16 if args.amp else 32,
        benchmark=True,
        deterministic=False,
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        sync_batchnorm=args.sync_batchnorm,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        default_root_dir=args.results,
        resume_from_checkpoint=ckpt_path,
        accelerator="ddp" if args.gpus > 1 else None,
        checkpoint_callback=args.save_ckpt,
        limit_train_batches=1.0 if args.train_batches == 0 else args.train_batches,
        limit_val_batches=1.0 if args.test_batches == 0 else args.test_batches,
        limit_test_batches=1.0 if args.test_batches == 0 else args.test_batches,
    )

    if args.benchmark:
        if args.exec_mode == "train":
            trainer.fit(model, train_dataloader=data_module.train_dataloader())
        else:
            # warmup
            trainer.test(model, test_dataloaders=data_module.test_dataloader())
            # benchmark run
            trainer.current_epoch = 1
            trainer.test(model, test_dataloaders=data_module.test_dataloader())
    elif args.exec_mode == "train":
        trainer.fit(model, data_module)
    elif args.exec_mode == "evaluate":
        model.args = args
        trainer.test(model, test_dataloaders=data_module.val_dataloader())
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
        trainer.test(model, test_dataloaders=data_module.test_dataloader())
