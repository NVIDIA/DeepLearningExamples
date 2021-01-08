import os

import pyprof
import torch
from dllogger import JSONStreamBackend, Logger, StdOutBackend, Verbosity
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from data_loading.data_module import DataModule
from models.nn_unet import NNUnet
from utils.gpu_affinity import set_affinity
from utils.logger import LoggingCallback
from utils.utils import get_main_args, is_main_process, make_empty_dir, set_cuda_devices, verify_ckpt_path


def log(logname, dice, epoch=None, dice_tta=None):
    dllogger = Logger(
        backends=[
            JSONStreamBackend(Verbosity.VERBOSE, os.path.join(args.results, logname)),
            StdOutBackend(Verbosity.VERBOSE, step_format=lambda step: ""),
        ]
    )
    metrics = {}
    if epoch is not None:
        metrics.update({"Epoch": epoch})
    metrics.update({"Mean dice": round(dice.mean().item(), 2)})
    if dice_tta is not None:
        metrics.update({"Mean TTA dice": round(dice_tta.mean().item(), 2)})
    metrics.update({f"L{j+1}": round(m.item(), 2) for j, m in enumerate(dice)})
    if dice_tta is not None:
        metrics.update({f"TTA_L{j+1}": round(m.item(), 2) for j, m in enumerate(dice_tta)})
    dllogger.log(step=(), data=metrics)
    dllogger.flush()


if __name__ == "__main__":
    args = get_main_args()
    if args.profile:
        pyprof.init(enable_function_stack=True)
        print("Profiling enabled")

    if args.affinity != "disabled":
        affinity = set_affinity(os.getenv("LOCAL_RANK", "0"), args.affinity)

    set_cuda_devices(args)
    if is_main_process():
        print(f"{args.exec_mode.upper()} TASK {args.task} FOLD {args.fold} SEED {args.seed}")
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
        log_dir = os.path.join(args.results, args.logname if args.logname is not None else "perf.json")
        callbacks = [
            LoggingCallback(
                log_dir=log_dir,
                global_batch_size=batch_size * args.gpus,
                mode=args.exec_mode,
                warmup=args.warmup,
                dim=args.dim,
                profile=args.profile,
            )
        ]
    elif args.exec_mode == "train":
        model = NNUnet(args)
        if args.save_ckpt:
            model_ckpt = ModelCheckpoint(monitor="dice_sum", mode="max", save_last=True)
        callbacks = [EarlyStopping(monitor="dice_sum", patience=args.patience, verbose=True, mode="max")]
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
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        sync_batchnorm=args.sync_batchnorm,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        default_root_dir=args.results,
        resume_from_checkpoint=ckpt_path,
        accelerator="ddp" if args.gpus > 1 else None,
        checkpoint_callback=model_ckpt,
        limit_train_batches=1.0 if args.train_batches == 0 else args.train_batches,
        limit_val_batches=1.0 if args.test_batches == 0 else args.test_batches,
        limit_test_batches=1.0 if args.test_batches == 0 else args.test_batches,
    )

    if args.benchmark:
        if args.exec_mode == "train":
            if args.profile:
                with torch.autograd.profiler.emit_nvtx():
                    trainer.fit(model, train_dataloader=data_module.train_dataloader())
            else:
                trainer.fit(model, train_dataloader=data_module.train_dataloader())
        else:
            trainer.test(model, test_dataloaders=data_module.test_dataloader())
    elif args.exec_mode == "train":
        trainer.fit(model, data_module)
        if model_ckpt is not None:
            model.args.exec_mode = "evaluate"
            model.args.tta = True
            trainer.interrupted = False
            trainer.test(test_dataloaders=data_module.val_dataloader())
            if is_main_process():
                log_name = args.logname if args.logname is not None else "train_log.json"
                log(log_name, model.best_sum_dice, model.best_sum_epoch, model.eval_dice)
    elif args.exec_mode == "evaluate":
        model.args = args
        trainer.test(model, test_dataloaders=data_module.val_dataloader())
        if is_main_process():
            log(args.logname if args.logname is not None else "eval_log.json", model.eval_dice)
    elif args.exec_mode == "predict":
        model.args = args
        if args.save_preds:
            dir_name = f"preds_task_{args.task}_dim_{args.dim}_fold_{args.fold}"
            if args.tta:
                dir_name += "_tta"
            save_dir = os.path.join(args.results, dir_name)
            model.save_dir = save_dir
            make_empty_dir(save_dir)
        trainer.test(model, test_dataloaders=data_module.test_dataloader())
