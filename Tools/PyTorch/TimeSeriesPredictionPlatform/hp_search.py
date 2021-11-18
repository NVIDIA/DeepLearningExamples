import argparse
import copy
import logging
import multiprocessing
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

import hydra
import omegaconf
import optuna
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from optuna.samplers import TPESampler

from distributed_utils import (
    get_device,
    get_rank,
    init_distributed,
    is_main_process,
    log,
)
from training.trainer import CTLTrainer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


def sample_param(param, trial, name=""):
    """ Sample parameters for trial """
    if param.sampling in ["categorical", "discrete"]:

        return trial.suggest_categorical(name, param.get("values"))
    if param.sampling == "int_uniform":
        step = param.step_value if (hasattr(param, "step_value") and param.step_value is not None) else 1
        return trial.suggest_int(name, param.min_value, param.max_value, step=step)
    if param.sampling == "float_uniform":
        return trial.suggest_uniform(name, param.min_value, param.max_value)
    if param.sampling == "log_uniform":
        return trial.suggest_loguniform(name, param.min_value, param.max_value)
    if param.sampling == "discrete_uniform":
        return trial.suggest_discrete_uniform(name, param.min_value, param.max_value, param.step_value)

    raise ValueError(f"Unknown sampling for param: {param.sampling}")


def traverse_conf(node, trial, name="Root"):
    if isinstance(node, (omegaconf.dictconfig.DictConfig, dict)):
        if node.get("sampling", None):
            return sample_param(node, trial, name=name)
        else:
            to_change = []
            for key, value in node.items():
                new_value = traverse_conf(value, trial, name=key)
                if new_value is not None:
                    to_change.append((key, new_value))
            for key, value in to_change:
                node[key] = value


def launch_trial(cfg):
    if not cfg.config.get("log_path", None):
        cfg.config.log_path = datetime.now().strftime("./outputs/%Y-%m-%d/%H-%M-%S-%f/")
    os.makedirs(os.path.join(cfg.config.get("log_path"), ".hydra"), exist_ok=True)
    with open(os.path.join(cfg.config.get("log_path"), ".hydra", "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)
    cfg._target_ = cfg.config.dataset._target_
    train, valid, test = hydra.utils.call(cfg)
    cfg._target_ = cfg.config.model._target_
    model = hydra.utils.instantiate(cfg)
    model = model.cuda()
    cfg._target_ = cfg.config.optimizer._target_
    optimizer = hydra.utils.instantiate(cfg, params=model.parameters())
    cfg._target_ = cfg.config.criterion._target_
    criterion = hydra.utils.call(cfg)
    cfg._target_ = cfg.config.evaluator._target_
    evaluator = hydra.utils.instantiate(cfg)

    trainer = CTLTrainer(model, train, valid, test, optimizer, evaluator, criterion, cfg.config)
    trainer.train()
    if is_main_process():
        result = trainer.evaluate()
        log(result)
        return result[cfg.config.optuna.get("goal_metric", "MAE")]


def main(args):
    if args.distributed_world_size > 1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.synchronize()

    with open(args.config_path, "rb") as f:
        cfg = OmegaConf.load(f)
    # Launching hp search with varied world size is problematic.
    if not isinstance(cfg.config.device.get("world_size", 1), int):
        print("HP search currently does not support varied world sizes. Setting world size = 1", file=sys.stderr)
        cfg.config.device.world_size = 1

    # BUG: Instantiating workers in second run of hp search causes cuda reinitialization which hits the same cuda context.
    if cfg.config.trainer.num_workers != 0:
        print("HP search currently does not support dataloading in subprocesses. Setting num_workers = 0", file=sys.stderr)
        cfg.config.trainer.num_workers = 0

    study = optuna.load_study(study_name=args.study_name, storage="sqlite:////workspace/{}.db".format(args.study_name))

    def objective(trial, cfg=cfg):
        if cfg.config.device.get("world_size", 1) > 1:
            trial = optuna.integration.TorchDistributedTrial(trial, device="cuda")
        cfg = copy.deepcopy(cfg)
        traverse_conf(cfg.config, trial)
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            result = launch_trial(cfg)
        return result

    if is_main_process():
        study.optimize(objective, n_trials=cfg.config.optuna.get("n_trials", 10), n_jobs=1)
    else:
        for _ in range(cfg.config.optuna.get("n_trials", 10)):
            objective(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distributed_world_size",
        type=int,
        metavar="N",
        default=torch.cuda.device_count(),
        help="total number of GPUs across all nodes (default: all visible GPUs)",
    )
    parser.add_argument(
        "--distributed_rank", default=os.getenv("LOCAL_RANK", 0), type=int, help="rank of the current worker"
    )
    parser.add_argument("--local_rank", default=0, type=int, help="rank of the current worker")
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--study_name", required=True, type=str)
    ARGS = parser.parse_args()
    main(ARGS)
