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

import errno
import os
import re
import io
import shutil
import tempfile
import logging
import json
import paddle
import numpy as np
from utils.task import Task

_PROGRESS_SUFFIX = '_progress.json'
_PDOPT_SUFFIX = '.pdopt'
_PDPARAMS_SUFFIX = '.pdparams'


def mkdir_if_not_exist(path):
    """
    Mkdir if not exists, ignore the exception when multiprocess mkdir together.
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logging.warning(
                    f"be happy if some process has already created {path}")
            else:
                raise OSError(f"Failed to mkdir {path}")


def load_train_progress(progress_file):
    """
    Load train progress info (such as file_list, epoch_id, step_id) from a given
        file, which is used to resume training.
    Args:
        progress_file(str): Path to a file named `progress.json` with progress info.
    Returns:
        pregress_dict(dict): A dict with progress info.
    """
    progress_dict = {}
    if os.path.isfile(progress_file):
        with open(progress_file, "r", encoding='utf-8') as reader:
            json_obj = json.loads(reader.read())
        for k, v in json_obj.items():
            progress_dict[k] = v
    else:
        logging.warning("progress file is not found")
    return progress_dict


def _load_state(path):
    """
    Load model parameters from .pdparams file.
    Args:
        path(str): Path to .pdparams file.
    Returns:
        state(dict): Dict of parameters loaded from file.
    """
    if os.path.exists(path + _PDOPT_SUFFIX):
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, os.path.basename(os.path.normpath(path)))
        shutil.copy(path + _PDPARAMS_SUFFIX, dst + _PDPARAMS_SUFFIX)
        state = paddle.static.load_program_state(dst)
        shutil.rmtree(tmp)
    else:
        state = paddle.static.load_program_state(path)
    return state


def load_params(prog, path, ignore_params=None):
    """
    Load model from the given path.
    Args:
        prog (paddle.static.Program): Load weight to which Program object.
        path (string): Model path.
        ignore_params (list): Ignore variable to load when finetuning.
    """
    if not (os.path.isdir(path) or os.path.exists(path + _PDPARAMS_SUFFIX)):
        raise ValueError(f"Model pretrain path {path} does not exists.")

    logging.info(f"Loading parameters from {path}...")

    ignore_set = set()
    state = _load_state(path)

    # ignore the parameter which mismatch the shape
    # between the model and pretrain weight.
    all_var_shape = {}
    for block in prog.blocks:
        for param in block.all_parameters():
            all_var_shape[param.name] = param.shape
    ignore_set.update([
        name for name, shape in all_var_shape.items()
        if name in state and shape != state[name].shape
    ])

    if ignore_params:
        all_var_names = [var.name for var in prog.list_vars()]
        ignore_list = filter(
            lambda var: any([re.match(name, var) for name in ignore_params]),
            all_var_names)
        ignore_set.update(list(ignore_list))

    if len(ignore_set) > 0:
        for k in ignore_set:
            if k in state:
                logging.warning(
                    f"variable {k} is already excluded automatically")
                del state[k]

    for n, p in state.items():
        state[n] = p.astype(np.float32)

    paddle.static.set_program_state(prog, state)


def init_ckpt(path_to_ckpt, program, exe):
    """
    Init from checkpoints or pretrained model in given path.
    Args:
        path_to_ckpt(str): The path to files of checkpoints,
                           including '.pdparams' and '.pdopt'.
        program(paddle.static.Program): The program to init model.
        exe(paddle.static.Executor): The executor to run program.
    """
    if path_to_ckpt:
        paddle.static.load(program, path_to_ckpt, exe)
        logging.info(f"Finish initing checkpoint from {path_to_ckpt}")
        return


def init_pretrained(path_to_pretrained, program):
    """
    Init from checkpoints or pretrained model in given path.
    Args:
        path_to_pretrained(str): The path to file of pretrained model.
        program(paddle.static.Program): The program to init model.
    """
    if path_to_pretrained:
        if not isinstance(path_to_pretrained, list):
            pretrained_model = [path_to_pretrained]
        for pretrain in pretrained_model:
            load_params(program, pretrain)
        logging.info(
            f"Finish initing pretrained model from {pretrained_model}")


def reset_program_state_dict(model, pretrained_file=None):
    """
    Initialize the parameter from the bert config, and set the parameter by
    reseting the state dict."
    """
    state_dict = model.state_dict()
    pretrained_state_dict = None
    if pretrained_file is not None:
        pretrained_state_dict = _load_state(pretrained_file)
    reset_state_dict = {}
    scale = model.bert.bert_config.initializer_range
    reset_parameter_names = []
    for n, p in state_dict.items():
        if pretrained_state_dict is not None and n in pretrained_state_dict:
            reset_state_dict[p.name] = np.array(
                pretrained_state_dict[n], dtype=np.float32)
            reset_parameter_names.append(n)
        elif pretrained_state_dict is not None and p.name in pretrained_state_dict and "bert" in n:
            reset_state_dict[p.name] = np.array(
                pretrained_state_dict[p.name], dtype=np.float32)
            reset_parameter_names.append(n)
        elif "layer_norm" not in p.name and "b_0" not in p.name:
            reset_state_dict[p.name] = np.random.normal(
                loc=0.0, scale=scale, size=p.shape).astype("float32")
    logging.info(
        f"the following parameter had reset, please check. {reset_parameter_names}"
    )
    return reset_state_dict


def init_program(args, program, exe, model, task=Task.pretrain):
    """
    Init from given checkpoint or pretrained parameters.
    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        program(paddle.static.Program): The program to init model.
        exe(paddle.static.Executor): The executor to run program.
        model(paddle.nn.Layer): An instance of BERT model defined in modeling.py.
    """
    progress = None

    if args.from_checkpoint is not None:
        init_ckpt(args.from_checkpoint, program, exe)
        progress = load_train_progress(args.from_checkpoint + _PROGRESS_SUFFIX)
    #elif task == Task.pretrain and args.from_pretrained_params is not None:
    elif args.from_pretrained_params is not None:
        init_pretrained(args.from_pretrained_params, program)
    else:
        reset_state_dict = reset_program_state_dict(
            model, args.from_pretrained_params)
        paddle.static.set_program_state(program, reset_state_dict)
    return progress


def save_model(program, model_path, prefix, progress=None):
    """
    Save a model to given path.
    Args:
        program(paddle.static.Program): The program to be saved.
        model_path(str): The path to save model.
        prefix(str): The prefix of model files.
    """
    if paddle.distributed.get_rank() != 0:
        return
    mkdir_if_not_exist(model_path)
    model_prefix = os.path.join(model_path, prefix)
    paddle.static.save(program, model_prefix)
    if progress is not None:
        progress_file = os.path.join(model_path, prefix + _PROGRESS_SUFFIX)
        out_json = json.dumps(progress, indent=2, sort_keys=True) + "\n"
        with io.open(progress_file, 'w', encoding="utf-8") as f:
            f.write(out_json)
    logging.info(f"Already save model in {model_path}")
