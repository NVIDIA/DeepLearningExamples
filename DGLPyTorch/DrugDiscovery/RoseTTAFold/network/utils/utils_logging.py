import os
import sys
import time
import datetime
import subprocess
import numpy as np
import torch

from utils.utils_data import to_np


_global_log = {}


def try_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# @profile
def make_logdir(checkpoint_dir, run_name=None):
    if run_name is None:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H.%M.%S")
    else:
        assert type(run_name) == str
        now = run_name

    log_dir = os.path.join(checkpoint_dir, now)
    try_mkdir(log_dir)
    return log_dir


def count_parameters(model):
    """
    count number of trainable parameters in module
    :param model: nn.Module instance
    :return: integer
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    return n_params


def write_info_file(model, FLAGS, UNPARSED_ARGV, wandb_log_dir=None):
    time_str = time.strftime("%m%d_%H%M%S")
    filename_log = "info_" + time_str + ".txt"
    filename_git_diff = "git_diff_" + time_str + ".txt"

    checkpoint_name = 'model'

    if wandb_log_dir:
        log_dir = wandb_log_dir
        os.mkdir(os.path.join(log_dir, 'checkpoints'))
        checkpoint_path = os.path.join(log_dir, 'checkpoints', checkpoint_name)
    elif FLAGS.restore:
        # set restore path
        assert FLAGS.run_name is not None
        log_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.run_name)
        checkpoint_path = os.path.join(log_dir, 'checkpoints', checkpoint_name)
    else:
        # makes logdir with time stamp
        log_dir = make_logdir(FLAGS.checkpoint_dir, FLAGS.run_name)
        os.mkdir(os.path.join(log_dir, 'checkpoints'))
        os.mkdir(os.path.join(log_dir, 'point_clouds'))
        # os.mkdir(os.path.join(log_dir, 'train_log'))
        # os.mkdir(os.path.join(log_dir, 'test_log'))
        checkpoint_path = os.path.join(log_dir, 'checkpoints', checkpoint_name)

    # writing arguments and git hash to info file
    file = open(os.path.join(log_dir, filename_log), "w")
    label = subprocess.check_output(["git", "describe", "--always"]).strip()
    file.write('latest git commit on this branch: ' + str(label) + '\n')
    file.write('\nFLAGS: \n')
    for key in sorted(vars(FLAGS)):
        file.write(key + ': ' + str(vars(FLAGS)[key]) + '\n')

    # count number of parameters
    if hasattr(model, 'parameters'):
        file.write('\nNumber of Model Parameters: ' + str(count_parameters(model)) + '\n')
    if hasattr(model, 'enc'):
        file.write('\nNumber of Encoder Parameters: ' + str(
            count_parameters(model.enc)) + '\n')
    if hasattr(model, 'dec'):
        file.write('\nNumber of Decoder Parameters: ' + str(
            count_parameters(model.dec)) + '\n')

    file.write('\nUNPARSED_ARGV:\n' + str(UNPARSED_ARGV))
    file.write('\n\nBASH COMMAND: \n')
    bash_command = 'python'
    for argument in sys.argv:
        bash_command += (' ' + argument)
    file.write(bash_command)
    file.close()

    # write 'git diff' output into extra file
    subprocess.call(["git diff > " + os.path.join(log_dir, filename_git_diff)], shell=True)

    return log_dir, checkpoint_path


def log_gradient_norm(tensor, variable_name):
    if variable_name not in _global_log:
        _global_log[variable_name] = []

    def log_gradient_norm_inner(gradient):
        gradient_norm = torch.norm(gradient, dim=-1)
        _global_log[variable_name].append(to_np(gradient_norm))

    tensor.register_hook(log_gradient_norm_inner)


def get_average(variable_name):
    if variable_name not in _global_log:
        return float('nan')
    elif _global_log[variable_name]:
        overall_tensor = np.concatenate(_global_log[variable_name])
        return np.mean(overall_tensor)
    else:
        return 0


def clear_data(variable_name):
    _global_log[variable_name] = []
