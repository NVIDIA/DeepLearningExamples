# SPDX-License-Identifier: Apache-2.0
import torch.optim as opt


def optimizer_wrapped(config, params):
    optimizer_dict = {
        "Adadelta": {"var": ["lr", "rho", "eps", "weight_decay"], "func": opt.Adadelta},
        "Adagrad": {"var": ["lr", "lr_decay", "weight_decay", "eps"], "func": opt.Adagrad},
        "Adam": {"var": ["lr", "betas", "eps", "weight_decay", "amsgrad"], "func": opt.Adam},
        "AdamW": {"var": ["lr", "betas", "eps", "weight_decay", "amsgrad"], "func": opt.AdamW},
        "SparseAdam": {"var": ["lr", "betas", "eps"], "func": opt.SparseAdam},
        "Adamax": {"var": ["lr", "betas", "eps", "weight_decay"], "func": opt.Adamax},
        "ASGD": {"var": ["lr", "lambd", "alpha", "t0", "weight_decay"], "func": opt.ASGD},
        "LBFGS": {
            "var": ["lr", "max_iter", "max_eval", "tolerance_grad", "tolerance_change", "history_size", "line_search_fn"],
            "func": opt.LBFGS,
        },
        "RMSprop": {"var": ["lr", "alpha", "eps", "weight_decay", "momentum", "centered"], "func": opt.RMSprop},
        "Rprop": {"var": ["lr", "etas", "step_sizes"], "func": opt.Rprop},
        "SGD": {"var": ["lr", "momentum", "weight_decay", "dampening", "nesterov"], "func": opt.SGD},
    }
    kwargs = {key: config.optimizer.get(key) for key in optimizer_dict[config.optimizer.name]["var"]}
    return optimizer_dict[config.optimizer.name]["func"](params, **kwargs)
