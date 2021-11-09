import math

import numpy as np
import torch
from torch import optim


def get_optimizer(parameters, lr, args, state=None):
    if args.optimizer == "sgd":
        optimizer = get_sgd_optimizer(
            parameters,
            lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
            bn_weight_decay=args.bn_weight_decay,
        )
    elif args.optimizer == "rmsprop":
        optimizer = get_rmsprop_optimizer(
            parameters,
            lr,
            alpha=args.rmsprop_alpha,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            eps=args.rmsprop_eps,
            bn_weight_decay=args.bn_weight_decay,
        )
    if not state is None:
        optimizer.load_state_dict(state)

    return optimizer


def get_sgd_optimizer(
    parameters, lr, momentum, weight_decay, nesterov=False, bn_weight_decay=False
):
    if bn_weight_decay:
        print(" ! Weight decay applied to BN parameters ")
        params = [v for n, v in parameters]
    else:
        print(" ! Weight decay NOT applied to BN parameters ")
        bn_params = [v for n, v in parameters if "bn" in n]
        rest_params = [v for n, v in parameters if not "bn" in n]
        print(len(bn_params))
        print(len(rest_params))

        params = [
            {"params": bn_params, "weight_decay": 0},
            {"params": rest_params, "weight_decay": weight_decay},
        ]

    optimizer = torch.optim.SGD(
        params, lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
    )

    return optimizer


def get_rmsprop_optimizer(
    parameters, lr, alpha, weight_decay, momentum, eps, bn_weight_decay=False
):
    bn_params = [v for n, v in parameters if "bn" in n]
    rest_params = [v for n, v in parameters if not "bn" in n]

    params = [
        {"params": bn_params, "weight_decay": weight_decay if bn_weight_decay else 0},
        {"params": rest_params, "weight_decay": weight_decay},
    ]

    optimizer = torch.optim.RMSprop(
        params,
        lr=lr,
        alpha=alpha,
        weight_decay=weight_decay,
        momentum=momentum,
        eps=eps,
    )

    return optimizer


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn)


def lr_linear_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1 - (e / es))
        return lr

    return lr_policy(_lr_fn)


def lr_cosine_policy(base_lr, warmup_length, epochs, end_lr=0):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = end_lr + (0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - end_lr))
        return lr

    return lr_policy(_lr_fn)


def lr_exponential_policy(
    base_lr,
    warmup_length,
    epochs,
    final_multiplier=0.001,
    decay_factor=None,
    decay_step=1,
    logger=None,
):
    """Exponential lr policy. Setting decay factor parameter overrides final_multiplier"""
    es = epochs - warmup_length

    if decay_factor is not None:
        epoch_decay = decay_factor
    else:
        epoch_decay = np.power(
            2, np.log2(final_multiplier) / math.floor(es / decay_step)
        )

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay ** math.floor(e / decay_step))
        return lr

    return lr_policy(_lr_fn, logger=logger)
