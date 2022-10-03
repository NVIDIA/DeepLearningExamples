import logging
import os
import sys

import dllogger
import horovod.mxnet as hvd


def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    if len(step) == 0:
        s = "Summary:"
    return s


def setup_logging(args):
    logging.basicConfig(level=logging.DEBUG, format='{asctime}:{levelname}: {message}', style='{')
    if hvd.rank() == 0:
        logging_dir = args.logdir if args.logdir is not None else args.workspace
        dllogger.init(backends=[
            dllogger.StdOutBackend(dllogger.Verbosity.DEFAULT, step_format=format_step),
            dllogger.JSONStreamBackend(
                dllogger.Verbosity.VERBOSE, os.path.join(logging_dir, args.dllogger_log)),
        ])
    else:
        dllogger.init([])

    dllogger.metadata("val.accuracy", {"unit": None})
    dllogger.metadata("val.top_k_accuracy_5", {"unit": None})
    dllogger.metadata("train.ips", {"unit": "images/s"})
    dllogger.metadata("val.ips", {"unit": "images/s"})
    dllogger.metadata("val.latency_50", {"unit": "s"})
    dllogger.metadata("val.latency_90", {"unit": "s"})
    dllogger.metadata("val.latency_avg", {"unit": "s"})
