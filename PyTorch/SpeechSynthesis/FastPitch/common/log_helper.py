import atexit
import glob
import os
import re
import numpy as np

from tensorboardX import SummaryWriter

import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity


def unique_dllogger_fpath(log_fpath):

    if not os.path.isfile(log_fpath):
        return log_fpath

    # Avoid overwriting old logs
    saved = sorted([int(re.search('\.(\d+)', f).group(1))
                    for f in glob.glob(f'{log_fpath}.*')])

    log_num = (saved[-1] if saved else 0) + 1
    return f'{log_fpath}.{log_num}'


def stdout_step_format(step):
    if isinstance(step, str):
        return step
    fields = []
    if len(step) > 0:
        fields.append("epoch {:>4}".format(step[0]))
    if len(step) > 1:
        fields.append("iter {:>3}".format(step[1]))
    if len(step) > 2:
        fields[-1] += "/{}".format(step[2])
    return " | ".join(fields)


def stdout_metric_format(metric, metadata, value):
    name = metadata["name"] if "name" in metadata.keys() else metric + " : "
    unit = metadata["unit"] if "unit" in metadata.keys() else None
    format = "{" + metadata["format"] + "}" if "format" in metadata.keys() else "{}"
    fields = [name, format.format(value) if value is not None else value, unit]
    fields = filter(lambda f: f is not None, fields)
    return "| " + " ".join(fields)


def init_dllogger(log_fpath=None, dummy=False):
    if dummy:
        DLLogger.init(backends=[])
        return
    DLLogger.init(backends=[
        JSONStreamBackend(Verbosity.DEFAULT, log_fpath),
        StdOutBackend(Verbosity.VERBOSE, step_format=stdout_step_format,
                      metric_format=stdout_metric_format)
        ]
    )
    DLLogger.metadata("train_loss", {"name": "loss", "format": ":>5.2f"})
    DLLogger.metadata("train_mel_loss", {"name": "mel loss", "format": ":>5.2f"})
    DLLogger.metadata("avg_train_loss", {"name": "avg train loss", "format": ":>5.2f"})
    DLLogger.metadata("avg_train_mel_loss", {"name": "avg train mel loss", "format": ":>5.2f"})
    DLLogger.metadata("val_loss", {"name": "  avg val loss", "format": ":>5.2f"})
    DLLogger.metadata("val_mel_loss", {"name": "  avg val mel loss", "format": ":>5.2f"})
    DLLogger.metadata(
        "val_ema_loss",
        {"name": "  EMA val loss", "format": ":>5.2f"})
    DLLogger.metadata(
        "val_ema_mel_loss",
        {"name": "  EMA val mel loss", "format": ":>5.2f"})
    DLLogger.metadata(
        "train_frames/s", {"name": None, "unit": "frames/s", "format": ":>10.2f"})
    DLLogger.metadata(
        "avg_train_frames/s", {"name": None, "unit": "frames/s", "format": ":>10.2f"})
    DLLogger.metadata(
        "val_frames/s", {"name": None, "unit": "frames/s", "format": ":>10.2f"})
    DLLogger.metadata(
        "val_ema_frames/s", {"name": None, "unit": "frames/s", "format": ":>10.2f"})
    DLLogger.metadata(
        "took", {"name": "took", "unit": "s", "format": ":>3.2f"})
    DLLogger.metadata("lrate_change", {"name": "lrate"})


class TBLogger(object):
    """
    xyz_dummies: stretch the screen with empty plots so the legend would
                 always fit for other plots
    """
    def __init__(self, local_rank, log_dir, name, interval=1, dummies=False):
        self.enabled = (local_rank == 0)
        self.interval = interval
        self.cache = {}
        if local_rank == 0:
            self.summary_writer = SummaryWriter(
                log_dir=os.path.join(log_dir, name),
                flush_secs=120, max_queue=200)
            atexit.register(self.summary_writer.close)
            if dummies:
                for key in ('aaa', 'zzz'):
                    self.summary_writer.add_scalar(key, 0.0, 1)

    def log_value(self, step, key, val, stat='mean'):
        if self.enabled:
            if key not in self.cache:
                self.cache[key] = []
            self.cache[key].append(val)
            if len(self.cache[key]) == self.interval:
                agg_val = getattr(np, stat)(self.cache[key])
                self.summary_writer.add_scalar(key, agg_val, step)
                del self.cache[key]

    def log_meta(self, step, meta):
        for k, v in meta.items():
            self.log_value(step, k, v.item())

    def log_grads(self, step, model):
        if self.enabled:
            norms = [p.grad.norm().item() for p in model.parameters()
                     if p.grad is not None]
            for stat in ('max', 'min', 'mean'):
                self.log_value(step, f'grad_{stat}', getattr(np, stat)(norms),
                               stat=stat)
