import atexit
import glob
import re
from itertools import product
from pathlib import Path

import dllogger
import torch
import numpy as np
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
from torch.utils.tensorboard import SummaryWriter


tb_loggers = {}


class TBLogger:
    """
    xyz_dummies: stretch the screen with empty plots so the legend would
                 always fit for other plots
    """
    def __init__(self, enabled, log_dir, name, interval=1, dummies=True):
        self.enabled = enabled
        self.interval = interval
        self.cache = {}
        if self.enabled:
            self.summary_writer = SummaryWriter(
                log_dir=Path(log_dir, name), flush_secs=120, max_queue=200)
            atexit.register(self.summary_writer.close)
            if dummies:
                for key in ('_', '✕'):
                    self.summary_writer.add_scalar(key, 0.0, 1)

    def log(self, step, data):
        for k, v in data.items():
            self.log_value(step, k, v.item() if type(v) is torch.Tensor else v)

    def log_value(self, step, key, val, stat='mean'):
        if self.enabled:
            if key not in self.cache:
                self.cache[key] = []
            self.cache[key].append(val)
            if len(self.cache[key]) == self.interval:
                agg_val = getattr(np, stat)(self.cache[key])
                self.summary_writer.add_scalar(key, agg_val, step)
                del self.cache[key]

    def log_grads(self, step, model):
        if self.enabled:
            norms = [p.grad.norm().item() for p in model.parameters()
                     if p.grad is not None]
            for stat in ('max', 'min', 'mean'):
                self.log_value(step, f'grad_{stat}', getattr(np, stat)(norms),
                               stat=stat)


def unique_log_fpath(fpath):
    """Have a unique log filename for every separate run"""
    log_num = max([0] + [int(re.search("\.(\d+)", Path(f).suffix).group(1))
                         for f in glob.glob(f"{fpath}.*")])
    return f"{fpath}.{log_num + 1}"


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
    name = metadata.get("name", metric + " : ")
    unit = metadata.get("unit", None)
    format = f'{{{metadata.get("format", "")}}}'
    fields = [name, format.format(value) if value is not None else value, unit]
    fields = [f for f in fields if f is not None]
    return "| " + " ".join(fields)


def init(log_fpath, log_dir, enabled=True, tb_subsets=[], **tb_kw):

    if enabled:
        backends = [
            JSONStreamBackend(Verbosity.DEFAULT, log_fpath, append=True),
            JSONStreamBackend(Verbosity.DEFAULT, unique_log_fpath(log_fpath)),
            StdOutBackend(Verbosity.VERBOSE, step_format=stdout_step_format,
                          metric_format=stdout_metric_format)
        ]
    else:
        backends = []

    dllogger.init(backends=backends)
    dllogger.metadata("train_lrate", {"name": "lrate", "unit": None, "format": ":>3.2e"})

    for id_, pref in [('train', ''), ('train_avg', 'avg train '),
                      ('val', '  avg val '), ('val_ema', '  EMA val ')]:

        dllogger.metadata(f"{id_}_loss",
                          {"name": f"{pref}loss", "unit": None, "format": ":>5.2f"})
        dllogger.metadata(f"{id_}_mel_loss",
                          {"name": f"{pref}mel loss", "unit": None, "format": ":>5.2f"})

        dllogger.metadata(f"{id_}_kl_loss",
                          {"name": f"{pref}kl loss", "unit": None, "format": ":>5.5f"})
        dllogger.metadata(f"{id_}_kl_weight",
                          {"name": f"{pref}kl weight", "unit": None, "format": ":>5.5f"})

        dllogger.metadata(f"{id_}_frames/s",
                          {"name": None, "unit": "frames/s", "format": ":>10.2f"})
        dllogger.metadata(f"{id_}_took",
                          {"name": "took", "unit": "s", "format": ":>3.2f"})

    global tb_loggers
    tb_loggers = {s: TBLogger(enabled, log_dir, name=s, **tb_kw)
                  for s in tb_subsets}


def init_inference_metadata(batch_size=None):

    modalities = [('latency', 's', ':>10.5f'), ('RTF', 'x', ':>10.2f'),
                  ('frames/s', 'frames/s', ':>10.2f'), ('samples/s', 'samples/s', ':>10.2f'),
                  ('letters/s', 'letters/s', ':>10.2f'), ('tokens/s', 'tokens/s', ':>10.2f')]

    if batch_size is not None:
        modalities.append((f'RTF@{batch_size}', 'x', ':>10.2f'))

    percs = ['', 'avg', '90%', '95%', '99%']
    models = ['', 'fastpitch', 'waveglow', 'hifigan']

    for perc, model, (mod, unit, fmt) in product(percs, models, modalities):
        name = f'{perc} {model} {mod}'.strip().replace('  ', ' ')
        dllogger.metadata(name.replace(' ', '_'),
                          {'name': f'{name: <26}', 'unit': unit, 'format': fmt})


def log(step, tb_total_steps=None, data={}, subset='train'):
    if tb_total_steps is not None:
        tb_loggers[subset].log(tb_total_steps, data)

    if subset != '':
        data = {f'{subset}_{key}': v for key, v in data.items()}
    dllogger.log(step, data=data)


def log_grads_tb(tb_total_steps, grads, tb_subset='train'):
    tb_loggers[tb_subset].log_grads(tb_total_steps, grads)


def parameters(data, verbosity=0, tb_subset=None):
    for k, v in data.items():
        dllogger.log(step="PARAMETER", data={k: v}, verbosity=verbosity)

    if tb_subset is not None and tb_loggers[tb_subset].enabled:
        tb_data = {k: v for k, v in data.items()
                   if type(v) in (str, bool, int, float)}
        tb_loggers[tb_subset].summary_writer.add_hparams(tb_data, {})


def flush():
    dllogger.flush()
    for tbl in tb_loggers.values():
        if tbl.enabled:
            tbl.summary_writer.flush()
