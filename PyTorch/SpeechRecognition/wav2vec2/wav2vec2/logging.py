# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import math
from pathlib import Path

import dllogger
import torch.distributed as dist
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from common import tb_dllogger
from common.metrics import MetricsAggregator
from common.tb_dllogger import (stdout_metric_format, stdout_step_format,
                                unique_log_fpath, TBLogger)


def init_logger(output_dir, log_file, ema_decay=0.0):
    local_rank = 0 if not dist.is_initialized() else dist.get_rank()

    if local_rank == 0:
        Path(output_dir).mkdir(parents=False, exist_ok=True)
        log_fpath = log_file or Path(output_dir, 'nvlog.json')
        dllogger.init(backends=[
            JSONStreamBackend(Verbosity.DEFAULT, log_fpath, append=True),
            JSONStreamBackend(Verbosity.DEFAULT, unique_log_fpath(log_fpath)),
            StdOutBackend(Verbosity.VERBOSE, step_format=stdout_step_format,
                          metric_format=stdout_metric_format)
        ])
        init_train_metadata()
    else:
        dllogger.init(backends=[])

    tb_train = ['train', 'train_avg']
    tb_val = ['val']
    tb_ema = [k + '_ema' for k in tb_val] if ema_decay > 0.0 else []

    subset_names = {
        'train': 'train_inner',
        'train_avg': 'train',
        'val': 'valid',
        'val_ema': 'valid_ema',
    }
    enabled = (local_rank == 0)
    tb_dllogger.tb_loggers = {
        s: TBLogger(enabled, log_dir=output_dir, name=subset_names[s])
        for s in tb_train + tb_val + tb_ema}


def init_train_metadata():
    for id_, pref in [('train', ''), ('train_avg', 'avg train '),
                      ('val', '  avg val '), ('val_ema', '  EMA val ')]:

        dllogger.metadata(f"{id_}_loss",
                          {"name": f"{pref} loss", "format": ":>6.3f"})

        dllogger.metadata(f"{id_}_accuracy",
                          {"name": f"{pref}acc", "format": ":>6.3f"})

        dllogger.metadata(f"{id_}_prob_perplexity",
                          {"name": f"{pref}p pplx", "format": ":>6.3f"})

        dllogger.metadata(f"{id_}_code_perplexity",
                          {"name": f"{pref}c pplx", "format": ":>6.3f"})

        dllogger.metadata(f"{id_}_ntokens",
                          {"name": None, "unit": "tokens", "format": ":>8.0f"})

        dllogger.metadata(f"{id_}_took",
                          {"name": "took", "unit": "s", "format": ":>3.2f"})

        dllogger.metadata(f"{id_}_ntokens/s",
                          {"name": None, "unit": "tokens/s", "format": ":>8.2f"})

        dllogger.metadata(f"{id_}_uer",
                          {"name": f"{pref} uer", "format": ":>6.2f"})

        dllogger.metadata(f"{id_}_wer",
                          {"name": f"{pref} wer", "format": ":>6.2f"})

        dllogger.metadata(f"{id_}_raw_wer",
                          {"name": f"{pref} raw wer", "format": ":>6.2f"})

        dllogger.metadata(f"{id_}_lr",
                          {"name": "lr", "format": ":>3.2e"})

        dllogger.metadata(f"{id_}_loss_scale",
                          {"name": "loss scale", "format": ":>3.2e"})


def init_infer_metadata():
    for step in ['DNN', 'data+DNN', 'data']:
        for c in [0.99, 0.95, 0.9, 0.5]:
            cs = 'avg' if c == 0.5 else f'{int(100 * c)}%'
            dllogger.metadata(f'{step.lower()}_latency_{c}',
                              {'name': f'{step} latency {cs}',
                               'format': ':>7.2f', 'unit': 'ms'})
    dllogger.metadata(
        'eval_wer', {'name': 'WER', 'format': ':>3.2f', 'unit': '%'})


class W2v2Metrics(MetricsAggregator):

    def __init__(self, benchmark_epochs, scopes=('train', 'train_avg'), cuda=True):
        super().__init__(
            benchmark_epochs=benchmark_epochs,
            benchmark_keys=('took', 'accuracy', 'loss', 'ntokens/s'),
            scopes=scopes,
            dllogger_keys=('loss', 'ntokens', 'accuracy', 'prob_perplexity',
                           'code_perplexity',
                           'took', 'loss_scale', 'lr', 'ntokens/s'),
            reduce_mean=('temp', 'prob_perplexity', 'code_perplexity'),
            reduce_last=('lr', 'loss_scale'),
            cuda=cuda)

    def accumulate(self, scopes=None):
        if 'ignore' not in self.partials or self.partials['ignore'] == 0.0:
            # compute_loss_and_accuracy
            ntokens = self.partials['ntokens']
            for k, v in self.partials.items():
                if k.startswith('loss'):
                    self.partials[k] = v / ntokens / math.log(2)  # as in fairseq

            self['accuracy'] = (self.partials.pop('correct')
                                / self.partials.pop('count'))
            part_counts = self.partial_counts
            assert part_counts['correct'] == part_counts['count'] == 1

        super().accumulate(scopes=scopes)

    def _finish_accumulating(self, scope='train'):
        super()._finish_accumulating(scope=scope)
        m = self.metrics[scope]
        count = self.metric_counts[scope]
        m['ntokens/s'] = m['ntokens'] * count['ntokens'] / m['took']


class W2v2FineTuningMetrics(MetricsAggregator):

    def __init__(
            self,
            benchmark_epochs,
            benchmark_keys=('took', 'accuracy', 'loss', 'ntokens/s'),
            scopes=('train', 'train_avg'),
            dllogger_keys=('loss', 'ntokens', 'accuracy', 'lr',
                           'prob_perplexity', 'took', 'ntokens/s', 'uer',
                           'wer', 'raw_wer'),
            reduce_mean=('temp', 'prob_perplexity', 'code_perplexity'),
            reduce_last=('lr',),
            cuda=True):
        super().__init__(
            benchmark_epochs=benchmark_epochs, benchmark_keys=benchmark_keys,
            scopes=scopes, dllogger_keys=dllogger_keys,
            reduce_mean=reduce_mean, reduce_last=reduce_last, cuda=cuda)

    def accumulate(self, scopes=None):
        if 'ignore' not in self.partials or self.partials['ignore'] == 0.0:
            # compute_loss_and_accuracy
            nsentences = self.partials['nsentences']
            for k, v in self.partials.items():
                if k.startswith('loss'):
                    self.partials[k] = v / nsentences / math.log(2)  # as in fairseq

        super().accumulate(scopes=scopes)

    def _finish_accumulating(self, scope='train'):
        super()._finish_accumulating(scope=scope)

        m = self.metrics[scope]
        count = self.metric_counts[scope]

        m['ntokens/s'] = m['ntokens'] * count['ntokens'] / m['took']

        if 'c_errs' in m:
            m['uer'] = 100 * m['c_errs'] / m['c_len']
        if 'w_errs' in m:
            m['wer'] = 100 * m['w_errs'] / m['w_len']
        if 'wv_errs' in m:
            m['raw_wer'] = 100 * m['wv_errs'] / m['w_len']
