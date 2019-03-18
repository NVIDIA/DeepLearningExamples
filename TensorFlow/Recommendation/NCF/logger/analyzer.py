# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
import sys
from collections import defaultdict
import json

from logger import logger as nvl
from logger.parser import NVLogParser
from logger import tags


def collect_by_scope(loglines):

    # dict to gather run scope results
    run_stats = dict()
    epoch_stats = dict()
    iteration_stats = dict()

    # TODO: check if there is only one tag per run scope
    # gather all lines with run_scope events & variables
    run_events = dict((l.tag, l) for l in loglines if l.scope == nvl.RUN_SCOPE)

    # gather all variable tags
    run_variables = dict(k for k in run_events.items() if k[1].value is not None)

    # find all time block names
    timed_blocks = [k[:-6] for k in run_events if k.endswith('_start')]

    # measure times for the run scope
    for prefix in timed_blocks:
        # only when both start & stop are found
        # TODO: assert when not paired
        if prefix + "_start" in run_events and prefix + "_stop" in run_events:
            start = run_events[prefix + "_start"].timestamp
            stop = run_events[prefix + "_stop"].timestamp
            run_stats[prefix + "_time"] = stop - start

    # collect all variables - even nested
    for k in run_variables:
        e = run_events[k]
        if isinstance(e.value, dict):
            for d in e.value.keys():
                run_stats[k + "_" + d] = e.value[d]
        else:
            run_stats[k] = e.value

    # find epochs
    epochs = sorted(list({int(l.epoch) for l in loglines if int(l.epoch) >= 0}))
    epoch_stats['x'] = epochs

    # gather eval_accuracy
    eval_accuracy_dup = [l.value for l in loglines if l.tag == tags.EVAL_ACCURACY]
    eval_accuracy = list({l['value']:l for l in eval_accuracy_dup})
    epoch_stats['eval_accuracy'] = eval_accuracy

    # gather it_per_sec
    eval_it_per_sec = [l.value for l in loglines if l.tag == tags.PERF_IT_PER_SEC]
    #eval_it_per_sec = list({l['value']:l for l in eval_it_per_sec_dup})
    epoch_stats['it_per_sec'] = eval_it_per_sec


    # gather all epoch-iter tuples
    # TODO: l.iteration is always set to -1 in parser.py
    all_iterations = {(int(l.epoch), int(l.iteration)) for l in loglines if int(l.iteration) >= 0}

    # group by epoch
    collected_iterations = defaultdict(list)
    for el in all_iterations:
        collected_iterations[el[0]].append(el[1])

    # convert to list of lists
    iterations = [sorted(collected_iterations[k]) for k in sorted(collected_iterations.keys())]
    iteration_stats['x'] = iterations

    # gather all epoch-iter-loss triples
    all_loss_dicts = [l.value for l in loglines if l.tag == tags.TRAIN_ITERATION_LOSS]
    all_loss = {(l['epoch'], l['iteration'], l['value']) for l in all_loss_dicts}

    # group by epoch
    collected_loss = defaultdict(list)
    for el in all_loss:
        collected_loss[el[0]].append(el[2])

    # convert to list of lists
    iterations_loss = [sorted(collected_loss[k]) for k in sorted(collected_loss.keys())]
    iteration_stats['loss'] = iterations_loss

    # find epoch events and variables
    epoch_events = [l for l in loglines if l.scope == nvl.EPOCH_SCOPE]
    epoch_event_names = {l.tag for l in epoch_events}
    epoch_timed_blocks = {k[:-6] for k in epoch_event_names if k.endswith('_start')}
    epoch_variables = {l.tag for l in epoch_events if l.value is not None}

    # TODO: WIP


    return {"run" : run_stats, "epoch": epoch_stats, "iter" : iteration_stats}


def analyze(input_path, output_path=None):
    parser = NVLogParser()
    loglines, errors, worker_loglines = parser.parse_file(input_path)

    stats = collect_by_scope(worker_loglines['(0)'])

    if not output_path:
        print(json.dumps(stats, indent=4))
    else:
        with open(output_path, 'w') as f:
            json.dump(obj=stats, fp=f, indent=4)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: analyzer.py FILENAME')
        print('       tests analyzing on the file.')
        sys.exit(1)

    analyze(input_path=sys.argv[1], output_path=None)

