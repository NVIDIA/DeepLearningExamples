# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import re
import sys
import json
from pathlib import Path
from subprocess import Popen, PIPE

parser = argparse.ArgumentParser(description='Parse training logs')
parser.add_argument('log', help='path to log file', type=Path)
args = parser.parse_args()

content = args.log.read_bytes()

bleu = list(map(lambda x: float(x[0]), re.findall(rb'\nbleu is ((\d|.)+)', content)))

training_speed = re.findall(rb'\ntraining time for epoch (\d+): ((\d|.)+) mins \(((\d|.)+) sent/sec, ((\d|.)+) tokens/sec\)', content)
training_tokens = list(map(lambda x: float(x[5]), training_speed))
training_sentences = list(map(lambda x: float(x[3]), training_speed))

eval_speed = re.findall(rb'\neval time for epoch (\d+): ((\d|.)+) mins \(((\d|.)+) sent/sec, ((\d|.)+) tokens/sec\)', content)
if not eval_speed:
    eval_speed = re.findall(rb'\neval time for ckpt(): ((\d|.)+) mins \(((\d|.)+) sent/sec, ((\d|.)+) tokens/sec\)', content)
eval_tokens = list(map(lambda x: float(x[5]), eval_speed))
eval_sentences = list(map(lambda x: float(x[3]), eval_speed))

experiment_duration = float(re.findall(rb'\nExperiment took ((\d|.)+) min', content)[0][0])

ret = {}
ret['bleu'] = bleu
ret['training_tokens_per_sec'] = training_tokens
ret['training_sentences_per_sec'] = training_sentences
ret['eval_tokens_per_sec'] = eval_tokens
ret['eval_sentences_per_sec'] = eval_sentences
ret['duration'] = experiment_duration

print(json.dumps(ret))
