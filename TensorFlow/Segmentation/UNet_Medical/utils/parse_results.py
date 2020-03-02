# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
import os
import numpy as np

import argparse


def process_performance_stats(timestamps, batch_size):
    timestamps_ms = 1000 * timestamps
    timestamps_ms = timestamps_ms[timestamps_ms > 0]
    latency_ms = timestamps_ms.mean()
    std = timestamps_ms.std()
    n = np.sqrt(len(timestamps_ms))
    throughput_imgps = (1000.0 * batch_size / timestamps_ms).mean()
    print('Throughput Avg:', round(throughput_imgps, 3), 'img/s')
    print('Latency Avg:', round(latency_ms, 3), 'ms')
    for ci, lvl in zip(["90%:", "95%:", "99%:"],
                       [1.645, 1.960, 2.576]):
        print("Latency", ci, round(latency_ms + lvl * std / n, 3), "ms")
    return float(throughput_imgps), float(latency_ms)


def parse_convergence_results(path, environment):
    dice_scores = []
    ce_scores = []
    logfiles = [f for f in os.listdir(path) if "log" in f and environment in f]
    if not logfiles:
        raise FileNotFoundError("No logfile found at {}".format(path))
    for logfile in logfiles:
        with open(os.path.join(path, logfile), "r") as f:
            content = f.readlines()
        if "eval_dice_score" not in content[-1]:
            print("Evaluation score not found. The file", logfile, "might be corrupted.")
            continue
        dice_scores.append(float([val for val in content[-1].split()
                                  if "eval_dice_score" in val][0].split(":")[1]))
        ce_scores.append(float([val for val in content[-1].split()
                                if "eval_ce_loss" in val][0].split(":")[1]))
    if dice_scores:
        print("Evaluation dice score:", sum(dice_scores) / len(dice_scores))
        print("Evaluation cross-entropy loss:", sum(ce_scores) / len(ce_scores))
    else:
        print("All logfiles were corrupted, no loss was obtained.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UNet-medical-utils")

    parser.add_argument('--exec_mode',
                        choices=['convergence', 'benchmark'],
                        type=str,
                        help="""Which execution mode to run the model into""")

    parser.add_argument('--model_dir',
                        type=str,
                        required=True)

    parser.add_argument('--env',
                        choices=['FP32_1GPU', 'FP32_8GPU', 'TF-AMP_1GPU', 'TF-AMP_8GPU'],
                        type=str,
                        required=True)

    args = parser.parse_args()
    if args.exec_mode == 'convergence':
        parse_convergence_results(path=args.model_dir, environment=args.env)
    elif args.exec_mode == 'benchmark':
        pass
    print()
