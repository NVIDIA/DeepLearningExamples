# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import argparse


def parse_convergence_results(path, environment):
    whole_tumor = []
    tumor_core = []
    peritumoral_edema = []
    enhancing_tumor = []
    mean_dice = []
    logfiles = [f for f in os.listdir(path) if "log" in f and environment in f]
    if not logfiles:
        raise FileNotFoundError("No logfile found at {}".format(path))
    for logfile in logfiles:
        with open(os.path.join(path, logfile), "r") as f:
            content = f.readlines()
        if "TumorCore" not in content[-1]:
            print("Evaluation score not found. The file", logfile, "might be corrupted.")
            continue
        content = content[-1].split("()")[1]
        whole_tumor.append(float([val for val in content.split("  ")
                                  if "WholeTumor" in val][0].split()[-1]))
        tumor_core.append(float([val for val in content.split("  ")
                                 if "TumorCore" in val][0].split()[-1]))
        peritumoral_edema.append(float([val for val in content.split("  ")
                                        if "PeritumoralEdema" in val][0].split()[-1]))
        enhancing_tumor.append(float([val for val in content.split("  ")
                                      if "EnhancingTumor" in val][0].split()[-1]))
        mean_dice.append(float([val for val in content.split("  ")
                                if "MeanDice" in val][0].split()[-1]))

    if whole_tumor:
        print("Evaluation average dice score:", sum(mean_dice) / len(mean_dice))
        print("Evaluation whole tumor dice score:", sum(whole_tumor) / len(whole_tumor))
        print("Evaluation tumor core dice score:", sum(tumor_core) / len(tumor_core))
        print("Evaluation peritumoral edema dice score:", sum(peritumoral_edema) / len(peritumoral_edema))
        print("Evaluation enhancing tumor dice score:", sum(enhancing_tumor) / len(enhancing_tumor))
    else:
        print("All logfiles were corrupted, no loss was obtained.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=str,
                        required=True)
    parser.add_argument('--env',
                        type=str,
                        required=True)

    args = parser.parse_args()
    parse_convergence_results(path=args.model_dir, environment=args.env)
