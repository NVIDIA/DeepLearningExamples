# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import torch


def add_parser_arguments(parser):
    parser.add_argument(
        "--checkpoint-path", metavar="<path>", help="checkpoint filename"
    )
    parser.add_argument(
        "--weight-path", metavar="<path>", help="name of file in which to store weights"
    )
    parser.add_argument("--ema", action="store_true", default=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    add_parser_arguments(parser)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))

    key = "state_dict" if not args.ema else "ema_state_dict"
    model_state_dict = {
        k[len("module.") :] if "module." in k else k: v
        for k, v in checkpoint["state_dict"].items()
    }
    print(f"Loaded model, acc : {checkpoint['best_prec1']}")

    torch.save(model_state_dict, args.weight_path)
