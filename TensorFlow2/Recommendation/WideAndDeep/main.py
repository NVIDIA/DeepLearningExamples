# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from trainer.model.widedeep import wide_deep_model
from trainer.run import train, evaluate
from trainer.utils.arguments import parse_args
from trainer.utils.setup import create_config


def main():
    args = parse_args()
    config = create_config(args)
    model = wide_deep_model(args)

    if args.evaluate:
        evaluate(args, model, config)
    else:
        train(args, model, config)


if __name__ == '__main__':
    main()
