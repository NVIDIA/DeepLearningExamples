# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


from functools import partial
import os

import numpy as np
import torch
from torch.cuda.amp import autocast

from moflow.config import CONFIGS
from moflow.data import transform
from moflow.data.data_loader import NumpyTupleDataset

from moflow.model.model import MoFlow
from moflow.utils import check_validity, convert_predictions_to_mols, predictions_to_smiles, check_novelty
from moflow.runtime.arguments import PARSER
from moflow.runtime.common import get_newest_checkpoint, load_state
from moflow.runtime.distributed_utils import get_device
from moflow.runtime.generate import infer
from moflow.runtime.logger import MetricsLogger, setup_logging


if __name__ == '__main__':
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    args = PARSER.parse_args()
    logger = setup_logging(args)

    snapshot_path = get_newest_checkpoint(args.results_dir)
    config = CONFIGS[args.config_name]
    model = MoFlow(config)

    device = get_device(args.local_rank)
    if snapshot_path is not None:
        epoch, ln_var = load_state(snapshot_path, model, device=device)
    elif args.allow_untrained:
        epoch, ln_var = 0, 0
    else:
        raise RuntimeError('Generating molecules from an untrained network! '
                           'If this was intentional, pass --allow_untrained flag.')
    model.to(device)
    model.eval()

    if args.steps == -1:
        args.steps = 1

    acc_logger = MetricsLogger(logger)
    valid_idx = transform.get_val_ids(config, args.data_dir)
    dataset = NumpyTupleDataset.load(
        os.path.join(args.data_dir, config.dataset_config.dataset_file),
        transform=partial(transform.transform_fn, config=config),
    )
    train_idx = [t for t in range(len(dataset)) if t not in valid_idx]
    n_train = len(train_idx)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    train_x = torch.Tensor(np.array([a[0] for a in train_dataset]))
    train_adj = torch.Tensor(np.array([a[1] for a in train_dataset]))

    train_smiles = set(predictions_to_smiles(train_adj, train_x, config))


    with autocast(enabled=args.amp):
        for i in range(args.steps):
            results = infer(
                model, config, ln_var=ln_var, temp=args.temperature, batch_size=args.batch_size,
                device=device)

            mols_batch = convert_predictions_to_mols(*results, correct_validity=args.correct_validity)
            validity_info = check_validity(mols_batch)
            novel_r, abs_novel_r = check_novelty(validity_info['valid_smiles'], train_smiles, len(mols_batch))
            _, nuv = check_novelty(list(set(validity_info['valid_smiles'])), train_smiles, len(mols_batch))
            metrics = {
                'validity': validity_info['valid_ratio'],
                'novelty': novel_r,
                'uniqueness': validity_info['unique_ratio'],
                'abs_novelty': abs_novel_r,
                'abs_uniqueness': validity_info['abs_unique_ratio'],
                'nuv': nuv,
            }

            acc_logger.update(metrics)

    acc_logger.summarize(step=tuple())
