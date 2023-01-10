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


from typing import Optional, Tuple

import numpy as np
from torch.cuda.amp import autocast
import torch

from moflow.config import CONFIGS, Config

from moflow.model.model import MoFlow
from moflow.utils import convert_predictions_to_mols, postprocess_predictions
from moflow.runtime.arguments import PARSER
from moflow.runtime.common import get_newest_checkpoint, load_state
from moflow.runtime.distributed_utils import get_device
from moflow.runtime.logger import PerformanceLogger, setup_logging


def infer(model: MoFlow, config: Config, device: torch.device, *,
          ln_var: float = 0, temp: float = 0.6, mu: Optional[torch.Tensor] = None,
          batch_size: int = 20) -> Tuple[np.ndarray, np.ndarray]:

    if mu is None:
        mu = torch.zeros(config.z_dim, dtype=torch.float32, device=device)

    sigma = temp * np.sqrt(np.exp(ln_var))
    with torch.no_grad():
        z = torch.normal(mu.reshape(-1, config.z_dim).repeat((batch_size, 1)), sigma)
        adj, x = model.reverse(z)
    x, adj = postprocess_predictions(x, adj, config=config)

    return adj, x


if __name__ == '__main__':
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    args = PARSER.parse_args()
    logger = setup_logging(args)
    perf_logger = PerformanceLogger(logger, args.batch_size, args.warmup_steps, mode='generate')
    if args.predictions_path:
        from rdkit.Chem import SmilesWriter
        smiles_writer = SmilesWriter(args.predictions_path)

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
    model.to(device=device, memory_format=torch.channels_last)
    model.eval()
    if args.jit:
        model.atom_model = torch.jit.script(model.atom_model)
        model.bond_model = torch.jit.script(model.bond_model)


    if args.steps == -1:
        args.steps = 1

    with autocast(enabled=args.amp):
        for i in range(args.steps):
            perf_logger.update()
            results = infer(
                model, config, ln_var=ln_var, temp=args.temperature, batch_size=args.batch_size,
                device=device)

            if (i + 1) % args.log_interval == 0:
                perf_logger.summarize(step=(0, i, i))
            if args.predictions_path:
                mols_batch = convert_predictions_to_mols(*results, correct_validity=args.correct_validity)
                for mol in mols_batch:
                    smiles_writer.write(mol)

    perf_logger.summarize(step=tuple())
    if args.predictions_path:
        smiles_writer.close()
