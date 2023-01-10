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


from dataclasses import asdict, dataclass, field
import json
from typing import Dict, List, Optional

from rdkit import Chem


_VALID_IDX_FILE = 'valid_idx_{}.json'
_CSV_FILE = '{}.csv'
_DATASET_FILE = '{}_relgcn_kekulized_ggnp.npz'

DUMMY_CODE = 0
CODE_TO_BOND = dict(enumerate([
    'DUMMY',
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
]))
BOND_TO_CODE = {v: k for k, v in CODE_TO_BOND.items()}
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


@dataclass
class DatasetConfig:
    dataset_name: str
    atomic_num_list: List[int]
    max_num_atoms: int
    labels: List[str]
    smiles_col: str
    code_to_atomic: Dict[int, int] = field(init=False)
    atomic_to_code: Dict[int, int] = field(init=False)
    valid_idx_file: str = field(init=False)
    csv_file: str = field(init=False)
    dataset_file: str = field(init=False)

    def __post_init__(self):
        self.valid_idx_file = _VALID_IDX_FILE.format(self.dataset_name)
        self.csv_file = _CSV_FILE.format(self.dataset_name)
        self.dataset_file = _DATASET_FILE.format(self.dataset_name)

        self.code_to_atomic = dict(enumerate(sorted([DUMMY_CODE] + self.atomic_num_list)))
        self.atomic_to_code = {v: k for k, v in self.code_to_atomic.items()}


@dataclass
class AtomFlowConfig:
    n_flow: int
    hidden_gnn: List[int]
    hidden_lin: List[int]
    n_block: int = 1
    mask_row_size_list: List[int] = field(default_factory=lambda: [1])
    mask_row_stride_list: List[int] = field(default_factory=lambda: [1])

@dataclass
class BondFlowConfig:
    hidden_ch: List[int]
    conv_lu: int
    n_squeeze: int
    n_block: int = 1
    n_flow: int = 10


@dataclass
class ModelConfig:
    atom_config: AtomFlowConfig
    bond_config: BondFlowConfig
    noise_scale: float = 0.6
    learn_dist: bool = True

@dataclass
class Config:
    dataset_config: DatasetConfig
    model_config: ModelConfig
    max_num_nodes: Optional[int] = None
    num_node_features: Optional[int] = None
    num_edge_features: int = len(CODE_TO_BOND)
    z_dim: int = field(init=False)

    def __post_init__(self):
        if self.max_num_nodes is None:
            self.max_num_nodes = self.dataset_config.max_num_atoms
        if self.num_node_features is None:
            self.num_node_features = len(self.dataset_config.code_to_atomic)
        bonds_dim = self.max_num_nodes * self.max_num_nodes * self.num_edge_features
        atoms_dim = self.max_num_nodes * self.num_node_features
        self.z_dim = bonds_dim + atoms_dim


    def save(self, path):
        self.path = path
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4, sort_keys=True)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def __repr__(self) -> str:
        return json.dumps(asdict(self), indent=4, separators=(',', ': '))


ZINC250K_CONFIG = Config(
    max_num_nodes=40,
    dataset_config=DatasetConfig(
        dataset_name='zinc250k',
        atomic_num_list=[6, 7, 8, 9, 15, 16, 17, 35, 53],
        max_num_atoms=38,
        labels=['logP', 'qed', 'SAS'],
        smiles_col='smiles',
    ),
    model_config=ModelConfig(
        AtomFlowConfig(
            n_flow=38,
            hidden_gnn=[256],
            hidden_lin=[512, 64],
        ),
        BondFlowConfig(
            n_squeeze=20,
            hidden_ch=[512, 512],
            conv_lu=2
        ),
    )
)

CONFIGS = {'zinc250k': ZINC250K_CONFIG}
