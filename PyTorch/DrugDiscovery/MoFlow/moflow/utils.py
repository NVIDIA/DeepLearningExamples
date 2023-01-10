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


# Copyright 2020 Chengxi Zang
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem
import torch

from moflow.config import Config, ATOM_VALENCY, CODE_TO_BOND, DUMMY_CODE


def postprocess_predictions(x: Union[torch.Tensor, np.ndarray], adj: Union[torch.Tensor, np.ndarray], config: Config) -> Tuple[np.ndarray, np.ndarray]:
    assert x.ndim == 3 and adj.ndim == 4, 'expected batched predictions'
    n = config.dataset_config.max_num_atoms
    adj = adj[:, :, :n, :n]
    x = x[:, :n]

    atoms = torch.argmax(x, dim=2)
    atoms = _to_numpy_array(atoms)

    adj = torch.argmax(adj, dim=1)
    adj = _to_numpy_array(adj)

    decoded = np.zeros_like(atoms)
    for code, atomic_num in config.dataset_config.code_to_atomic.items():
        decoded[atoms == code] = atomic_num

    return decoded, adj


def convert_predictions_to_mols(adj: np.ndarray, x: np.ndarray, correct_validity: bool = False) -> List[Chem.Mol]:
    molecules = [construct_mol(x_elem, adj_elem) for x_elem, adj_elem in zip(x, adj)]

    if correct_validity:
        molecules = [correct_mol(mol) for mol in molecules]
    return molecules


def construct_mol(atoms: np.ndarray, adj: np.ndarray) -> Chem.Mol:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    atoms_exist = (atoms != 0)
    atoms = atoms[atoms_exist]
    adj = adj[atoms_exist][:, atoms_exist]

    mol = Chem.RWMol()

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atom)))

    for start, end in zip(*np.where(adj != DUMMY_CODE)):
        if start > end:
            mol.AddBond(int(start), int(end), CODE_TO_BOND[int(adj[start, end])])
            # add formal charge to atom: e.g. [O+], [N+] [S+]
            # not support [O-], [N-] [S-]  [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def valid_mol(x: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
    if x is None:
        # RDKit wasn't able to create the mol
        return None
    smi = Chem.MolToSmiles(x, isomericSmiles=True)
    if len(smi) == 0 or '.' in smi:
        # Mol is empty or fragmented
        return None
    reloaded = Chem.MolFromSmiles(smi)
    # if smiles is invalid - it will be None, otherwise mol is valid
    return reloaded


def check_valency(mol: Chem.Mol) -> Tuple[bool, List[int]]:
    """Checks that no atoms in the mol have exceeded their possible
    valency. Returns True if no valency issues, False otherwise
    plus information about problematic atom.
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(mol: Chem.Mol) -> Chem.Mol:
    flag, atomid_valence = check_valency(mol)
    while not flag:
        assert len(atomid_valence) == 2
        idx = atomid_valence[0]
        v = atomid_valence[1]
        queue = []
        for b in mol.GetAtomWithIdx(idx).GetBonds():
            queue.append(
                (b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
            )
        queue.sort(key=lambda tup: tup[1], reverse=True)
        if len(queue) > 0:
            start = queue[0][2]
            end = queue[0][3]
            t = queue[0][1] - 1
            mol.RemoveBond(start, end)
            if t >= 1:
                mol.AddBond(start, end, CODE_TO_BOND[t])
        flag, atomid_valence = check_valency(mol)

    # if mol is fragmented, select the largest fragment
    mols = Chem.GetMolFrags(mol, asMols=True)
    mol = max(mols, key=lambda m: m.GetNumAtoms())

    return mol


def predictions_to_smiles(adj: torch.Tensor, x: torch.Tensor, config: Config) -> List[str]:
    x, adj = postprocess_predictions(x, adj, config=config)
    valid = [Chem.MolToSmiles(construct_mol(x_elem, adj_elem), isomericSmiles=True)
             for x_elem, adj_elem in zip(x, adj)]
    return valid


def check_validity(molecules: List[Chem.Mol]) -> dict:
    valid = [valid_mol(mol) for mol in molecules]
    valid = [mol for mol in valid if mol is not None]

    n_mols = len(molecules)
    valid_ratio = len(valid) / n_mols
    valid_smiles = [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in valid]
    unique_smiles = list(set(valid_smiles))
    unique_ratio = 0.
    if len(valid) > 0:
        unique_ratio = len(unique_smiles) / len(valid)
    valid_mols = [Chem.MolFromSmiles(s) for s in valid_smiles]
    abs_unique_ratio = len(unique_smiles) / n_mols

    results = dict()
    results['valid_mols'] = valid_mols
    results['valid_smiles'] = valid_smiles
    results['valid_ratio'] = valid_ratio * 100
    results['unique_ratio'] = unique_ratio * 100
    results['abs_unique_ratio'] = abs_unique_ratio * 100

    return results


def check_novelty(gen_smiles: List[str], train_smiles: List[str], n_generated_mols: int):
    if len(gen_smiles) == 0:
        novel_ratio = 0.
        abs_novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]
        novel = len(gen_smiles) - sum(duplicates)
        novel_ratio = novel * 100. / len(gen_smiles)
        abs_novel_ratio = novel * 100. / n_generated_mols
    return novel_ratio, abs_novel_ratio


def _to_numpy_array(a):
    if isinstance(a, torch.Tensor):
        a = a.cpu().detach().numpy()
    elif isinstance(a, np.ndarray):
        pass
    else:
        raise TypeError("a ({}) is not a torch.Tensor".format(type(a)))
    return a
