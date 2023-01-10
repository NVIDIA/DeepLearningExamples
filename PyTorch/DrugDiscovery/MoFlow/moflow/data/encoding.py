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


from typing import Tuple
import numpy as np
from rdkit import Chem

from moflow.config import BOND_TO_CODE, DUMMY_CODE


class MolEncoder:
    """Encodes atoms and adjecency matrix.

    Args:
        out_size (int): It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
    """

    def __init__(self, out_size: int):
        super(MolEncoder, self).__init__()
        self.out_size = out_size

    def encode_mol(self, mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
        """get input features

        Args:
            mol (Mol):

        Returns:

        """
        mol = self._standardize_mol(mol)
        self._check_num_atoms(mol)
        atom_array = self.construct_atomic_number_array(mol)
        adj_array = self.construct_discrete_edge_matrix(mol)
        return atom_array, adj_array

    def _standardize_mol(self, mol: Chem.Mol) -> Chem.Mol:
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False,
                                            canonical=True)
        mol = Chem.MolFromSmiles(canonical_smiles)
        Chem.Kekulize(mol)
        return mol

    def _check_num_atoms(self, mol: Chem.Mol) -> None:
        """Check number of atoms in `mol` does not exceed `out_size`"""
        num_atoms = mol.GetNumAtoms()
        if num_atoms > self.out_size:
            raise EncodingError(f'Number of atoms in mol {num_atoms} exceeds num_max_atoms {self.out_size}')


    def construct_atomic_number_array(self, mol: Chem.Mol) -> np.ndarray:
        """Returns atomic numbers of atoms consisting a molecule.

        Args:
            mol (rdkit.Chem.Mol): Input molecule.

        Returns:
            numpy.ndarray: an array consisting of atomic numbers
                of atoms in the molecule.
        """

        atom_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
        n_atom = len(atom_list)
        if self.out_size < n_atom:
            raise EncodingError(f'out_size {self.out_size} is smaller than number of atoms in mol {n_atom}')
        atom_array = np.full(self.out_size, DUMMY_CODE, dtype=np.uint8)
        atom_array[:n_atom] = atom_list
        return atom_array

        
    def construct_discrete_edge_matrix(self, mol: Chem.Mol) -> np.ndarray:
        """Returns the edge-type dependent adjacency matrix of the given molecule.

        Args:
            mol (rdkit.Chem.Mol): Input molecule.

        Returns:
            adj_array (numpy.ndarray): The adjacent matrix of the input molecule.
                It is symmetrical 2-dimensional array with shape (out_size, out_size),
                filled with integers representing bond types. It two atoms are not
                conncted, DUMMY_CODE is used instead.
        """
        if mol is None:
            raise EncodingError('mol is None')
        n_atom = mol.GetNumAtoms()

        if self.out_size < n_atom:
            raise EncodingError(f'out_size {self.out_size} is smaller than number of atoms in mol {n_atom}')

        adjs = np.full((self.out_size, self.out_size), DUMMY_CODE, dtype=np.uint8)

        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            # we need to use code here - bond types are rdkit objects
            code = BOND_TO_CODE[bond_type]
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adjs[[i, j], [j, i]] = code
        return adjs


class EncodingError(Exception):
    pass
