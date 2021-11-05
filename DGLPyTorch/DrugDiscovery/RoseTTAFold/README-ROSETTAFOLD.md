# *RoseTTAFold* 
This package contains deep learning models and related scripts to run RoseTTAFold.  
This repository is the official implementation of RoseTTAFold: Accurate prediction of protein structures and interactions using a 3-track network.

## Installation

1. Clone the package
```
git clone https://github.com/RosettaCommons/RoseTTAFold.git
cd RoseTTAFold
```

2. Create conda environment using `RoseTTAFold-linux.yml` file and `folding-linux.yml` file. The latter is required to run a pyrosetta version only (run_pyrosetta_ver.sh).
```
# create conda environment for RoseTTAFold
#   If your NVIDIA driver compatible with cuda11
conda env create -f RoseTTAFold-linux.yml
#   If not (but compatible with cuda10)
conda env create -f RoseTTAFold-linux-cu101.yml

# create conda environment for pyRosetta folding & running DeepAccNet
conda env create -f folding-linux.yml
```

3. Download network weights (under Rosetta-DL Software license -- please see below)  
While the code is licensed under the MIT License, the trained weights and data for RoseTTAFold are made available for non-commercial use only under the terms of the Rosetta-DL Software license. You can find details at https://files.ipd.uw.edu/pub/RoseTTAFold/Rosetta-DL_LICENSE.txt

```
wget https://files.ipd.uw.edu/pub/RoseTTAFold/weights.tar.gz
tar xfz weights.tar.gz
```

4. Download and install third-party software.
```
./install_dependencies.sh
```

5. Download sequence and structure databases
```
# uniref30 [46G]
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
mkdir -p UniRef30_2020_06
tar xfz UniRef30_2020_06_hhsuite.tar.gz -C ./UniRef30_2020_06

# BFD [272G]
wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
mkdir -p bfd
tar xfz bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -C ./bfd

# structure templates (including *_a3m.ffdata, *_a3m.ffindex) [over 100G]
wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz
tar xfz pdb100_2021Mar03.tar.gz
# for CASP14 benchmarks, we used this one: https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2020Mar11.tar.gz
```

6. Obtain a [PyRosetta licence](https://els2.comotion.uw.edu/product/pyrosetta) and install the package in the newly created `folding` conda environment ([link](http://www.pyrosetta.org/downloads)).

## Usage

```
# For monomer structure prediction
cd example
../run_[pyrosetta, e2e]_ver.sh input.fa .

# For complex modeling
# please see README file under example/complex_modeling/README for details.
python network/predict_complex.py -i paired.a3m -o complex -Ls 218 310 
```

## Expected outputs
For the pyrosetta version, user will get five final models having estimated CA rms error at the B-factor column (model/model_[1-5].crderr.pdb).  
For the end-to-end version, there will be a single PDB output having estimated residue-wise CA-lddt at the B-factor column (t000_.e2e.pdb).

## FAQ
1. Segmentation fault while running hhblits/hhsearch  
For easy install, we used a statically compiled version of hhsuite (installed through conda). Currently, we're not sure what exactly causes segmentation fault error in some cases, but we found that it might be resolved if you compile hhsuite from source and use this compiled version instead of conda version. For installation of hhsuite, please see [here](https://github.com/soedinglab/hh-suite).

2. Submitting jobs to computing nodes  
The modeling pipeline provided here (run_pyrosetta_ver.sh/run_e2e_ver.sh) is a kind of guidelines to show how RoseTTAFold works. For more efficient use of computing resources, you might want to modify the provided bash script to submit separate jobs with proper dependencies for each of steps (more cpus/memory for hhblits/hhsearch, using gpus only for running the networks, etc). 

## Links:

* [Robetta server](https://robetta.bakerlab.org/) (RoseTTAFold option)
* [RoseTTAFold models for CASP14 targets](https://files.ipd.uw.edu/pub/RoseTTAFold/casp14_models.tar.gz) [input MSA and hhsearch files are included]

## Credit to performer-pytorch and SE(3)-Transformer codes
The code in the network/performer_pytorch.py is strongly based on [this repo](https://github.com/lucidrains/performer-pytorch) which is pytorch implementation of [Performer architecture](https://arxiv.org/abs/2009.14794).
The codes in network/equivariant_attention is from the original SE(3)-Transformer [repo](https://github.com/FabianFuchsML/se3-transformer-public) which accompanies [the paper](https://arxiv.org/abs/2006.10503) 'SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks' by Fabian et al.


## References

M Baek, et al., Accurate prediction of protein structures and interactions using a 3-track network, bioRxiv (2021). [link](https://www.biorxiv.org/content/10.1101/2021.06.14.448402v1)

