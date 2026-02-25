# MACE rMD17 Benchmark Reproduction

Reproduction of the rMD17 molecular dynamics benchmark from the MACE paper.

## Overview

This repository contains code to reproduce the rMD17 benchmark results from:

> Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields", NeurIPS 2022

The revised MD17 (rMD17) dataset contains 10 molecules with energies and forces recalculated at the PBE/def2-SVP level of theory.

## Target Results (from Paper)

Results from MACE paper Table 2 (Energy E in meV, Forces F in meV/Å):

![MACE rMD17 Results](assets/mace_rmd17_results.png)

**Key targets for N_train=1000:**
| Molecule | Energy MAE (meV) | Forces MAE (meV/Å) |
|----------|------------------|---------------------|
| Aspirin | 2.2 | 6.6 |
| Ethanol | 0.4 | 2.1 |
| Benzene | 0.4 | 0.3 |
| Toluene | 0.5 | 1.5 |
| ... | ... | ... |

## Molecules

- aspirin
- azobenzene
- benzene
- ethanol
- malonaldehyde
- naphthalene
- paracetamol
- salicylic
- toluene
- uracil

## Requirements

- Python 3.8+
- MACE (`pip install mace-torch`)
- ASE (Atomic Simulation Environment)
- PyTorch with CUDA support

## Usage

### Prepare datasets
```bash
python MACE_rMD17.py --prepare-only
```

### Train single molecule
```bash
python MACE_rMD17.py --molecule ethanol --device cuda
```

### Train all molecules
```bash
python MACE_rMD17.py --device cuda
```

### Generate training scripts
```bash
python MACE_rMD17.py --generate-scripts
```

## Model Configuration

Following MACE paper Appendix A.5.1:
- 2 interaction layers
- 256 channels per irrep (L=0 to L=3)
- correlation order 3 (body order 4)
- r_max: 5.0 Å
- batch size: 5
- λE=1, λF=1000

## Experiment Progress

See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for detailed experiment records and results.

## References

- [MACE Paper](https://arxiv.org/abs/2206.07697)
- [rMD17 Dataset](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)
- [MACE GitHub](https://github.com/ACEsuit/mace)

## License

MIT
