# rMD17 Dataset Download Instructions

## Download

Download the rMD17 dataset from Figshare:

**URL:** https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038

Or use wget:
```bash
wget https://figshare.com/ndownloader/files/23950376 -O rmd17.tar.bz2
```

## Extract

```bash
tar -xjf rmd17.tar.bz2
```

After extraction, your `data/` folder should look like:
```
data/
├── README.md
├── rmd17.tar.bz2
└── rmd17/
    ├── npz_data/
    │   ├── rmd17_aspirin.npz
    │   ├── rmd17_azobenzene.npz
    │   ├── rmd17_benzene.npz
    │   ├── rmd17_ethanol.npz
    │   ├── rmd17_malonaldehyde.npz
    │   ├── rmd17_naphthalene.npz
    │   ├── rmd17_paracetamol.npz
    │   ├── rmd17_salicylic.npz
    │   ├── rmd17_toluene.npz
    │   └── rmd17_uracil.npz
    ├── splits/
    │   ├── index_train_01.csv ... index_train_05.csv
    │   └── index_test_01.csv ... index_test_05.csv
    └── readme.txt
```

## Data Format

The NPZ files contain:
- `nuclear_charges`: Atomic numbers for the molecule
- `coords`: Coordinates for each conformation (Å)
- `energies`: Total energy of each conformation (kcal/mol)
- `forces`: Cartesian forces (kcal/mol/Å)

## Important Notes

⚠️ **DO NOT train on more than 1000 samples!**

The structures are from molecular dynamics simulations (time series data) and are not independent samples. Training on >1000 samples leads to data leakage.

## Citation

```bibtex
@article{christensen2020role,
  title={On the role of gradients for machine learning of molecular energies and forces},
  author={Christensen, Anders S and Von Lilienfeld, O Anatole},
  journal={Machine Learning: Science and Technology},
  year={2020}
}
```
