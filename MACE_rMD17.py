"""
MACE rMD17 Benchmark Reproduction Script
=========================================
This script reproduces the rMD17 benchmark from the MACE paper.

The revised MD17 (rMD17) dataset contains 10 molecules with energies and forces
recalculated at the PBE/def2-SVP level of theory.

Reference:
- MACE Paper: Batatia et al., "MACE: Higher Order Equivariant Message Passing 
  Neural Networks for Fast and Accurate Force Fields", NeurIPS 2022
- rMD17: Christensen & von Lilienfeld, "On the role of gradients for machine 
  learning of molecular energies and forces"
"""

import os
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from ase import Atoms
from ase.io import write

# Configuration
DATA_DIR = Path("/home/kai/Desktop/mace-rmd17-benchmark/data/rmd17")
OUTPUT_DIR = Path("/home/kai/Desktop/mace-rmd17-benchmark/results")
NPZ_DIR = DATA_DIR / "npz_data"
SPLITS_DIR = DATA_DIR / "splits"

# rMD17 molecules
MOLECULES = [
    "aspirin",
    "azobenzene", 
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "paracetamol",
    "salicylic",
    "toluene",
    "uracil",
]

# Atomic number to element symbol mapping
Z_TO_SYMBOL = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S'
}

# Training configuration (following MACE paper Appendix A.5.1)
# Paper: 950 training, 50 validation from 1000 total
# Paper settings:
#   - 256 uncoupled channels (hidden_irreps='256x0e')
#   - lmax=3 (spherical harmonics)
#   - correlation=3 (body order 4)
#   - batch_size=5
#   - λE=1, λF=1000
#   - float32 precision
TRAIN_SIZE = 1000  # As per rMD17 recommendation
SPLIT_INDEX = 1    # Use split 01 (can be 1-5)


def load_npz_data(molecule_name):
    """Load molecule data from NPZ file."""
    npz_path = NPZ_DIR / f"rmd17_{molecule_name}.npz"
    data = np.load(npz_path)
    return {
        'nuclear_charges': data['nuclear_charges'],
        'coords': data['coords'],
        'energies': data['energies'],
        'forces': data['forces'],
    }


def load_split_indices(split_idx=1):
    """Load train/test split indices."""
    train_path = SPLITS_DIR / f"index_train_{split_idx:02d}.csv"
    test_path = SPLITS_DIR / f"index_test_{split_idx:02d}.csv"
    
    train_indices = np.loadtxt(train_path, dtype=int)
    test_indices = np.loadtxt(test_path, dtype=int)
    
    return train_indices, test_indices


def create_xyz_dataset(molecule_name, split_idx=1):
    """
    Convert NPZ data to XYZ format for MACE training.
    
    The rMD17 dataset provides:
    - energies in kcal/mol
    - forces in kcal/mol/Å
    
    MACE typically uses eV and eV/Å, so we convert:
    - 1 kcal/mol = 0.0433641 eV
    """
    print(f"\nProcessing {molecule_name}...")
    
    # Load data
    data = load_npz_data(molecule_name)
    train_indices, test_indices = load_split_indices(split_idx)
    
    # Conversion factor: kcal/mol to eV
    KCAL_TO_EV = 0.0433641
    
    nuclear_charges = data['nuclear_charges']
    coords = data['coords']
    energies = data['energies'] * KCAL_TO_EV  # Convert to eV
    forces = data['forces'] * KCAL_TO_EV      # Convert to eV/Å
    
    # Create output directory for this molecule
    mol_dir = OUTPUT_DIR / molecule_name
    mol_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training set
    train_atoms_list = []
    for idx in train_indices:
        symbols = [Z_TO_SYMBOL[z] for z in nuclear_charges]
        atoms = Atoms(symbols=symbols, positions=coords[idx])
        atoms.info['energy'] = energies[idx]
        atoms.info['config_type'] = 'train'
        atoms.arrays['forces'] = forces[idx]
        train_atoms_list.append(atoms)
    
    # Create test set
    test_atoms_list = []
    for idx in test_indices:
        symbols = [Z_TO_SYMBOL[z] for z in nuclear_charges]
        atoms = Atoms(symbols=symbols, positions=coords[idx])
        atoms.info['energy'] = energies[idx]
        atoms.info['config_type'] = 'test'
        atoms.arrays['forces'] = forces[idx]
        test_atoms_list.append(atoms)
    
    # Write XYZ files
    train_path = mol_dir / "train.xyz"
    test_path = mol_dir / "test.xyz"
    
    write(train_path, train_atoms_list, format='extxyz')
    write(test_path, test_atoms_list, format='extxyz')
    
    print(f"  Training samples: {len(train_atoms_list)}")
    print(f"  Test samples: {len(test_atoms_list)}")
    print(f"  Files saved to: {mol_dir}")
    
    return mol_dir, train_path, test_path


def get_mace_training_command(molecule_name, train_path, test_path, output_dir, timestamp=None, channels=128):
    """
    Generate MACE training command following paper settings (Appendix A.5.1).
    
    MACE paper rMD17 settings:
    - Model: MACE with 2 interaction layers
    - Hidden irreps: NxLe (N channels per L level)
    - lmax: 3 (spherical harmonics up to l=3)
    - r_max: 5.0 Å
    - Batch size: 5
    - Training: 950, Validation: 50
    - λE=1, λF=1000
    - Radial MLP: [64, 64, 64, 1024]
    - float32 precision
    
    Args:
        channels: Number of channels per L level (default: 128)
    """
    # 加入時間戳讓每次執行的 log 獨立
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_name = f"MACE_{molecule_name}_ch{channels}_{timestamp}"
    
    cmd = [
        "mace_run_train",
        f"--name={model_name}",
        f"--train_file={train_path}",
        f"--test_file={test_path}",
        f"--results_dir={output_dir}",
        f"--checkpoints_dir={output_dir}/checkpoints",
        f"--model_dir={output_dir}",
        # Paper: 950 train, 50 valid from 1000 total
        "--valid_fraction=0.05",
        "--model=MACE",
        # Paper: 2 layers
        "--num_interactions=2",
        # L=3 equivariant features with configurable channels
        f"--hidden_irreps={channels}x0e+{channels}x1o+{channels}x2e+{channels}x3o",
        # Paper: correlation order 3 (body order 4)
        "--correlation=3",
        # Paper: spherical harmonics up to l=3
        "--max_ell=3",
        # Paper: 5 Å cutoff
        "--r_max=5.0",
        # Paper: 8 Bessel basis, polynomial envelope p=5
        "--num_radial_basis=8",
        "--num_cutoff_basis=5",
        # Paper: radial MLP [64, 64, 64, 1024] - but last 1024 is internal
        "--radial_MLP=[64,64,64]",
        # Paper: batch size 5
        "--batch_size=5",
        "--max_num_epochs=1000",
        # Paper: Stage Two (SWA) enabled
        "--stage_two",
        "--start_stage_two=800",
        # Paper: EMA with 0.99 decay
        "--ema",
        "--ema_decay=0.99",
        # Paper: AMSGrad optimizer
        "--amsgrad",
        # Paper: float32 precision for rMD17
        "--default_dtype=float32",
        "--device=cuda",
        "--seed=42",
        "--energy_key=energy",
        "--forces_key=forces",
        "--E0s=average",
        # Paper: learning rate 0.01
        "--lr=0.01",
        "--scheduler_patience=50",
        # Paper: λE=1, λF=1000
        "--energy_weight=1.0",
        "--forces_weight=1000.0",
        # Stage Two: λE=1000, λF=1000 (論文維持 forces_weight=1000)
        "--swa_energy_weight=1000.0",
        "--swa_forces_weight=1000.0",
        # Paper reports MAE
        "--error_table=PerAtomMAE",
    ]
    
    return cmd


def train_single_molecule(molecule_name, split_idx=1, device="cuda", channels=128):
    """Train MACE model on a single molecule.
    
    Args:
        molecule_name: Name of the molecule
        split_idx: Which train/test split to use (1-5)
        device: cuda or cpu
        channels: Number of channels per L level (128 or 256)
    """
    
    # 生成時間戳，讓每次執行獨立
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create XYZ datasets
    mol_dir, train_path, test_path = create_xyz_dataset(molecule_name, split_idx)
    
    # 建立本次執行的獨立目錄
    run_dir = mol_dir / f"run_ch{channels}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training command with timestamp and channels
    cmd = get_mace_training_command(molecule_name, train_path, test_path, run_dir, timestamp, channels)
    
    # Update device
    cmd = [c if not c.startswith("--device=") else f"--device={device}" for c in cmd]
    
    print(f"\n{'='*60}")
    print(f"Training MACE on {molecule_name}")
    print(f"Run ID: {timestamp}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}")
    print("Command:")
    print(" \\\n    ".join(cmd))
    print()
    
    # Run training
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    return result.returncode == 0


def train_all_molecules(split_idx=1, device="cuda"):
    """Train MACE on all rMD17 molecules."""
    
    results = {}
    
    for molecule in MOLECULES:
        success = train_single_molecule(molecule, split_idx, device)
        results[molecule] = "Success" if success else "Failed"
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for mol, status in results.items():
        print(f"  {mol}: {status}")
    
    return results


def prepare_all_datasets(split_idx=1):
    """Prepare XYZ datasets for all molecules without training."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Preparing rMD17 datasets for MACE training")
    print(f"Split index: {split_idx}")
    print("="*60)
    
    for molecule in MOLECULES:
        create_xyz_dataset(molecule, split_idx)
    
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print(f"All XYZ files saved to: {OUTPUT_DIR}")
    print("="*60)


def generate_training_script(molecule_name, split_idx=1, device="cuda"):
    """Generate a bash script for training a single molecule."""
    
    mol_dir, train_path, test_path = create_xyz_dataset(molecule_name, split_idx)
    cmd = get_mace_training_command(molecule_name, train_path, test_path, mol_dir)
    cmd = [c if not c.startswith("--device=") else f"--device={device}" for c in cmd]
    
    script_content = "#!/bin/bash\n\n"
    script_content += f"# MACE training script for {molecule_name}\n"
    script_content += f"# Generated for rMD17 benchmark reproduction\n\n"
    script_content += " \\\n    ".join(cmd) + "\n"
    
    script_path = mol_dir / f"train_{molecule_name}.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"Training script saved to: {script_path}")
    
    return script_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MACE rMD17 Benchmark")
    parser.add_argument("--molecule", type=str, default=None,
                        help="Specific molecule to train (default: all)")
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Split index to use (default: 1)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"],
                        help="Device to train on (default: cuda)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only prepare datasets, don't train")
    parser.add_argument("--generate-scripts", action="store_true",
                        help="Generate training scripts instead of training")
    parser.add_argument("--channels", type=int, default=128, choices=[128, 256],
                        help="Number of channels per L level (default: 128)")
    
    args = parser.parse_args()
    
    if args.prepare_only:
        prepare_all_datasets(args.split)
    elif args.generate_scripts:
        if args.molecule:
            generate_training_script(args.molecule, args.split, args.device)
        else:
            for mol in MOLECULES:
                generate_training_script(mol, args.split, args.device)
    elif args.molecule:
        train_single_molecule(args.molecule, args.split, args.device, args.channels)
    else:
        train_all_molecules(args.split, args.device)
