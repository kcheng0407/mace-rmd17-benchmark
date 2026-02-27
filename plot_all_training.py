"""Batch plot training curves for all rMD17 molecules."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

# Paper Table 2 results (E: meV total, F: meV/Å)
PAPER_RESULTS = {
    'aspirin':       {'E': 2.2, 'F': 6.6},
    'azobenzene':    {'E': 1.2, 'F': 3.0},
    'benzene':       {'E': 0.4, 'F': 0.3},
    'ethanol':       {'E': 0.4, 'F': 2.1},
    'malonaldehyde': {'E': 0.8, 'F': 4.1},
    'naphthalene':   {'E': 0.5, 'F': 1.6},
    'paracetamol':   {'E': 1.3, 'F': 4.8},
    'salicylic':     {'E': 0.9, 'F': 3.1},
    'toluene':       {'E': 0.5, 'F': 1.5},
    'uracil':        {'E': 0.5, 'F': 2.1},
}

MOLECULE_ATOMS = {
    'aspirin': 21,
    'azobenzene': 24,
    'benzene': 12,
    'ethanol': 9,
    'malonaldehyde': 9,
    'naphthalene': 18,
    'paracetamol': 20,
    'salicylic': 16,
    'toluene': 15,
    'uracil': 12,
}

MOLECULES = ['aspirin', 'azobenzene', 'benzene', 'ethanol', 'malonaldehyde', 
             'naphthalene', 'paracetamol', 'salicylic', 'toluene', 'uracil']

BASE_DIR = Path("/home/kai/Desktop/mace-rmd17-benchmark")
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

def find_latest_run(molecule, channels=256):
    """Find latest run directory."""
    mol_dir = RESULTS_DIR / molecule
    pattern = f"run_ch{channels}_*"
    runs = sorted(mol_dir.glob(pattern))
    return runs[-1] if runs else None

def parse_training_log(train_file):
    """Parse MACE training log, return (epochs, mae_e_per_atom, mae_f, losses)."""
    epochs, mae_e_per_atom, mae_f, losses = [], [], [], []
    
    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('{') and '"mode": "eval"' in line:
                try:
                    data = json.loads(line)
                    if 'epoch' in data and 'mae_e_per_atom' in data:
                        epochs.append(data['epoch'])
                        mae_e_per_atom.append(data['mae_e_per_atom'] * 1000)  # eV -> meV
                        mae_f.append(data['mae_f'] * 1000)  # eV/Å -> meV/Å
                        losses.append(data['loss'])
                except json.JSONDecodeError:
                    continue
    
    return np.array(epochs), np.array(mae_e_per_atom), np.array(mae_f), np.array(losses)

def get_test_results(log_file):
    """Extract test MAE from log file (meV/atom, meV/Å)."""
    if not log_file.exists():
        return None, None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Pattern: | test_Default |    0.1    |    3.3    |
    matches = re.findall(r'\| test_Default \|[\s]+([\d.]+)[\s]+\|[\s]+([\d.]+)[\s]+\|', content)
    if matches:
        return float(matches[-1][0]), float(matches[-1][1])
    return None, None


def get_precise_test_energy(train_file, n_atoms):
    """Get precise total energy MAE (meV) from train.txt JSON."""
    if not train_file.exists():
        return None
    
    # Read the last eval line from train.txt
    last_eval = None
    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('{') and '"mode": "eval"' in line:
                try:
                    last_eval = json.loads(line)
                except json.JSONDecodeError:
                    continue
    
    if last_eval and 'mae_e_per_atom' in last_eval:
        # Convert eV -> meV, per-atom -> total
        return last_eval['mae_e_per_atom'] * 1000 * n_atoms
    return None

def plot_molecule(molecule, channels=256):
    """Plot training curves for a molecule and save to run directory."""
    run_dir = find_latest_run(molecule, channels)
    if not run_dir:
        print(f"  [SKIP] No run found for {molecule}")
        return None
    
    train_files = list(run_dir.glob("*_train.txt"))
    if not train_files:
        print(f"  [SKIP] No training log for {molecule}")
        return None
    
    train_file = train_files[0]
    epochs, mae_e_per_atom, mae_f, losses = parse_training_log(train_file)
    
    if len(epochs) == 0:
        print(f"  [SKIP] No data in {molecule}")
        return None
    
    # Get test results
    log_name = train_file.stem.replace('_train', '')
    log_file = LOGS_DIR / f"{log_name}.log"
    test_energy_mae_per_atom, test_forces_mae = get_test_results(log_file)
    
    paper = PAPER_RESULTS.get(molecule, {'E': 1.0, 'F': 2.0})
    n_atoms = MOLECULE_ATOMS.get(molecule, 10)
    paper_e_per_atom = paper['E'] / n_atoms
    test_energy_mae_total = get_precise_test_energy(train_file, n_atoms)
    if test_energy_mae_total is None and test_energy_mae_per_atom is not None:
        test_energy_mae_total = test_energy_mae_per_atom * n_atoms
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'MACE Training on {molecule.capitalize()} (rMD17)', fontsize=14, fontweight='bold')
    
    ax1 = axes[0, 0]  # Loss
    ax1.semilogy(epochs, losses, 'b-', linewidth=1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=800, color='r', linestyle='--', alpha=0.5, label='Stage Two Start')
    ax1.legend()
    
    ax2 = axes[0, 1]  # Energy MAE
    ax2.plot(epochs, mae_e_per_atom, 'g-', linewidth=1, label='Validation')
    ax2.axhline(y=paper_e_per_atom, color='r', linestyle='--', alpha=0.7, 
                label=f'Paper target ({paper_e_per_atom:.3f} meV/atom)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Energy MAE (meV/atom)')
    ax2.set_title('Energy MAE per Atom (Validation)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3 = axes[1, 0]  # Forces MAE
    ax3.plot(epochs, mae_f, 'orange', linewidth=1, label='Validation')
    ax3.axhline(y=paper['F'], color='r', linestyle='--', alpha=0.7, 
                label=f'Paper target ({paper["F"]} meV/Å)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Forces MAE (meV/Å)')
    ax3.set_title('Forces MAE (Validation)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    ax4 = axes[1, 1]  # Results summary
    ax4.axis('off')
    test_e_per_atom_str = f"{test_energy_mae_per_atom:.1f}" if test_energy_mae_per_atom is not None else "N/A"
    test_e_total_str = f"{test_energy_mae_total:.1f}" if test_energy_mae_total is not None else "N/A"
    test_f_str = f"{test_forces_mae:.1f}" if test_forces_mae is not None else "N/A"
    
    results_text = f"""
Final Results (Epoch {epochs[-1]:.0f})
{'='*45}

VALIDATION:
  Energy MAE (per atom):  {mae_e_per_atom[-1]:.4f} meV
  Forces MAE:             {mae_f[-1]:.2f} meV/Å

TEST (Stage Two):
  Energy MAE (per atom):  {test_e_per_atom_str} meV
  Energy MAE (total):     {test_e_total_str} meV
  Forces MAE:             {test_f_str} meV/Å

Paper Target ({molecule.capitalize()}):
  Energy MAE (total):     {paper['E']} meV
  Forces MAE:             {paper['F']} meV/Å

{'='*45}
Model: MACE ({channels}x0e+{channels}x1o+{channels}x2e+{channels}x3o)
       2 layers, correlation=3
Dataset: rMD17 {molecule.capitalize()} ({n_atoms} atoms)
"""
    
    ax4.text(0.05, 0.5, results_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_path = run_dir / "training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if test_energy_mae_total is not None and test_forces_mae is not None:
        print(f"  [OK] {molecule}: E={test_energy_mae_total:.1f} meV (Paper: {paper['E']}), F={test_forces_mae:.1f} meV/Å (Paper: {paper['F']})")
    else:
        print(f"  [OK] {molecule}: (test results not found)")
    
    return {
        'molecule': molecule,
        'n_atoms': n_atoms,
        'test_energy_mae_total': test_energy_mae_total,
        'test_forces_mae': test_forces_mae,
        'paper_energy_mae': paper['E'],
        'paper_forces_mae': paper['F'],
        'output': output_path,
    }

def main():
    print("="*70)
    print("Plotting training curves for all rMD17 molecules")
    print("="*70)
    
    results = []
    for mol in MOLECULES:
        result = plot_molecule(mol)
        if result:
            results.append(result)
    
    print("\n" + "="*70)
    print("Summary: Our Results vs Paper (Batatia et al., NeurIPS 2022)")
    print("="*70)
    print(f"{'Molecule':<15} {'Ours E':<10} {'Paper E':<10} {'Ours F':<10} {'Paper F':<10}")
    print(f"{'':15} {'(meV)':<10} {'(meV)':<10} {'(meV/Å)':<10} {'(meV/Å)':<10}")
    print("-"*70)
    for r in results:
        e_ours = f"{r['test_energy_mae_total']:.1f}" if r['test_energy_mae_total'] is not None else "N/A"
        f_ours = f"{r['test_forces_mae']:.1f}" if r['test_forces_mae'] is not None else "N/A"
        print(f"{r['molecule']:<15} {e_ours:<10} {r['paper_energy_mae']:<10.1f} {f_ours:<10} {r['paper_forces_mae']:<10.1f}")
    print("="*70)
    print(f"\nGenerated {len(results)} plots in results/*/run_ch256_*/training_curves.png")

if __name__ == "__main__":
    main()
