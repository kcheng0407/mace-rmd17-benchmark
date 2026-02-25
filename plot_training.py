"""
Plot MACE training curves from log file
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Log file path
log_file = Path("/home/kai/Desktop/Onboarding/results/ethanol/run_20260225_104639/MACE_ethanol_20260225_104639_run-42_train.txt")

# Parse log file
epochs = []
mae_e_per_atom = []
mae_f = []
losses = []

with open(log_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('{') and '"mode": "eval"' in line:
            try:
                data = json.loads(line)
                if 'epoch' in data and 'mae_e_per_atom' in data:
                    epochs.append(data['epoch'])
                    # Convert to meV
                    mae_e_per_atom.append(data['mae_e_per_atom'] * 1000)  # eV to meV
                    mae_f.append(data['mae_f'] * 1000)  # eV/Å to meV/Å
                    losses.append(data['loss'])
            except json.JSONDecodeError:
                continue

# Convert to numpy arrays
epochs = np.array(epochs)
mae_e_per_atom = np.array(mae_e_per_atom)
mae_f = np.array(mae_f)
losses = np.array(losses)

print(f"Parsed {len(epochs)} epochs")
print(f"Final Energy MAE (per atom): {mae_e_per_atom[-1]:.4f} meV")
print(f"Final Forces MAE: {mae_f[-1]:.4f} meV/Å")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('MACE Training on Ethanol (rMD17)', fontsize=14, fontweight='bold')

# 1. Loss curve
ax1 = axes[0, 0]
ax1.semilogy(epochs, losses, 'b-', linewidth=1)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=800, color='r', linestyle='--', alpha=0.5, label='Stage Two Start')
ax1.legend()

# 2. Energy MAE
ax2 = axes[0, 1]
ax2.plot(epochs, mae_e_per_atom, 'g-', linewidth=1)
ax2.axhline(y=0.4/9, color='r', linestyle='--', alpha=0.7, label=f'Paper target (0.044 meV/atom)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Energy MAE (meV/atom)')
ax2.set_title('Energy MAE per Atom')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Forces MAE
ax3 = axes[1, 0]
ax3.plot(epochs, mae_f, 'orange', linewidth=1)
ax3.axhline(y=2.1, color='r', linestyle='--', alpha=0.7, label='Paper target (2.1 meV/Å)')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Forces MAE (meV/Å)')
ax3.set_title('Forces MAE')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. Final Results Summary
ax4 = axes[1, 1]
ax4.axis('off')

# Results table
results_text = f"""
Final Results (Epoch {epochs[-1]:.0f})
{'='*40}

Energy MAE (per atom):  {mae_e_per_atom[-1]:.4f} meV
Energy MAE (total):     {mae_e_per_atom[-1]*9:.4f} meV
Paper target:           0.4 meV

Forces MAE:             {mae_f[-1]:.2f} meV/Å
Paper target:           2.1 meV/Å

{'='*40}
Model: MACE (128x0e+128x1o+128x2e, 2 layers, correlation=3)
Dataset: rMD17 Ethanol (950 train, 50 valid)
"""

ax4.text(0.1, 0.5, results_text, transform=ax4.transAxes, fontsize=12,
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save figure
output_path = Path("/home/kai/Desktop/Onboarding/results/ethanol/run_20260225_104639/training_curves.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

# Also show
plt.show()
