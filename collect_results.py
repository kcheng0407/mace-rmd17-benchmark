"""Collect final results to summary/ folder for GitHub (~4MB vs ~4GB full)."""

import json
import shutil
from pathlib import Path

BASE_DIR = Path("/home/kai/Desktop/mace-rmd17-benchmark")
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
SUMMARY_DIR = BASE_DIR / "summary"

MOLECULES = [
    "aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
    "naphthalene", "paracetamol", "salicylic", "toluene", "uracil",
]

# Paper results from Table 2 (Energy MAE in meV, Forces MAE in meV/Å)
PAPER_RESULTS = {
    'aspirin':       {'E': 2.2, 'F': 6.6, 'atoms': 21},
    'azobenzene':    {'E': 1.2, 'F': 2.6, 'atoms': 24},
    'benzene':       {'E': 0.4, 'F': 0.3, 'atoms': 12},
    'ethanol':       {'E': 0.4, 'F': 2.1, 'atoms': 9},
    'malonaldehyde': {'E': 0.8, 'F': 5.0, 'atoms': 9},
    'naphthalene':   {'E': 0.5, 'F': 1.3, 'atoms': 18},
    'paracetamol':   {'E': 1.5, 'F': 4.1, 'atoms': 20},
    'salicylic':     {'E': 0.9, 'F': 3.8, 'atoms': 16},
    'toluene':       {'E': 0.5, 'F': 1.5, 'atoms': 15},
    'uracil':        {'E': 0.5, 'F': 2.8, 'atoms': 12},
}


def find_latest_ch256_run(molecule):
    """Find latest ch256 run directory."""
    mol_dir = RESULTS_DIR / molecule
    runs = sorted(mol_dir.glob("run_ch256_*"))
    return runs[-1] if runs else None


def extract_final_results(train_txt_path):
    """Extract final eval metrics from _train.txt."""
    results = {}
    with open(train_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '"mode": "eval"' in line:
                try:
                    data = json.loads(line)
                    results = data  # keep overwriting, last one = final
                except json.JSONDecodeError:
                    continue
    return results


def main():
    # Create clean summary directory
    if SUMMARY_DIR.exists():
        shutil.rmtree(SUMMARY_DIR)
    
    logs_out = SUMMARY_DIR / "logs"
    plots_out = SUMMARY_DIR / "plots"
    logs_out.mkdir(parents=True, exist_ok=True)
    plots_out.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for mol in MOLECULES:
        run_dir = find_latest_ch256_run(mol)
        if run_dir is None:
            print(f"⚠ {mol}: no ch256 run found, skipping")
            continue

        print(f"Processing {mol}... ({run_dir.name})")

        # Copy training curve plot
        png = run_dir / "training_curves.png"
        if png.exists():
            shutil.copy2(png, plots_out / f"{mol}_training_curves.png")

        # Copy readable .log file
        for log_file in LOGS_DIR.glob(f"*{run_dir.name.replace('run_ch256_', 'ch256_')}*"):
            if '_debug' not in log_file.name:
                shutil.copy2(log_file, logs_out / f"{mol}.log")

        # Extract final metrics
        train_txts = list(run_dir.glob("*_train.txt"))
        if train_txts:
            metrics = extract_final_results(train_txts[0])
            if metrics:
                all_results[mol] = {
                    'mae_e_meV': metrics['mae_e'] * 1000,
                    'mae_f_meV': metrics['mae_f'] * 1000,
                    'rmse_e_meV': metrics['rmse_e'] * 1000,
                    'rmse_f_meV': metrics['rmse_f'] * 1000,
                    'run_dir': run_dir.name,
                }

    # Generate results_summary.md
    md_lines = [
        "# MACE rMD17 Benchmark Results",
        "",
        "**Model:** MACE (256x0e+256x1o+256x2e+256x3o, L=3, 256 channels)",
        "**Settings:** 2 layers, correlation=3, r_max=5.0 Å, batch_size=5, 1000 epochs",
        "**Split:** index 1 (1000 train → 950 train + 50 valid, 1000 test)",
        "**Date:** 2026-02-26 ~ 2026-02-27",
        "",
        "## Results Table",
        "",
        "| Molecule | Atoms | Energy MAE (meV) | Paper E (meV) | Forces MAE (meV/Å) | Paper F (meV/Å) | F Ratio |",
        "|----------|------:|-----------------:|--------------:|-------------------:|----------------:|--------:|",
    ]

    total_ratio = 0
    count = 0
    for mol in MOLECULES:
        if mol not in all_results:
            continue
        r = all_results[mol]
        p = PAPER_RESULTS[mol]
        ratio = r['mae_f_meV'] / p['F']
        total_ratio += ratio
        count += 1
        md_lines.append(
            f"| {mol} | {p['atoms']} | {r['mae_e_meV']:.2f} | {p['E']} | "
            f"{r['mae_f_meV']:.2f} | {p['F']} | {ratio:.2f}x |"
        )

    avg_ratio = total_ratio / count if count else 0
    md_lines.append(f"| **Average** | | | | | | **{avg_ratio:.2f}x** |")
    
    md_lines.extend([
        "",
        "## RMSE Table",
        "",
        "| Molecule | Energy RMSE (meV) | Forces RMSE (meV/Å) |",
        "|----------|------------------:|--------------------:|",
    ])
    
    for mol in MOLECULES:
        if mol not in all_results:
            continue
        r = all_results[mol]
        md_lines.append(f"| {mol} | {r['rmse_e_meV']:.2f} | {r['rmse_f_meV']:.2f} |")

    md_lines.extend([
        "",
        "## Training Curves",
        "",
    ])
    for mol in MOLECULES:
        if mol in all_results:
            md_lines.append(f"### {mol.capitalize()}")
            md_lines.append(f"![{mol}](plots/{mol}_training_curves.png)")
            md_lines.append("")

    md_lines.extend([
        "## Hyperparameters (following MACE paper Appendix A.5.1)",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| hidden_irreps | 256x0e+256x1o+256x2e+256x3o |",
        "| num_interactions | 2 |",
        "| correlation | 3 |",
        "| max_ell | 3 |",
        "| r_max | 5.0 Å |",
        "| radial_MLP | [64, 64, 64] |",
        "| num_radial_basis | 8 |",
        "| num_cutoff_basis | 5 |",
        "| batch_size | 5 |",
        "| max_num_epochs | 1000 |",
        "| lr | 0.01 |",
        "| scheduler | ReduceLROnPlateau (patience=50, factor=0.8) |",
        "| optimizer | AMSGrad Adam (β₁=0.9, β₂=0.999, ε=1e-8) |",
        "| weight_decay | 5e-7 |",
        "| energy_weight | 1.0 (Stage Two: 1000.0) |",
        "| forces_weight | 1000.0 (Stage Two: 1000.0) |",
        "| EMA decay | 0.99 |",
        "| Stage Two start | epoch 800 |",
        "| activation | SiLU |",
        "| readout MLP | 16x0e |",
        "| default_dtype | float32 |",
        "| seed | 42 |",
        "",
    ])

    summary_md = SUMMARY_DIR / "results_summary.md"
    summary_md.write_text("\n".join(md_lines))

    print(f"\n{'='*60}")
    print(f"Summary collected to: {SUMMARY_DIR}")
    print(f"{'='*60}")
    print(f"  logs/   : {len(list(logs_out.iterdir()))} files")
    print(f"  plots/  : {len(list(plots_out.iterdir()))} files")
    print(f"  results_summary.md")
    
    import subprocess
    result = subprocess.run(['du', '-sh', str(SUMMARY_DIR)], capture_output=True, text=True)
    print(f"  Total size: {result.stdout.strip().split()[0]}")


if __name__ == "__main__":
    main()
