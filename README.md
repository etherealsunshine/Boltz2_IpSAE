# Boltz2_IpSAE

Utilities for running ipSAE on Boltz co-folding predictions, including a
high-level pipeline script that takes binders and a target directly from the
command line (no manual YAML editing).

---

## 1. Overview

This folder contains:

- `ipsae.py` – the ipSAE calculator for AF2/AF3/Boltz structures.
- `visualise_binder_validation.py` – runs ipSAE on sets of Boltz predictions,
  aggregates results and makes plots/heatmaps.
- `make_binder_validation_scripts.py` – legacy helper that reads a config YAML
  and generates Boltz YAMLs and run scripts.
- `run_ipsae_pipeline.py` – **recommended** CLI wrapper that:
  - takes binders (CSV/FASTA/sequence) and a target (± antitarget, self),
  - runs Boltz predictions,
  - runs ipSAE per binder,
  - streams a compact summary CSV and prints per-binder ipSAE numbers,
  - produces the same global ipSAE heatmaps and `ipsae_summary_all_binders.csv`
    as the legacy flow.

The examples under `example_yaml/` provide a complete Nipah G use case
including known binders.

---

## 2. Prerequisites

You can install and use these scripts anywhere, as long as the `boltz` Python
package is installed and importable in your environment.

### 2.1. Install Boltz in your environment

Create and activate a Python virtualenv (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Boltz from PyPI (GPU: `[cuda]`, CPU-only: drop `[cuda]`):

```bash
python -m pip install --upgrade pip
python -m pip install "boltz[cuda]"
```

or from the main Boltz repo if you prefer:

```bash
git clone https://github.com/jwohlwend/boltz.git
cd boltz
python -m pip install -e .[cuda]
```

Install plotting libraries used by the ipSAE helpers:

```bash
python -m pip install seaborn matplotlib
```

Verify that the `boltz` CLI is available:

```bash
boltz --help
```

All commands below assume your virtualenv is activated and `boltz` is
installed in that environment.

### 2.2. Install Boltz2_IpSAE scripts

Clone this repository:

```bash
git clone https://github.com/profdocpizza/Boltz2_IpSAE.git
cd Boltz2_IpSAE
```

From here you can run the helper scripts directly, e.g.:

```bash
python run_ipsae_pipeline.py --help
```

---

## 3. `run_ipsae_pipeline.py` – binder-by-binder pipeline

Run from the cloned `Boltz2_IpSAE` directory (or pass absolute/relative paths
to it); it does not depend on a specific folder layout beyond the paths you
provide.

### 3.1. Inputs

**Binders** (choose exactly one source):

- CSV:
  - `--binder_csv my_binders.csv`
  - `--binder_name_col binder_name` (default: `name`)
  - `--binder_seq_col binder_sequence` (default: `sequence`)
  - Sequences can be multi-chain as `CHAINA:CHAINB`.

- FASTA directory:
  - `--binder_fasta_dir path/to/binder_fastas/`
  - One FASTA file per binder; binder name from filename.

- Single binder from FASTA:
  - `--binder_fasta path/to/binder.fasta`
  - `--binder_name Binder1`

- Single binder from CLI sequence:
  - `--binder_seq "AA...AA:BB...BB"`
  - `--binder_name Binder1`

Optional binder tweak:

- `--add_n_terminal_lysine` – prepend `K` at N-terminus of each chain if missing.

**Target (required, single)**:

- `--target_name nipah_g`
- Either:
  - `--target_seq "QNYTRS..."` (chains separated by `:`), or
  - `--target_fasta Boltz2_IpSAE/example_yaml/nipah_g.fasta`
- Optional:
  - `--target_msa Boltz2_IpSAE/example_yaml/nipah.a3m` (applied to chain 0).

**Antitarget (optional, single)**:

- `--antitarget_name Sialidase_2F29`
- Either:
  - `--antitarget_seq "GSMASL..."`, or
  - `--antitarget_fasta Boltz2_IpSAE/example_yaml/sialidase_2F29.fasta`
- Optional:
  - `--antitarget_msa path/to/msa.a3m`

**Self-binding control (optional)**:

- `--include_self` – also run each binder against itself.

**Boltz options**:

- `--out_dir ./boltz_ipsae` – root output directory.
- `--recycling_steps 10`
- `--diffusion_samples 5`
- `--use_msa_server {auto,true,false}` (default `auto`).

**ipSAE options**:

- `--ipsae_pae_cutoff 15` (Å)
- `--ipsae_dist_cutoff 15` (Å)
- `--use_best_model` (affects global heatmap aggregation only)
- `--num_cpu 4` (used for the final global stage in `visualise_binder_validation`).

Logging:

- Default: clean stage/progress messages, logs written under `out_dir/logs/`.
- `--verbose`: stream full Boltz/ipSAE output to the terminal as well.

---

### 3.2. Nipah G example with bundled known binders

The repo includes:

- Nipah G FASTA: `Boltz2_IpSAE/example_yaml/nipah_g.fasta`
- Nipah G MSA: `Boltz2_IpSAE/example_yaml/nipah.a3m`
- Known binders: `Boltz2_IpSAE/example_yaml/known_binders/`
- Sialidase off-target FASTA: `Boltz2_IpSAE/example_yaml/sialidase_2F29.fasta`

Run the pipeline from the repo root:

```bash
python Boltz2_IpSAE/run_ipsae_pipeline.py \
  --binder_fasta_dir Boltz2_IpSAE/example_yaml/known_binders \
  --target_name nipah_g \
  --target_fasta Boltz2_IpSAE/example_yaml/nipah_g.fasta \
  --target_msa Boltz2_IpSAE/example_yaml/nipah.a3m \
  --antitarget_name Sialidase_2F29 \
  --antitarget_fasta Boltz2_IpSAE/example_yaml/sialidase_2F29.fasta \
  --include_self \
  --out_dir Boltz2_IpSAE/example_yaml/boltz_ipsae_nipah \
  --recycling_steps 10 \
  --diffusion_samples 5 \
  --use_msa_server auto \
  --ipsae_pae_cutoff 15 \
  --ipsae_dist_cutoff 15 \
  --num_cpu 4
```

For each binder the script:

1. Runs Boltz for binder vs Nipah G / antitarget / self.
2. Runs ipSAE on those predictions.
3. Prints binder‑level ipSAE summaries (target and antitarget).
4. Appends one row to:
   - `Boltz2_IpSAE/example_yaml/boltz_ipsae_nipah/summary/binder_pair_summary.csv`

After all binders are processed, it also writes:

- Per-binder CSVs/plots:
  - `boltz_ipsae_nipah/binder_*/plots/ipsae_summary.csv`
- Global data/heatmaps:
  - `boltz_ipsae_nipah/summary/ipsae_summary_all_binders.csv`
  - `boltz_ipsae_nipah/summary/ipSAE_min_heatmap.csv`
  - corresponding PNG/SVG heatmaps.

---

## 4. Legacy helpers

The older `make_binder_validation_scripts.py` + `run_all_cofolding.sh` +
`visualise_binder_validation.py` workflow is still present for compatibility,
but `run_ipsae_pipeline.py` should usually be more convenient: it exposes the
same functionality from a single CLI entry point and streams binder‑level
summaries into a small CSV as the script progresses.
