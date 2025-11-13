#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize Boltz/ipSAE validation outputs.

Key assumptions / behaviour:

- Boltz configs use chain IDs:
    * Binder chains: A, B, C, ...
    * Target chains: TA, TB, TC, ...
    * Antitarget chains: AA, AB, AC, ...
- Binder is always the A-chain (chain_of_focus = "A").
- Boltz outputs live under:
      binder_<binder_name>/outputs/boltz_results_<yaml_stem>/
- YAML stems look like:
      binder_<binder>_vs_target_<target>
      binder_<binder>_vs_antitarget_<name>

This script:
  * runs ipSAE on all models for each binder–(anti)target pair
  * extracts metrics for chain A vs its best partner
  * stores:
        - binder
        - vs (full name)
        - partner (e.g. Spike, HA, ...)
        - target_type (target / antitarget / unknown)
        - model_idx
        - numeric ipSAE metrics (_min, _max)
  * makes per-binder stripplots (all targets & antitargets together)
  * makes global heatmaps (ipSAE_min, ipSAE_max) averaged across models
"""

import argparse
import subprocess
import re
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def run_ipsae(
    pae_file,
    cif_file,
    pae_cutoff=10,
    dist_cutoff=15,
    chain_of_focus="A"
):
    """
    Run ipsae.py and compute min/max metrics for chain_of_focus.

    If multiple partner chains exist, choose the one with the highest ipSAE_max
    (using only ASYM rows). Then compute metric_min / metric_max from only
    those rows.
    """

    import pandas as pd
    import numpy as np
    import os
    import subprocess

    # ---------------------------
    # Run ipSAE
    # ---------------------------
    cmd = [
        "python", "ipsae.py",
        str(pae_file),
        str(cif_file),
        str(pae_cutoff),
        str(dist_cutoff)
    ]
    subprocess.run(cmd, check=True)

    out_txt = str(cif_file).replace(".cif", f"_{pae_cutoff}_{dist_cutoff}.txt")
    if not os.path.exists(out_txt):
        raise FileNotFoundError(f"Missing ipSAE output: {out_txt}")

    # ---------------------------
    # Load table
    # ---------------------------
    try:
        df = pd.read_csv(out_txt, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(out_txt, delim_whitespace=True)

    # Convert numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Required columns
    chn1_col = "Chn1"
    chn2_col = "Chn2"
    type_col = "Type"

    # ---------------------------
    # Keep only ASYM rows
    # ---------------------------
    df = df[df[type_col].str.lower() == "asym"]

    # ---------------------------
    # Keep only rows where focus chain appears
    # ---------------------------
    df = df[(df[chn1_col] == chain_of_focus) | (df[chn2_col] == chain_of_focus)]
    if df.empty:
        raise ValueError(f"No ASYM rows involving chain {chain_of_focus}")

    # ---------------------------
    # Identify partner chains
    # ---------------------------
    partners = set()
    for _, row in df.iterrows():
        partner = row[chn2_col] if row[chn1_col] == chain_of_focus else row[chn1_col]
        partners.add(partner)
    partners = sorted(partners)

    # ---------------------------
    # If multiple partners → choose highest ipSAE_max partner
    # ---------------------------
    partner_best = None
    partner_best_score = -np.inf
    best_df = None

    for p in partners:
        sub = df[
            ((df[chn1_col] == chain_of_focus) & (df[chn2_col] == p)) |
            ((df[chn2_col] == chain_of_focus) & (df[chn1_col] == p))
        ]
        if sub.empty:
            continue

        ipSAE_max = sub["ipSAE"].max()
        if ipSAE_max > partner_best_score:
            partner_best_score = ipSAE_max
            partner_best = p
            best_df = sub.copy()

    if partner_best is None or best_df is None:
        raise ValueError(f"No valid partner rows found for {chain_of_focus}")

    # ---------------------------
    # Compute min/max metrics from best partner rows only
    # ---------------------------
    numeric_cols = best_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() != "model"]

    output = {
        "chain_of_focus": chain_of_focus,
        "involved_chains": partner_best
    }

    for col in numeric_cols:
        output[f"{col}_min"] = best_df[col].min()
        output[f"{col}_max"] = best_df[col].max()

    return output


def parse_vs_name(vs_name: str):
    """
    Parse vs_name of the form:
        binder_<binder>_vs_target_<partner>
        binder_<binder>_vs_antitarget_<partner>
    Returns (partner_name, target_type).
    """
    m = re.search(r"_vs_(target|antitarget)_(.*)$", vs_name)
    if m:
        target_type = m.group(1)  # 'target' or 'antitarget'
        partner_name = m.group(2)
    else:
        target_type = "unknown"
        m2 = re.search(r"_vs_(.*)$", vs_name)
        partner_name = m2.group(1) if m2 else vs_name
    return partner_name, target_type


def analyse_binder(binder_dir: Path):
    """
    Analyse a binder directory: compute ipSAE for all vs_* pairs, save plots.
    """
    plots_dir = binder_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    binder_records = []

    for vs_dir in (binder_dir / "outputs").glob("boltz_results_*vs*"):
        vs_name = vs_dir.name.replace("boltz_results_", "")
        pred_root = vs_dir / "predictions" / vs_name

        partner_name, target_type = parse_vs_name(vs_name)

        for pae_file in pred_root.glob("pae_*_model_*.npz"):
            m = re.search(r"model_(\d+)", pae_file.name)
            if not m:
                continue
            model_idx = m.group(1)
            cif_file = pred_root / pae_file.name.replace("pae_", "").replace(".npz", ".cif")
            if not cif_file.exists():
                continue

            try:
                rec = run_ipsae(pae_file, cif_file, chain_of_focus="A")
            except Exception as e:
                print(f"⚠️ ipSAE failed for {pae_file} ({e}). Skipping.")
                continue

            rec.update({
                "binder": binder_dir.name,
                "vs": vs_name,
                "model_idx": int(model_idx),
                "partner": partner_name,
                "target_type": target_type,
            })
            binder_records.append(rec)

    if not binder_records:
        print(f"No valid ipSAE data for {binder_dir.name}")
        return

    df = pd.DataFrame(binder_records)

    csv_path = plots_dir / "ipsae_summary_chainA.csv"
    df.to_csv(csv_path, index=False)

    metrics = ["ipSAE_min", "ipSAE_max"]
    partner_order = sorted(df["partner"].dropna().unique().tolist())

    for metric in metrics:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(6, 3.5))
        sns.stripplot(
            data=df,
            x="partner",
            y=metric,
            hue="model_idx",
            alpha=0.7,
            order=partner_order,
        )
        short_title = re.sub(r"^binder_", "", binder_dir.name)
        plt.title(f"{metric} for {short_title}")
        plt.ylabel(metric)
        plt.xlabel("Target / Antitarget")
        plt.xticks(rotation=30)
        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            order_idx = sorted(range(len(labels)), key=lambda i: int(labels[i]))
            plt.legend(
                [handles[i] for i in order_idx],
                [labels[i] for i in order_idx],
                title="model_idx",
                loc="best",
            )
        plt.tight_layout()
        for ext in ["png"]:
            plt.savefig(plots_dir / f"{metric}_stripplot_chainA.{ext}", dpi=200)
        plt.close()

    print(f"Saved: {csv_path}")


def plot_overall(root_dir: Path):
    """
    Combine all per-binder CSVs and plot heatmaps for ipSAE_min and ipSAE_max
    (averages across models).

    All targets & antitargets are pooled together; target_type is kept in the
    DataFrame but not used to split the plots (for now).
    """
    csvs = list(root_dir.glob("binder_*/plots/ipsae_summary_chainA.csv"))
    if not csvs:
        print("No binder CSVs found.")
        return

    dfs = []
    for csv in csvs:
        df = pd.read_csv(csv)
        # Backwards compatibility: older CSVs may not have 'partner' or 'target_type'
        if "partner" not in df.columns and "vs" in df.columns:
            df["partner"] = df["vs"].str.extract(r"_vs_(.*)$")
        if "target_type" not in df.columns and "vs" in df.columns:
            df["target_type"] = "unknown"

        df["binder_short"] = df["binder"].str.replace(r"^binder_", "", regex=True)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    metrics = [m for m in ["ipSAE_min", "ipSAE_max"] if m in all_df.columns]
    if not metrics:
        print("No ipSAE_min/ipSAE_max metrics found for heatmap plotting.")
        return

    agg = all_df.groupby(["binder_short", "partner"])[metrics].mean().reset_index()

    partners = sorted(agg["partner"].unique().tolist())
    binders = sorted(agg["binder_short"].unique().tolist())

    for metric in metrics:
        pivot = agg.pivot(index="partner", columns="binder_short", values=metric)
        pivot = pivot.reindex(index=partners, columns=binders)

        plt.figure(figsize=(max(7, len(binders) * 0.7), max(5, len(partners) * 0.4)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={"label": metric},
            linewidths=0.5,
            vmin=0,
            vmax=1,
        )
        plt.title(metric)
        plt.ylabel("Target / Antitarget", rotation=90)
        plt.xlabel("Binder")
        plt.yticks(rotation=0)
        plt.tight_layout()

        for ext in ["png", "svg"]:
            plt.savefig(root_dir / f"{metric}_heatmap_chainA.{ext}", dpi=300)

        plt.close()
        print(f"Saved heatmap for {metric}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--generate_data", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    root = Path(args.root_dir)
    if args.generate_data:
        for binder_dir in sorted(root.glob("binder_*")):
            if binder_dir.is_dir():
                analyse_binder(binder_dir)
    if args.plot:
        plot_overall(root)


if __name__ == "__main__":
    main()
