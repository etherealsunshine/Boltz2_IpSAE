#!/usr/bin/env python3
"""
Run Boltz + ipSAE pipeline binder‑by‑binder without manual YAML.

For each binder the script:
  1) Builds small Boltz YAMLs for binder vs target / antitarget / self.
  2) Runs `boltz predict` for that binder only.
  3) Runs ipSAE on this binder’s predictions.
  4) Appends a compact summary row (ipSAE mean/std for target and antitarget)
     to a global CSV and prints the numbers.

After all binders are processed, it also generates the legacy global
heatmaps and `ipsae_summary_all_binders.csv` via visualise_binder_validation.

By default, the script prints high‑level progress and hides Boltz/ipSAE logs;
use `--verbose` to surface full subprocess output.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from make_binder_validation_scripts import (
    add_n_terminal_lysine,
    read_fasta_dir_entities,
    read_fasta_multi,
    sanitize_name,
    yaml_for_pair,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run Boltz + ipSAE pipeline for one target and many binders "
            "without manually writing YAML files. Processes binders one‑by‑one."
        )
    )

    # Binder sources (exactly one)
    binder_group = ap.add_mutually_exclusive_group(required=True)
    binder_group.add_argument(
        "--binder_csv",
        type=str,
        help="CSV file with binders (binder name + sequence columns).",
    )
    binder_group.add_argument(
        "--binder_fasta_dir",
        type=str,
        help="Directory containing per-binder FASTA files.",
    )
    binder_group.add_argument(
        "--binder_fasta",
        type=str,
        help="Single binder FASTA file (use with --binder_name).",
    )
    binder_group.add_argument(
        "--binder_seq",
        type=str,
        help="Single binder amino acid sequence (use with --binder_name). "
        "For multi-chain binders, separate chains with ':'.",
    )

    ap.add_argument(
        "--binder_name",
        type=str,
        help="Binder name when using --binder_fasta or --binder_seq.",
    )
    ap.add_argument(
        "--binder_name_col",
        type=str,
        default="name",
        help="Column name for binder names in --binder_csv (default: name).",
    )
    ap.add_argument(
        "--binder_seq_col",
        type=str,
        default="sequence",
        help="Column name for binder sequences in --binder_csv (default: sequence).",
    )
    ap.add_argument(
        "--add_n_terminal_lysine",
        action="store_true",
        help="Prepend 'K' to each binder chain if missing.",
    )

    # Target (required, single)
    ap.add_argument(
        "--target_name",
        required=True,
        type=str,
        help="Name of the target protein (e.g. nipah_g).",
    )
    ap.add_argument(
        "--target_seq",
        type=str,
        help="Target amino acid sequence. For multi-chain targets, separate chains "
        "with ':'. Mutually exclusive with --target_fasta.",
    )
    ap.add_argument(
        "--target_fasta",
        type=str,
        help="FASTA file with target sequence(s). Mutually exclusive with --target_seq.",
    )
    ap.add_argument(
        "--target_msa",
        type=str,
        help="Optional MSA file (e.g. .a3m) for target chain 0.",
    )

    # Antitarget (optional, single)
    ap.add_argument(
        "--antitarget_name",
        type=str,
        help="Name of an off-target protein to penalize binding (optional).",
    )
    ap.add_argument(
        "--antitarget_seq",
        type=str,
        help="Antitarget amino acid sequence. Multi-chain: chains separated by ':'.",
    )
    ap.add_argument(
        "--antitarget_fasta",
        type=str,
        help="FASTA file with antitarget sequence(s).",
    )
    ap.add_argument(
        "--antitarget_msa",
        type=str,
        help="Optional MSA file for antitarget chain 0.",
    )

    ap.add_argument(
        "--include_self",
        action="store_true",
        help="Also run each binder against itself (self-binding control).",
    )

    # Boltz options
    ap.add_argument(
        "--out_dir",
        type=str,
        default="./boltz_ipsae",
        help="Root output directory for all results (default: ./boltz_ipsae).",
    )
    ap.add_argument(
        "--recycling_steps",
        type=int,
        default=10,
        help="Boltz recycling steps (default: 10).",
    )
    ap.add_argument(
        "--diffusion_samples",
        type=int,
        default=5,
        help="Boltz diffusion samples (default: 5).",
    )
    ap.add_argument(
        "--use_msa_server",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help=(
            "Whether to use the MSA server for Boltz predictions. "
            "'auto' (default) uses the server only when neither binder "
            "nor partner has an explicit MSA."
        ),
    )

    # ipSAE options
    ap.add_argument(
        "--ipsae_pae_cutoff",
        type=int,
        default=15,
        help="ipSAE PAE cutoff in Å (default: 15).",
    )
    ap.add_argument(
        "--ipsae_dist_cutoff",
        type=int,
        default=15,
        help="ipSAE distance cutoff in Å (default: 15).",
    )
    ap.add_argument(
        "--use_best_model",
        action="store_true",
        help=(
            "For global heatmaps, use only the best model (highest ipSAE_max) "
            "per binder/partner instead of averaging across models."
        ),
    )
    ap.add_argument(
        "--num_cpu",
        type=int,
        default=1,
        help="Number of CPUs to use for ipSAE scoring (default: 1, used for global stage).",
    )

    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Show full Boltz and ipSAE subprocess output.",
    )

    return ap.parse_args()


def _clean_seq_string(seq: str) -> str:
    """Normalize a sequence string: remove whitespace, uppercase."""
    return "".join(seq.split()).upper()


def _split_multi_chain(seq: str) -> List[str]:
    """
    Split a multi-chain sequence string into chains, using ':' as separator.
    """
    seq = seq.strip()
    if not seq:
        return []
    parts = [p for p in seq.split(":") if p.strip()]
    return [_clean_seq_string(p) for p in parts]


def load_binders_from_csv(
    csv_path: Path,
    name_col: str,
    seq_col: str,
    addK: bool,
) -> List[Dict[str, object]]:
    binders: List[Dict[str, object]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")
        if name_col not in reader.fieldnames:
            raise ValueError(
                f"CSV file {csv_path} missing binder name column {name_col!r}."
            )
        if seq_col not in reader.fieldnames:
            raise ValueError(
                f"CSV file {csv_path} missing binder sequence column {seq_col!r}."
            )
        for row in reader:
            raw_name = (row.get(name_col) or "").strip()
            raw_seq = (row.get(seq_col) or "").strip()
            if not raw_name or not raw_seq:
                continue
            name = sanitize_name(raw_name)
            chains = _split_multi_chain(raw_seq)
            if not chains:
                continue
            if addK:
                chains = add_n_terminal_lysine(chains)
            msas = [None] * len(chains)
            binders.append({"name": name, "seqs": chains, "msas": msas})
    if not binders:
        raise ValueError(f"No valid binders found in CSV: {csv_path}")
    return binders


def load_binders_from_dir(
    fasta_dir: Path,
    addK: bool,
) -> List[Dict[str, object]]:
    if not fasta_dir.is_dir():
        raise ValueError(f"Binder FASTA directory not found: {fasta_dir}")
    entities = read_fasta_dir_entities(fasta_dir)
    binders: List[Dict[str, object]] = []
    for name, seqs in entities:
        if not seqs:
            continue
        chains = [_clean_seq_string(s) for s in seqs]
        if addK:
            chains = add_n_terminal_lysine(chains)
        msas = [None] * len(chains)
        binders.append({"name": name, "seqs": chains, "msas": msas})
    if not binders:
        raise ValueError(f"No valid binder sequences found in directory: {fasta_dir}")
    return binders


def load_single_binder_from_fasta(
    name: str,
    fasta_path: Path,
    addK: bool,
) -> List[Dict[str, object]]:
    if not fasta_path.is_file():
        raise ValueError(f"Binder FASTA not found: {fasta_path}")
    seqs = read_fasta_multi(fasta_path)
    if not seqs:
        raise ValueError(f"No sequences found in binder FASTA: {fasta_path}")
    chains = [_clean_seq_string(s) for s in seqs]
    if addK:
        chains = add_n_terminal_lysine(chains)
    msas = [None] * len(chains)
    return [{"name": sanitize_name(name), "seqs": chains, "msas": msas}]


def load_single_binder_from_seq(
    name: str,
    seq: str,
    addK: bool,
) -> List[Dict[str, object]]:
    chains = _split_multi_chain(seq)
    if not chains:
        raise ValueError("Binder sequence is empty.")
    if addK:
        chains = add_n_terminal_lysine(chains)
    msas = [None] * len(chains)
    return [{"name": sanitize_name(name), "seqs": chains, "msas": msas}]


def load_partner(
    name: str,
    seq: Optional[str],
    fasta: Optional[str],
    msa: Optional[str],
    role: str,
) -> Dict[str, object]:
    """
    Load a target or antitarget from CLI args.
    Returns dict: {name, role, seqs, msas}
    """
    if seq and fasta:
        raise ValueError(f"{role} {name!r}: specify only one of sequence or FASTA.")

    if seq:
        seqs = _split_multi_chain(seq)
    elif fasta:
        fasta_path = Path(fasta).resolve()
        if not fasta_path.is_file():
            raise ValueError(f"{role} FASTA not found: {fasta_path}")
        seqs = [_clean_seq_string(s) for s in read_fasta_multi(fasta_path)]
    else:
        raise ValueError(f"{role} {name!r}: must specify sequence or FASTA.")

    if not seqs:
        raise ValueError(f"{role} {name!r}: no sequences found.")

    msas: List[Optional[str]] = [None] * len(seqs)
    if msa:
        # Attach MSA to chain 0 only by default
        msas[0] = str(Path(msa).resolve())

    return {"name": sanitize_name(name), "role": role, "seqs": seqs, "msas": msas}


def decide_use_msa_server(
    global_mode: str,
    binder_msas: List[Optional[str]],
    partner_msas: List[Optional[str]],
) -> bool:
    """
    Decide whether to use MSA server for a given binder-partner pair.
    """
    if global_mode == "true":
        return True
    if global_mode == "false":
        return False
    # auto: use server only if neither binder nor partner has explicit MSAs
    any_msa = any(binder_msas) or any(partner_msas)
    return not any_msa


def run_subprocess(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    verbose: bool,
) -> int:
    """
    Run a subprocess in the given cwd, optionally capturing output to log_path.
    Returns the process return code.
    """
    cwd.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        proc = subprocess.run(cmd, cwd=str(cwd))
        return proc.returncode

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(proc.stdout or "")
    return proc.returncode


def append_binder_summary(
    binder_dir: Path,
    out_root: Path,
    metric_preference: str = "ipSAE_min",
) -> None:
    """
    Read binder_*/plots/ipsae_summary.csv, compute mean/std of selected metric
    for target and antitarget, append a one-row summary to out_root/summary/*.csv
    and print the values.
    """
    plots_dir = binder_dir / "plots"
    csv_path = plots_dir / "ipsae_summary.csv"
    if not csv_path.is_file():
        print(f"    !! No ipsae_summary.csv for {binder_dir.name}; skipping summary row.")
        return

    df = pd.read_csv(csv_path)

    # Backwards-compat: ensure target_type is present
    if "target_type" not in df.columns and "vs" in df.columns:
        df["target_type"] = "unknown"

    # Choose metric column
    preferred = [metric_preference, "ipSAE_min", "ipSAE_max", "ipSAE"]
    metric_col = next((m for m in preferred if m in df.columns), None)
    if metric_col is None:
        print(f"    !! No ipSAE metric columns found in {csv_path}; skipping summary row.")
        return

    binder_short = binder_dir.name.replace("binder_", "", 1)

    def stats_for_type(tt: str) -> tuple[Optional[float], Optional[float], int]:
        sub = df[df["target_type"] == tt]
        n = len(sub)
        if n == 0:
            return None, None, 0
        mean = float(sub[metric_col].mean())
        std = float(sub[metric_col].std(ddof=1)) if n > 1 else 0.0
        return mean, std, n

    tgt_mean, tgt_std, tgt_n = stats_for_type("target")
    at_mean, at_std, at_n = stats_for_type("antitarget")

    # Print to console
    if tgt_n > 0:
        print(
            f"    Binder '{binder_short}': target {metric_col} "
            f"mean={tgt_mean:.3f}, std={tgt_std:.3f} (n={tgt_n})"
        )
    else:
        print(f"    Binder '{binder_short}': no target rows found in ipsae_summary.csv")

    if at_n > 0:
        print(
            f"                             antitarget {metric_col} "
            f"mean={at_mean:.3f}, std={at_std:.3f} (n={at_n})"
        )

    # Append to global summary CSV
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "binder_pair_summary.csv"

    new_file = not summary_path.is_file()
    with summary_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(
                [
                    "binder_name",
                    "metric",
                    "target_ipSAE_mean",
                    "target_ipSAE_std",
                    "antitarget_ipSAE_mean",
                    "antitarget_ipSAE_std",
                    "n_target_models",
                    "n_antitarget_models",
                ]
            )
        writer.writerow(
            [
                binder_short,
                metric_col,
                tgt_mean,
                tgt_std,
                at_mean,
                at_std,
                tgt_n,
                at_n,
            ]
        )


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Make visualise_binder_validation importable
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    try:
        import visualise_binder_validation as viz  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - import error surface
        raise SystemExit(
            f"ERROR: could not import visualise_binder_validation.py from {script_dir} "
            f"({exc})"
        )

    print("Stage 1/2: Parsing inputs and configuring binders/partners...")

    addK = bool(args.add_n_terminal_lysine)

    # --- Load binders ---
    if args.binder_csv:
        binders = load_binders_from_csv(
            Path(args.binder_csv).resolve(),
            name_col=args.binder_name_col,
            seq_col=args.binder_seq_col,
            addK=addK,
        )
    elif args.binder_fasta_dir:
        binders = load_binders_from_dir(
            Path(args.binder_fasta_dir).resolve(),
            addK=addK,
        )
    elif args.binder_fasta:
        if not args.binder_name:
            raise SystemExit(
                "ERROR: --binder_name is required when using --binder_fasta."
            )
        binders = load_single_binder_from_fasta(
            args.binder_name,
            Path(args.binder_fasta).resolve(),
            addK=addK,
        )
    elif args.binder_seq:
        if not args.binder_name:
            raise SystemExit(
                "ERROR: --binder_name is required when using --binder_seq."
            )
        binders = load_single_binder_from_seq(
            args.binder_name,
            args.binder_seq,
            addK=addK,
        )
    else:
        raise SystemExit("ERROR: No binder source provided.")

    # --- Load target (required) ---
    target = load_partner(
        name=args.target_name,
        seq=args.target_seq,
        fasta=args.target_fasta,
        msa=args.target_msa,
        role="target",
    )

    # --- Load antitarget (optional) ---
    antitarget = None
    if args.antitarget_name:
        if not (args.antitarget_seq or args.antitarget_fasta):
            raise SystemExit(
                "ERROR: antitarget_name was provided but neither "
                "--antitarget_seq nor --antitarget_fasta was given."
            )
        antitarget = load_partner(
            name=args.antitarget_name,
            seq=args.antitarget_seq,
            fasta=args.antitarget_fasta,
            msa=args.antitarget_msa,
            role="antitarget",
        )

    partners: List[Dict[str, object]] = [target]
    if antitarget is not None:
        partners.append(antitarget)

    if not partners:
        raise SystemExit("ERROR: At least one target/antitarget must be defined.")

    print(
        f"  Loaded {len(binders)} binder(s) and "
        f"{1 + int(antitarget is not None)} partner type(s) (target / antitarget / self)."
    )

    # ------------------------------------------------------------------
    # Stage 2: Binder‑by‑binder Boltz + ipSAE
    # ------------------------------------------------------------------
    print(
        f"Stage 2/2: Running Boltz + ipSAE per binder ({len(binders)} total binder(s))..."
    )

    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Dummy args object for analyse_binder (needs ipsae_e / ipsae_d)
    ipsae_args = types.SimpleNamespace(
        ipsae_e=int(args.ipsae_pae_cutoff),
        ipsae_d=int(args.ipsae_dist_cutoff),
    )

    for idx, binder in enumerate(binders, start=1):
        bname = binder["name"]  # type: ignore[assignment]
        bseqs = binder["seqs"]  # type: ignore[assignment]
        bmsas = binder["msas"]  # type: ignore[assignment]

        binder_dir = out_root / f"binder_{bname}"
        binder_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\nBinder {idx}/{len(binders)}: '{bname}' "
            f"(chains={len(bseqs)}, partners={1 + int(antitarget is not None) + int(args.include_self)})"
        )

        # --------------------------------------------------------------
        # 2a. Build YAMLs for this binder
        # --------------------------------------------------------------
        partner_jobs: List[Dict[str, object]] = []

        for partner in partners:
            role = partner["role"]  # type: ignore[index]
            pname = partner["name"]  # type: ignore[index]
            tseqs = partner["seqs"]  # type: ignore[index]
            tmsas = partner["msas"]  # type: ignore[index]

            yaml_stem = f"binder_{bname}_vs_{role}_{pname}"
            yaml_path = binder_dir / f"{yaml_stem}.yaml"

            yaml_text = yaml_for_pair(
                binder_seqs=bseqs,  # type: ignore[arg-type]
                partner_seqs=tseqs,  # type: ignore[arg-type]
                partner_role=role,  # type: ignore[arg-type]
                binder_msas=bmsas,  # type: ignore[arg-type]
                partner_msas=tmsas,  # type: ignore[arg-type]
            )
            yaml_path.write_text(yaml_text, encoding="utf-8")

            partner_jobs.append(
                {
                    "role": role,
                    "partner": pname,
                    "yaml_path": yaml_path,
                    "binder_msas": bmsas,
                    "partner_msas": tmsas,
                }
            )

        if args.include_self:
            role = "self"
            pname = "self"
            tseqs = list(bseqs)
            tmsas = list(bmsas)
            yaml_stem = f"binder_{bname}_vs_self"
            yaml_path = binder_dir / f"{yaml_stem}.yaml"

            yaml_text = yaml_for_pair(
                binder_seqs=bseqs,  # type: ignore[arg-type]
                partner_seqs=tseqs,  # type: ignore[arg-type]
                partner_role=role,
                binder_msas=bmsas,  # type: ignore[arg-type]
                partner_msas=tmsas,  # type: ignore[arg-type]
            )
            yaml_path.write_text(yaml_text, encoding="utf-8")

            partner_jobs.append(
                {
                    "role": role,
                    "partner": pname,
                    "yaml_path": yaml_path,
                    "binder_msas": bmsas,
                    "partner_msas": tmsas,
                }
            )

        if not partner_jobs:
            print("  !! No partner jobs created for this binder; skipping.")
            continue

        # --------------------------------------------------------------
        # 2b. Run Boltz for this binder vs all partners
        # --------------------------------------------------------------
        print(f"  Running Boltz for {len(partner_jobs)} complex(es)...")

        for jdx, job in enumerate(partner_jobs, start=1):
            role = job["role"]
            partner_name = job["partner"]
            yaml_path = job["yaml_path"]
            bmsas = job["binder_msas"]
            tmsas = job["partner_msas"]

            out_dir = binder_dir / "outputs"
            use_msa = decide_use_msa_server(
                args.use_msa_server,
                binder_msas=bmsas,  # type: ignore[arg-type]
                partner_msas=tmsas,  # type: ignore[arg-type]
            )

            print(
                f"    [{jdx}/{len(partner_jobs)}] Boltz: binder='{bname}' "
                f"vs {role}='{partner_name}'..."
            )

            cmd = [
                sys.executable,
                "-m",
                "boltz.main",
                "predict",
                yaml_path.name,
                "--out_dir",
                str(out_dir),
                "--recycling_steps",
                str(args.recycling_steps),
                "--diffusion_samples",
                str(args.diffusion_samples),
            ]
            if use_msa:
                cmd.append("--use_msa_server")

            log_name = f"boltz_binder_{bname}_vs_{role}_{partner_name}.log"
            log_path = logs_dir / sanitize_name(log_name)

            returncode = run_subprocess(
                cmd=cmd,
                cwd=binder_dir,
                log_path=log_path,
                verbose=args.verbose,
            )
            if returncode != 0:
                print(
                    f"      !! Boltz failed for binder='{bname}' vs {role}='{partner_name}'. "
                    f"See log: {log_path}"
                )
            elif not args.verbose:
                print(f"      Done. (log: {log_path})")

        # --------------------------------------------------------------
        # 2c. Run ipSAE for this binder only
        # --------------------------------------------------------------
        print("  Running ipSAE + binder‑level summaries...")
        viz.analyse_binder(binder_dir, ipsae_args)
        append_binder_summary(
            binder_dir=binder_dir,
            out_root=out_root,
            metric_preference="ipSAE_min",
        )

    # --------------------------------------------------------------
    # Final global summaries / heatmaps (optional but kept)
    # --------------------------------------------------------------
    print(
        "\nFinalizing global ipSAE summaries and heatmaps across all binders..."
    )

    viz.plot_overall(
        out_root,
        use_best_model=args.use_best_model,
    )

    print(
        f"\nDone. Results are under: {out_root}\n"
        f"  - Per-binder summaries and plots: {out_root}/binder_*/plots\n"
        f"  - Compact binder summary table:   {out_root}/summary/binder_pair_summary.csv\n"
        f"  - Global ipSAE data & heatmaps:   {out_root}/summary/\n"
    )


if __name__ == "__main__":
    main()

