#!/usr/bin/env python3
"""
Run script 30, then generate split-activity boxplots and histograms.

This is intentionally a thin wrapper around:
  - scripts/30_run_yaml_fig4_experiment.py
  - scripts/32_plot_early_stop_counters.py

All unknown CLI arguments are passed through to script 30. Wrapper-only plotting
options are prefixed with --plot-*.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence


REPO = Path(__file__).resolve().parent.parent
SCRIPT_30 = REPO / "scripts" / "30_run_yaml_fig4_experiment.py"
SCRIPT_31 = REPO / "scripts" / "31_plot_fig4.py"
SCRIPT_32 = REPO / "scripts" / "32_plot_early_stop_counters.py"
DEFAULT_OUTPUT_DIR = REPO / "results" / "dnn_experiments"

DEFAULT_SPLIT_METRICS = [
    "k_split_calls",
    "k_split_unique_model_chunks",
    "k_split_unique_model_masks",
    "k_split_unique_task_chunks",
    "k_split_unique_task_masks",
]


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run scripts/30_run_yaml_fig4_experiment.py and then plot split "
            "counts/counters from per_taskset_results.csv."
        )
    )
    parser.add_argument(
        "--plot-metrics",
        nargs="+",
        default=DEFAULT_SPLIT_METRICS,
        help="per_taskset_results.csv columns to plot after script 30 completes.",
    )
    parser.add_argument(
        "--plot-algorithms",
        nargs="+",
        default=None,
        help=(
            "Algorithm labels to include in plots. Defaults to every "
            "algorithm_label found in per_taskset_results.csv."
        ),
    )
    parser.add_argument(
        "--plot-output-prefix",
        default="split_counters",
        help="Output prefix for boxplot/histogram files in the run directory.",
    )
    parser.add_argument("--plot-dpi", type=int, default=150)
    parser.add_argument("--plot-hist-bins", type=int, default=24)
    parser.add_argument(
        "--plot-max-utilization",
        type=float,
        default=None,
        help=(
            "Maximum utilization included in plots. By default this wrapper "
            "includes every utilization in the run."
        ),
    )
    parser.add_argument(
        "--no-split-plots",
        action="store_true",
        help="Only run script 30; skip all post-run plots.",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print the script 30 and plotting commands before running them.",
    )
    return parser.parse_known_args(argv)


def _arg_value(args: Sequence[str], name: str) -> str | None:
    prefix = name + "="
    for idx, arg in enumerate(args):
        if arg.startswith(prefix):
            return arg[len(prefix) :]
        if arg == name and idx + 1 < len(args):
            return args[idx + 1]
    return None


def _has_flag(args: Sequence[str], name: str) -> bool:
    return name in args or any(arg.startswith(name + "=") for arg in args)


def _default_run_name(script30_args: Sequence[str]) -> str:
    config = _arg_value(script30_args, "--config")
    stem = Path(config).stem if config else "run"
    mode = "live" if _has_flag(script30_args, "--live") else "dry"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"yaml_fig4_{stem}_{mode}_{stamp}"


def _resolve_output_dir(script30_args: Sequence[str]) -> Path:
    value = _arg_value(script30_args, "--output-dir")
    if value:
        path = Path(value)
        return path if path.is_absolute() else REPO / path
    return DEFAULT_OUTPUT_DIR


def _read_csv_header(path: Path) -> list[str]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])


def _read_algorithm_labels(path: Path) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("algorithm_label") or row.get("algorithm")
            if label and label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def _filter_existing_algorithms(csv_path: Path, requested: Sequence[str] | None) -> list[str]:
    existing = _read_algorithm_labels(csv_path)
    if requested is None:
        return existing
    existing_set = set(existing)
    kept = [label for label in requested if label in existing_set]
    missing = [label for label in requested if label not in existing_set]
    if missing:
        print(
            "[warn] skipping missing plot algorithm(s): " + ", ".join(missing),
            file=sys.stderr,
        )
    return kept


def _write_ratio_without_fb_off(run_dir: Path) -> Path:
    src = run_dir / "schedulability_ratio.csv"
    dst = run_dir / "schedulability_ratio_without_fb_off.csv"
    with src.open(newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    kept = [row for row in rows if (row.get("algorithm") or "") != "SS-tol-fb-off"]
    with dst.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    return dst


def _filter_existing_metrics(csv_path: Path, requested: Iterable[str]) -> list[str]:
    header = set(_read_csv_header(csv_path))
    existing: list[str] = []
    missing: list[str] = []
    for metric in requested:
        if metric in header:
            existing.append(metric)
        else:
            missing.append(metric)
    if missing:
        print(
            "[warn] skipping missing plot metric(s): " + ", ".join(missing),
            file=sys.stderr,
        )
    return existing


def _run(cmd: Sequence[str], print_command: bool) -> int:
    if print_command:
        print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(list(cmd)).returncode


def main(argv: Sequence[str] | None = None) -> int:
    plot_args, script30_args = parse_args(list(argv or sys.argv[1:]))

    if not script30_args:
        print("[error] pass the script-30 arguments after any --plot-* options", file=sys.stderr)
        return 2

    run_name = _arg_value(script30_args, "--run-name")
    forwarded = list(script30_args)
    if not run_name:
        run_name = _default_run_name(script30_args)
        forwarded += ["--run-name", run_name]

    out_dir = _resolve_output_dir(forwarded)
    run_dir = out_dir / run_name

    cmd30 = [sys.executable, str(SCRIPT_30), *forwarded]
    rc = _run(cmd30, plot_args.print_commands)
    if rc != 0:
        print(f"[error] script 30 failed with exit code {rc}; skipping plots", file=sys.stderr)
        return rc

    if plot_args.no_split_plots:
        return 0

    csv_path = run_dir / "per_taskset_results.csv"
    if not csv_path.exists():
        print(f"[error] missing expected CSV: {csv_path}", file=sys.stderr)
        return 1

    metrics = _filter_existing_metrics(csv_path, plot_args.plot_metrics)
    if not metrics:
        print("[error] none of the requested plot metrics exist in the CSV", file=sys.stderr)
        return 1

    algorithms = _filter_existing_algorithms(csv_path, plot_args.plot_algorithms)
    if not algorithms:
        print("[warn] no requested plot algorithms found; skipping split-counter plots", file=sys.stderr)
        algorithms = []

    if algorithms:
        cmd32 = [
            sys.executable,
            str(SCRIPT_32),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(run_dir),
            "--output-prefix",
            plot_args.plot_output_prefix,
            "--metrics",
            *metrics,
            "--algorithms",
            *algorithms,
            "--dpi",
            str(plot_args.plot_dpi),
            "--hist-bins",
            str(plot_args.plot_hist_bins),
            "--max-utilization",
            str(plot_args.plot_max_utilization if plot_args.plot_max_utilization is not None else 1.0e9),
        ]
        rc = _run(cmd32, plot_args.print_commands)
        if rc != 0:
            return rc

    ratio_csv = _write_ratio_without_fb_off(run_dir)
    cmd31 = [
        sys.executable,
        str(SCRIPT_31),
        "--csv",
        str(ratio_csv),
        "--output-dir",
        str(run_dir),
        "--output",
        "schedulability_ratio",
        "--plot-mode",
        "all",
        "--dpi",
        str(plot_args.plot_dpi),
    ]
    return _run(cmd31, plot_args.print_commands)


if __name__ == "__main__":
    raise SystemExit(main())
