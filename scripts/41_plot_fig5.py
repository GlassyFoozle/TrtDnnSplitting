#!/usr/bin/env python3
"""
63_plot_fig5_design_time.py — Plot Fig.5-style design-phase runtime/cost bars.

Reads fig5_design_time_summary.csv from a script-62 run and saves:
  - <output>_runtime.{png,pdf}
  - <output>_cost.{png,pdf}
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PLOTS_DIR = REPO / "results" / "dnn_experiments" / "plots"


def load_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Run with the conda env, e.g. "
            "`conda run -n trt python scripts/63_plot_fig5_design_time.py ...`, "
            "or install matplotlib."
        ) from exc


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot Fig.5 design-time summary bars")
    ap.add_argument("--run", default=None, help="Run directory containing fig5_design_time_summary.csv")
    ap.add_argument("--csv", default=None, dest="csv_path", help="Explicit summary CSV path")
    ap.add_argument("--output", default="fig5_design_time", help="Output base name")
    ap.add_argument("--output-dir", default=str(PLOTS_DIR))
    ap.add_argument("--title", default="Fig.5-style Design-Time Cost")
    ap.add_argument("--dpi", type=int, default=150)
    return ap.parse_args()


def load_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.csv_path:
        path = Path(args.csv_path)
    elif args.run:
        path = Path(args.run) / "fig5_design_time_summary.csv"
    else:
        raise SystemExit("--run or --csv is required")
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_runtime(rows: List[Dict[str, str]], output_base: Path, title: str, dpi: int) -> None:
    plt = load_matplotlib()

    labels = [r["algorithm"] for r in rows]
    means = [float(r.get("mean_wall_clock_s", 0.0) or 0.0) for r in rows]
    p95 = [float(r.get("p95_wall_clock_s", 0.0) or 0.0) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = range(len(labels))
    ax.bar(x, means, color="#4c78a8", label="mean")
    ax.scatter(list(x), p95, color="#f58518", zorder=3, label="p95")
    ax.set_xticks(list(x), labels, rotation=25, ha="right")
    ax.set_ylabel("Design runtime (s)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_name(output_base.name + "_runtime").with_suffix(".png"), dpi=dpi)
    fig.savefig(output_base.with_name(output_base.name + "_runtime").with_suffix(".pdf"))
    plt.close(fig)


def plot_cost(rows: List[Dict[str, str]], output_base: Path, title: str, dpi: int) -> None:
    plt = load_matplotlib()

    labels = [r["algorithm"] for r in rows]
    masks = [float(r.get("mean_masks_evaluated", 0.0) or 0.0) for r in rows]
    real = [float(r.get("mean_real_profiles", 0.0) or 0.0) for r in rows]
    cache = [float(r.get("mean_cache_hits", 0.0) or 0.0) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = list(range(len(labels)))
    width = 0.25
    ax.bar([v - width for v in x], masks, width=width, label="masks evaluated", color="#4c78a8")
    ax.bar(x, real, width=width, label="real profiles", color="#e45756")
    ax.bar([v + width for v in x], cache, width=width, label="cache hits", color="#72b7b2")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Mean count per taskset")
    ax.set_title(title + " Counts")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_name(output_base.name + "_cost").with_suffix(".png"), dpi=dpi)
    fig.savefig(output_base.with_name(output_base.name + "_cost").with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = load_rows(args)
    if not rows:
        print("[error] no rows", file=sys.stderr)
        return 1
    output_base = Path(args.output_dir) / args.output
    plot_runtime(rows, output_base, args.title, args.dpi)
    plot_cost(rows, output_base, args.title, args.dpi)
    print(f"Saved plots under {output_base.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
