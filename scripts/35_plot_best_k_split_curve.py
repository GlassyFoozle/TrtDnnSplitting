#!/usr/bin/env python3
"""
Plot best-K split curves directly from measured_k_split_cache.json.

The cache stores the selected best mask and score for each
model/precision/metric/policy/K tuple. This script does not enumerate candidate
masks, profile, export, build, or read per-mask evaluation files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List


REPO = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO / "results" / "optimization" / "measured_k_split_cache.json"
DEFAULT_OUTPUT_DIR = REPO / "results" / "best_k_split_curves"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot best-K split curves from measured-K cache")
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to plot. Default: every model present in the measured-K cache.",
    )
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    ap.add_argument("--wcet-metric", default="max", choices=["max", "p99", "mean"])
    ap.add_argument("--split-policy", default="trt_fusion_safe")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="measured_k_split_cache.json path")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument(
        "--timing-source",
        default="interval-wall-max",
        choices=["interval-wall-max", "cache-score"],
        help="Use current interval wall_max timings or the score stored in the K cache.",
    )
    ap.add_argument("--dpi", type=int, default=150)
    return ap.parse_args()


def load_rows(cache_path: Path, args: argparse.Namespace) -> List[dict]:
    raw = json.loads(cache_path.read_text())
    rows: List[dict] = []
    selected_models = set(args.models or [])
    for cache_key, entry in raw.get("entries", {}).items():
        model = str(entry.get("model_name", ""))
        precision = str(entry.get("precision", ""))
        metric = str(entry.get("wcet_metric", ""))
        policy = str(entry.get("policy_name", ""))
        if selected_models and model not in selected_models:
            continue
        if precision != args.precision or metric != args.wcet_metric or policy != args.split_policy:
            continue
        score = score_for_entry(entry, args)
        if len(score) < 2:
            continue
        rows.append(
            {
                "model": model,
                "precision": precision,
                "wcet_metric": metric,
                "split_policy": policy,
                "k": int(entry.get("k", 0)),
                "boundary_count": int(entry.get("boundary_count", 0)),
                "enabled_boundaries": ",".join(str(v) for v in entry.get("enabled_boundaries", [])),
                "best_mask": "".join(str(v) for v in entry.get("mask", [])),
                "max_block_ms": float(score[0]),
                "total_g_ms": float(score[1]),
                "spread_ms": float(score[2]) if len(score) > 2 else "",
                "cache_key": cache_key,
            }
        )
    rows.sort(key=lambda r: (str(r["model"]), int(r["k"])))
    return rows


def score_for_entry(entry: dict, args: argparse.Namespace) -> list[float]:
    if args.timing_source == "cache-score":
        return list(entry.get("score") or [])
    model = str(entry.get("model_name") or "")
    mask = entry.get("mask")
    if not model or not isinstance(mask, list):
        return []
    times: list[float] = []
    for group in mask_groups(mask):
        timing_path = (
            REPO
            / "artifacts"
            / "chunk_cache"
            / model
            / f"int_{group[0]}_{group[-1]}"
            / "timing.json"
        )
        try:
            timing = json.loads(timing_path.read_text())
            times.append(float(timing[f"wall_max_ms_{args.precision}"]))
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return []
    return [max(times), sum(times), max(times) - min(times)] if times else []


def mask_groups(mask: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    current = [0]
    for idx, boundary in enumerate(mask):
        if boundary:
            groups.append(current)
            current = [idx + 1]
        else:
            current.append(idx + 1)
    groups.append(current)
    return groups


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "precision",
        "wcet_metric",
        "split_policy",
        "k",
        "boundary_count",
        "enabled_boundaries",
        "best_mask",
        "max_block_ms",
        "total_g_ms",
        "spread_ms",
        "cache_key",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_model(rows: List[dict], output_dir: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    model = str(rows[0]["model"])
    precision = str(rows[0]["precision"])
    metric = str(rows[0]["wcet_metric"])
    policy = str(rows[0]["split_policy"])
    ks = [int(r["k"]) for r in rows]
    max_blocks = [float(r["max_block_ms"]) for r in rows]
    totals = [float(r["total_g_ms"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(ks, max_blocks, marker="o", linewidth=1.8, label="Max block WCET")
    ax.plot(ks, totals, marker="s", linewidth=1.8, label="Total G")
    ax.set_title(f"{model} best-K cache curve ({precision}, {metric}, {policy})")
    ax.set_xlabel("K chunks")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(ks)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    stem = f"{model}_{precision}_{metric}_{policy}_best_k_cache_curve"
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main() -> int:
    args = parse_args()
    cache_path = Path(args.cache)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(cache_path, args)
    by_model: dict[str, List[dict]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    if not by_model:
        print("[warn] no matching measured-K cache entries found")

    for model_rows in by_model.values():
        plot_model(model_rows, output_dir, args.dpi)

    csv_path = output_dir / "best_k_split_curve.csv"
    write_csv(csv_path, rows)
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
