#!/usr/bin/env python3
"""Print non-split and policy-full-split WCETs from measured-K cache only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO / "results" / "optimization" / "measured_k_split_cache.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--precision", required=True, choices=("fp16", "fp32"))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--policy", default="trt_fusion_safe")
    ap.add_argument("--wcet-metric", default="max", choices=("max", "p99", "mean"))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    return ap.parse_args()


def fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def chunk_text(values: Iterable[float]) -> str:
    return ", ".join(fmt(float(v)) for v in values)


def load_entries(args: argparse.Namespace) -> list[dict]:
    raw = json.loads(Path(args.cache).read_text())
    selected_models = set(args.models or [])
    entries: list[dict] = []
    for entry in raw.get("entries", {}).values():
        if entry.get("precision") != args.precision:
            continue
        if entry.get("policy_name") != args.policy:
            continue
        if entry.get("wcet_metric") != args.wcet_metric:
            continue
        if selected_models and entry.get("model_name") not in selected_models:
            continue
        times = entry.get("selected_chunk_times")
        if not isinstance(times, list) or not times:
            continue
        entries.append(entry)
    return entries


def endpoint_rows(entries: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = {}
    for entry in entries:
        by_model.setdefault(str(entry["model_name"]), []).append(entry)

    rows: list[dict] = []
    for model, model_entries in sorted(by_model.items()):
        max_k = max(int(e["k"]) for e in model_entries)
        by_k = {int(e["k"]): e for e in model_entries}
        for label, k in (("non-split", 1), ("full-split", max_k)):
            entry = by_k.get(k)
            if entry is None:
                continue
            times = [float(v) for v in entry["selected_chunk_times"]]
            rows.append(
                {
                    "model": model,
                    "config": label,
                    "k": k,
                    "chunk_wcet_ms": times,
                    "total_g_ms": sum(times),
                }
            )
    return rows


def print_markdown(rows: list[dict], precision: str) -> None:
    print(f"# Endpoint WCET from measured-K cache ({precision})")
    print()
    print("| model | config | K | per-chunk WCET (ms) | total G / sum WCET (ms) |")
    print("|---|---|---:|---|---:|")
    for row in rows:
        print(
            f"| {row['model']} | {row['config']} | {row['k']} | "
            f"{chunk_text(row['chunk_wcet_ms'])} | {fmt(row['total_g_ms'])} |"
        )


def main() -> int:
    args = parse_args()
    rows = endpoint_rows(load_entries(args))
    if not rows:
        print("[warn] no matching endpoint entries found")
        return 1
    print_markdown(rows, args.precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
