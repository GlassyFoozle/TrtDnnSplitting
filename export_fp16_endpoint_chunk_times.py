#!/usr/bin/env python3
"""Export FP16 non-split/full-split chunk timings from the measured-K cache."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parent
DEFAULT_CACHE = REPO / "results" / "optimization" / "measured_k_split_cache.json"
DEFAULT_OUTPUT = REPO / "fp16_endpoint_chunk_times.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--policy", default="trt_fusion_safe")
    ap.add_argument("--wcet-metric", default="max", choices=("max", "p99", "mean"))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument(
        "--timing-field",
        default="adjusted_interval_wall_max",
        choices=("adjusted_interval_wall_max", "selected_chunk_times", "measured_selected_chunk_times"),
        help=(
            "Timing source to export. adjusted_interval_wall_max rebuilds values "
            "from the current interval cache; selected_chunk_times uses the K-cache "
            "snapshot; measured_selected_chunk_times is raw measured timing."
        ),
    )
    return ap.parse_args()


def load_endpoint_rows(args: argparse.Namespace) -> list[dict]:
    raw = json.loads(Path(args.cache).read_text())
    selected_models = set(args.models or [])
    by_model: dict[str, list[dict]] = {}

    for entry in raw.get("entries", {}).values():
        if entry.get("precision") != "fp16":
            continue
        if entry.get("policy_name") != args.policy:
            continue
        if entry.get("wcet_metric") != args.wcet_metric:
            continue
        model = str(entry.get("model_name") or "")
        if not model or (selected_models and model not in selected_models):
            continue
        times = _entry_times(entry, args.timing_field)
        if not isinstance(times, list) or not times:
            continue
        by_model.setdefault(model, []).append(entry)

    rows: list[dict] = []
    for model, entries in sorted(by_model.items()):
        by_k = {int(entry["k"]): entry for entry in entries}
        max_k = max(by_k)
        for config, k in (("non-split", 1), ("full-split", max_k)):
            entry = by_k.get(k)
            if entry is None:
                continue
            times = [float(v) for v in _entry_times(entry, args.timing_field)]
            total = sum(times)
            for chunk_index, chunk_time in enumerate(times, start=1):
                rows.append(
                    {
                        "model": model,
                        "config": config,
                        "k": k,
                        "chunk_index": chunk_index,
                        "chunk_time_ms": chunk_time,
                        "total_time_ms": total,
                        "timing_field": args.timing_field,
                    }
                )
    return rows


def _entry_times(entry: dict, timing_field: str) -> list[float] | None:
    if timing_field != "adjusted_interval_wall_max":
        times = entry.get(timing_field)
        return times if isinstance(times, list) else None
    model = str(entry.get("model_name") or "")
    mask = entry.get("mask")
    if not model or not isinstance(mask, list):
        return None
    groups = _mask_groups(mask)
    times: list[float] = []
    for group in groups:
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
            times.append(float(timing["wall_max_ms_fp16"]))
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
    return times


def _mask_groups(mask: list[int]) -> list[list[int]]:
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


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "model",
        "config",
        "k",
        "chunk_index",
        "chunk_time_ms",
        "total_time_ms",
        "timing_field",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = load_endpoint_rows(args)
    if not rows:
        print("[warn] no matching FP16 endpoint entries found")
        return 1
    output = Path(args.output)
    write_csv(output, rows)
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
