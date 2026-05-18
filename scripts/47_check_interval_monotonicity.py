#!/usr/bin/env python3
"""Check whether every cached interval partition is no faster than its parent."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.optimization.monotonic_interval_repair import MONOTONIC_EPSILON_MS


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=None, help="Models to check (default: all)")
    ap.add_argument("--precisions", nargs="+", default=["fp16"], choices=["fp16", "fp32"])
    ap.add_argument(
        "--timing-field",
        default="wall_max",
        choices=["wall_max", "measured_wall_max"],
        help="Check adjusted wall_max or preserved raw measured_wall_max values.",
    )
    ap.add_argument("--json-output", default=None, help="Optional JSON report path")
    ap.add_argument("--csv-output", default=None, help="Optional CSV report path")
    ap.add_argument("--top", type=int, default=20, help="Rows to print per precision (default: 20)")
    return ap.parse_args()


def progress(items: Iterable, *, total: int, desc: str):
    try:
        from tqdm import tqdm

        yield from tqdm(items, total=total, desc=desc, unit="parent", file=sys.stdout)
    except Exception:
        for idx, item in enumerate(items, start=1):
            if idx == 1 or idx == total or idx % max(1, total // 100) == 0:
                print(f"{desc}: {idx}/{total} parents", flush=True)
            yield item


def discover_models(requested: list[str] | None) -> list[str]:
    if requested:
        return sorted(set(requested))
    root = REPO / "artifacts" / "chunk_cache"
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def load_model_intervals(
    model: str,
    precision: str,
    timing_field: str,
) -> tuple[dict[tuple[int, int], Path], dict[tuple[int, int], float]]:
    root = REPO / "artifacts" / "chunk_cache" / model
    by_bounds: dict[tuple[int, int], Path] = {}
    values: dict[tuple[int, int], float] = {}
    key = f"{timing_field}_ms_{precision}"
    if not root.exists():
        return by_bounds, values
    for path in root.iterdir():
        bounds = parse_bounds(path)
        if bounds is None or not (path / f"chunk_{precision}.engine").exists():
            continue
        try:
            timing = json.loads((path / "timing.json").read_text())
            value = float(timing[key])
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        by_bounds[bounds] = path
        values[bounds] = value
    return by_bounds, values


def parse_bounds(path: Path) -> tuple[int, int] | None:
    parts = path.name.split("_")
    if len(parts) != 3 or parts[0] != "int":
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


def violation_rows(model: str, precision: str, timing_field: str) -> tuple[list[dict], dict]:
    by_bounds, values = load_model_intervals(model, precision, timing_field)
    rows: list[dict] = []
    parent_bounds = sorted(
        (bounds for bounds in by_bounds if bounds[0] < bounds[1]),
        key=lambda bounds: (bounds[0], bounds[1]),
    )
    partition_count = 0
    for start, end in progress(parent_bounds, total=len(parent_bounds), desc=f"{model} {precision}"):
        parent_dir = by_bounds[(start, end)]
        count, min_total, min_path = minimum_partition(values, start, end)
        partition_count += count
        if count == 0 or min_total is None:
            continue
        parent_time = values[(start, end)]
        if min_total + MONOTONIC_EPSILON_MS >= parent_time:
            continue
        rows.append(
            {
                "model": model,
                "precision": precision,
                "timing_field": timing_field,
                "parent": str(parent_dir.relative_to(REPO)),
                "parent_time_ms": parent_time,
                "subset_partition_count": count,
                "minimum_subset_partition": [
                    str(by_bounds[bounds].relative_to(REPO)) for bounds in min_path
                ],
                "minimum_subset_times_ms": [values[bounds] for bounds in min_path],
                "minimum_subset_total_ms": min_total,
                "gap_ms": parent_time - min_total,
            }
        )
    rows.sort(key=lambda row: (-row["gap_ms"], row["model"], row["parent"]))
    return rows, {
        "parent_intervals_checked": len(parent_bounds),
        "subset_partitions_checked": partition_count,
    }


def minimum_partition(
    values: dict[tuple[int, int], float],
    start: int,
    end: int,
) -> tuple[int, float | None, list[tuple[int, int]]]:
    """Return count and cheapest cached partition with at least two children."""
    # State is (next_position, segment_count_capped_at_2).
    counts: dict[tuple[int, int], int] = {(start, 0): 1}
    best: dict[tuple[int, int], tuple[float, list[tuple[int, int]]]] = {
        (start, 0): (0.0, [])
    }
    for pos in range(start, end + 1):
        for seg_count in range(3):
            state = (pos, seg_count)
            if state not in counts:
                continue
            base_count = counts[state]
            base_total, base_path = best[state]
            for child_end in range(pos, end + 1):
                child = (pos, child_end)
                if child not in values:
                    continue
                next_state = (child_end + 1, min(2, seg_count + 1))
                counts[next_state] = counts.get(next_state, 0) + base_count
                candidate = (base_total + values[child], base_path + [child])
                if next_state not in best or candidate[0] < best[next_state][0]:
                    best[next_state] = candidate
    final_state = (end + 1, 2)
    if final_state not in counts:
        return 0, None, []
    min_total, min_path = best[final_state]
    return counts[final_state], min_total, min_path


def write_json(path: Path, rows: list[dict], args: argparse.Namespace, models: list[str]) -> None:
    payload = {
        "models": models,
        "precisions": args.precisions,
        "timing_field": args.timing_field,
        "violation_count": len(rows),
        "violations": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "model",
        "precision",
        "timing_field",
        "parent",
        "parent_time_ms",
        "subset_partition_count",
        "minimum_subset_partition",
        "minimum_subset_times_ms",
        "minimum_subset_total_ms",
        "gap_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    models = discover_models(args.models)
    all_rows: list[dict] = []
    for precision in args.precisions:
        rows: list[dict] = []
        parents_checked = 0
        partitions_checked = 0
        for model in models:
            model_rows, stats = violation_rows(model, precision, args.timing_field)
            rows.extend(model_rows)
            parents_checked += stats["parent_intervals_checked"]
            partitions_checked += stats["subset_partitions_checked"]
        all_rows.extend(rows)
        print(
            f"{precision}: checked parents={parents_checked}, "
            f"subset_partitions={partitions_checked}, violations={len(rows)}"
        )
        for row in rows[: args.top]:
            partition = " + ".join(Path(path).name for path in row["minimum_subset_partition"])
            print(
                f"  {row['model']} {Path(row['parent']).name}: "
                f"{partition} = {row['minimum_subset_total_ms']:.6f} "
                f"< {row['parent_time_ms']:.6f} "
                f"(gap={row['gap_ms']:.6f})"
            )
    if args.json_output:
        write_json(Path(args.json_output), all_rows, args, models)
        print(f"json: {args.json_output}")
    if args.csv_output:
        write_csv(Path(args.csv_output), all_rows)
        print(f"csv: {args.csv_output}")
    return 1 if all_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
