#!/usr/bin/env python3
"""Report intervals whose adjusted wall-max is suspiciously above raw timing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--precisions", nargs="+", default=["fp16", "fp32"], choices=["fp16", "fp32"])
    ap.add_argument("--ratio-threshold", type=float, default=3.0)
    ap.add_argument("--json-output", default=None)
    ap.add_argument("--csv-output", default=None)
    return ap.parse_args()


def interval_rows(models: set[str] | None, precisions: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((REPO / "artifacts" / "chunk_cache").glob("*/int_*_*/timing.json")):
        model = path.parents[1].name
        if models and model not in models:
            continue
        try:
            timing = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for precision in precisions:
            raw = timing.get(f"measured_wall_max_ms_{precision}")
            adjusted = timing.get(f"wall_max_ms_{precision}")
            if raw in (None, 0) or adjusted is None:
                continue
            reasons = timing.get(f"wall_max_ratio_inflation_reasons_{precision}") or []
            rows.append(
                {
                    "model": model,
                    "precision": precision,
                    "interval": path.parent.name,
                    "raw_ms": float(raw),
                    "adjusted_ms": float(adjusted),
                    "ratio": float(adjusted) / float(raw),
                    "reason_count": len(reasons),
                    "latest_parent": reasons[-1].get("parent") if reasons else "",
                    "latest_split": reasons[-1].get("split") if reasons else "",
                    "latest_parent_wall_max_ms": reasons[-1].get("parent_wall_max_ms") if reasons else "",
                    "path": str(path.parent.relative_to(REPO)),
                }
            )
    return rows


def write_outputs(rows: list[dict], args: argparse.Namespace) -> None:
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))
    if args.csv_output:
        out = Path(args.csv_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)


def main() -> int:
    args = parse_args()
    models = set(args.models or []) or None
    rows = interval_rows(models, args.precisions)
    outliers = [row for row in rows if row["ratio"] >= args.ratio_threshold]
    outliers.sort(key=lambda row: (-row["ratio"], row["model"], row["interval"]))
    print(
        f"checked={len(rows)} outliers={len(outliers)} "
        f"ratio_threshold={args.ratio_threshold:.3f}",
        flush=True,
    )
    for row in outliers:
        print(
            f"{row['precision']} {row['model']}/{row['interval']}: "
            f"{row['ratio']:.2f}x raw={row['raw_ms']:.6f} "
            f"adjusted={row['adjusted_ms']:.6f} "
            f"latest_parent={row['latest_parent']}",
            flush=True,
        )
    write_outputs(outliers, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
