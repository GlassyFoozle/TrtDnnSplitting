#!/usr/bin/env python3
"""Delete cache records that are not explicitly backed by sync wall-time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def remove(path: Path, *, dry_run: bool) -> None:
    if not dry_run:
        path.unlink(missing_ok=True)


def purge_eval_caches(*, dry_run: bool) -> tuple[int, int]:
    removed = kept = 0
    root = REPO / "results" / "evaluations"
    for path in root.glob("*/*_fp*.json"):
        if path.stem.endswith("_cpp_raw"):
            continue
        data = load_json(path)
        if data and data.get("timing_basis") == "cpu_wall_sync":
            kept += 1
            continue
        remove(path, dry_run=dry_run)
        raw = path.with_name(f"{path.stem}_cpp_raw.json")
        remove(raw, dry_run=dry_run)
        removed += 1
    return removed, kept


def purge_interval_timings(*, dry_run: bool) -> tuple[int, int]:
    removed = kept = 0
    for path in (REPO / "artifacts" / "chunk_cache").glob("*/int_*/timing.json"):
        data = load_json(path)
        if not data:
            remove(path, dry_run=dry_run)
            removed += 1
            continue
        wall_keys = [k for k in data if k.startswith("timing_basis_")]
        keep = any(data.get(key) == "cpu_wall_sync" for key in wall_keys)
        if keep:
            kept += 1
            continue
        remove(path, dry_run=dry_run)
        removed += 1
    return removed, kept


def purge_profiling_db(*, dry_run: bool) -> int:
    path = REPO / "results" / "optimization" / ".profiling_cache.json"
    data = load_json(path) or {"version": 1, "entries": {}}
    entries = data.get("entries", {})
    kept: dict[str, dict] = {}
    for key, entry in entries.items():
        source = entry.get("source_json")
        source_path = Path(source) if source else None
        if source_path and not source_path.is_absolute():
            source_path = REPO / source_path
        source_data = load_json(source_path) if source_path else None
        if source_data and source_data.get("timing_basis") == "cpu_wall_sync":
            kept[key] = entry
    removed = len(entries) - len(kept)
    if not dry_run:
        path.write_text(json.dumps({"version": data.get("version", 1), "entries": kept}, indent=2))
    return removed


def purge_apply_k_cache(*, dry_run: bool) -> tuple[int, int]:
    path = REPO / "results" / "optimization" / "measured_k_split_cache.json"
    data = load_json(path) or {"version": 1, "entries": {}}
    entries = data.get("entries", {})
    kept: dict[str, dict] = {}
    for key, entry in entries.items():
        source = entry.get("profile_result_path")
        source_path = Path(source) if source else None
        if source_path and not source_path.is_absolute():
            source_path = REPO / source_path
        source_data = load_json(source_path) if source_path else None
        if source_data and source_data.get("timing_basis") == "cpu_wall_sync":
            kept[key] = entry
    removed = len(entries) - len(kept)
    if not dry_run:
        path.write_text(json.dumps({"version": data.get("version", 1), "entries": kept}, indent=2))
    return removed, len(kept)


def main() -> int:
    args = parse_args()
    eval_removed, eval_kept = purge_eval_caches(dry_run=args.dry_run)
    interval_removed, interval_kept = purge_interval_timings(dry_run=args.dry_run)
    profiling_removed = purge_profiling_db(dry_run=args.dry_run)
    apply_k_removed, apply_k_kept = purge_apply_k_cache(dry_run=args.dry_run)
    mode = "would remove" if args.dry_run else "removed"
    print(f"eval caches {mode}={eval_removed} kept_wall={eval_kept}")
    print(f"interval timings {mode}={interval_removed} kept_wall={interval_kept}")
    print(f"profiling DB entries {mode}={profiling_removed}")
    print(f"apply-K entries {mode}={apply_k_removed} kept_wall={apply_k_kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
