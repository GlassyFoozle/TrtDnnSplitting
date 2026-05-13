"""
24_backfill_interval_timings.py — Populate interval timing.json GPU fields from eval JSONs.

Reads results/evaluations/**/*_fp32.json (or fp16), extracts per_chunk_gpu_mean_ms,
per_chunk_gpu_p99_ms, and per_chunk_gpu_max_ms from each eval record, and writes
them back to the matching interval timing.json files in artifacts/chunk_cache/.

This allows cache-only assembly (can_assemble_from_intervals) to work for masks whose
intervals were built and profiled in a prior live run, but whose interval timing.json
files were never populated with GPU timing data (pre-commit 9c02444 behavior).

Usage
-----
  # Dry-run: show what would be written
  python scripts/24_backfill_interval_timings.py --dry-run

  # Actually backfill
  python scripts/24_backfill_interval_timings.py

  # Specific precision
  python scripts/24_backfill_interval_timings.py --precision fp16

  # Specific model
  python scripts/24_backfill_interval_timings.py --model vgg19
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def parse_args():
    ap = argparse.ArgumentParser(description="Backfill interval GPU timing from eval JSONs")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--model", default=None, help="Only backfill this model (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="Print what would change without writing")
    ap.add_argument("--verbose", action="store_true", help="Print every interval updated")
    return ap.parse_args()


def main():
    args = parse_args()
    precision = args.precision

    from src.optimization.config_evaluator import (
        backfill_interval_gpu_timing_from_evals,
        _load_interval_timing,
    )

    eval_dir = REPO / "results" / "evaluations"
    if not eval_dir.exists():
        print("No results/evaluations/ directory found. Nothing to backfill.")
        return

    models = []
    for model_dir in sorted(eval_dir.iterdir()):
        if model_dir.is_dir():
            if args.model is None or model_dir.name == args.model:
                models.append(model_dir.name)

    if not models:
        print(f"No models found in {eval_dir} (filter: {args.model!r})")
        return

    total_intervals_updated = 0
    total_intervals_already_ok = 0
    total_evals_processed = 0

    for model in models:
        print(f"\n[{model}] scanning evaluations...")
        eval_jsons = list((eval_dir / model).glob(f"*_{precision}.json"))
        print(f"  Found {len(eval_jsons)} eval JSONs for {precision}")

        # Collect per-interval GPU timing data from all eval JSONs
        # A group (e.g. [0,1,2]) → (mean_ms, p99_ms, max_ms) from eval chunk data.
        interval_data: dict[str, tuple[float, float, float | None]] = {}

        for eval_path in eval_jsons:
            try:
                d = json.loads(eval_path.read_text())
            except Exception as e:
                print(f"  WARN: could not read {eval_path.name}: {e}")
                continue
            if d.get("error"):
                continue
            means = d.get("per_chunk_gpu_mean_ms")
            p99s  = d.get("per_chunk_gpu_p99_ms")
            maxs  = d.get("per_chunk_gpu_max_ms") or []
            groups = d.get("groups")
            if not (means and p99s and groups):
                continue
            if len(means) != len(p99s) or len(means) != len(groups):
                continue
            total_evals_processed += 1
            for i, grp in enumerate(groups):
                int_key = f"int_{grp[0]}_{grp[-1]}"
                if int_key not in interval_data:
                    interval_data[int_key] = (
                        float(means[i]),
                        float(p99s[i]),
                        float(maxs[i]) if i < len(maxs) and maxs[i] is not None else None,
                    )
                # If already set, keep the first (deterministic — same interval should give same timing)

        # Write back to interval timing.json files
        cache_model_dir = REPO / "artifacts" / "chunk_cache" / model
        if not cache_model_dir.exists():
            print(f"  No chunk_cache dir for {model} — skipping")
            continue

        for int_key, (mean_ms, p99_ms, max_ms) in sorted(interval_data.items()):
            int_dir = cache_model_dir / int_key
            if not int_dir.exists():
                if args.verbose:
                    print(f"    WARN: {int_key} dir not found in chunk_cache")
                continue

            timing_path = int_dir / "timing.json"
            if timing_path.exists():
                try:
                    timing = json.loads(timing_path.read_text())
                except Exception:
                    timing = {}
            else:
                timing = {}

            mean_key = f"gpu_mean_ms_{precision}"
            p99_key  = f"gpu_p99_ms_{precision}"
            max_key  = f"gpu_max_ms_{precision}"

            if timing.get(mean_key) and timing.get(p99_key) and timing.get(max_key):
                total_intervals_already_ok += 1
                if args.verbose:
                    print(f"    {int_key}: already has GPU timing ({timing[mean_key]:.4f} ms)")
                continue

            if args.dry_run:
                max_part = f", {max_key}={max_ms:.4f}" if max_ms is not None else ""
                print(
                    f"    [dry-run] {int_key}: would write "
                    f"{mean_key}={mean_ms:.4f}, {p99_key}={p99_ms:.4f}{max_part}"
                )
                total_intervals_updated += 1
                continue

            timing[mean_key] = mean_ms
            timing[p99_key]  = p99_ms
            if max_ms is not None:
                timing[max_key] = max_ms
            timing.setdefault("model", model)
            timing_path.parent.mkdir(parents=True, exist_ok=True)
            timing_path.write_text(json.dumps(timing, indent=2))
            total_intervals_updated += 1
            if args.verbose:
                max_part = f", {max_key}={max_ms:.4f}" if max_ms is not None else ""
                print(
                    f"    {int_key}: wrote "
                    f"{mean_key}={mean_ms:.4f}, {p99_key}={p99_ms:.4f}{max_part}"
                )

        print(f"  Processed {total_evals_processed} eval JSONs, "
              f"updated {total_intervals_updated} intervals")

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Summary:")
    print(f"  Eval JSONs processed: {total_evals_processed}")
    print(f"  Intervals already had GPU timing: {total_intervals_already_ok}")
    print(f"  Intervals {'that would be' if args.dry_run else ''} updated: {total_intervals_updated}")

    if args.dry_run:
        print("\nRe-run without --dry-run to write the timing data.")
    else:
        print("\nDone. Run scripts/23_inspect_cache_coverage.py to verify coverage.")


if __name__ == "__main__":
    main()
