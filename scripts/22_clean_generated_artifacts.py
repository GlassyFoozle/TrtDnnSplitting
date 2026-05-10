"""
22_clean_generated_artifacts.py — Remove old variant-specific artifact copies.

After the path-consolidation fix (interval cache = canonical storage), the old
per-variant artifact dirs under artifacts/onnx/ and artifacts/engines/ are
redundant.  This script removes them.

Safe by default
---------------
  --dry-run      Print what would be deleted without deleting anything (default ON
                 when --yes is absent).
  --yes          Actually delete.  Implied by combining flags but always explicit.

Never deleted
-------------
  artifacts/chunk_cache/   (interval cache: canonical ONNX + engines + timing)
  results/evaluations/     (mask-level eval JSONs)
  results/table4/          (C++ profiling output)
  results/optimization/    (profiling cache + algorithm outputs)
  results/live_*/          (diagnosis + scalability reports)

Usage
-----
  # See what would be removed (safe)
  python scripts/22_clean_generated_artifacts.py --dry-run

  # Remove variant ONNX copies only
  python scripts/22_clean_generated_artifacts.py --onnx --yes

  # Remove everything (onnx + engines + logs)
  python scripts/22_clean_generated_artifacts.py --onnx --engines --logs --yes
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SAFE_DIRS = [
    REPO / "artifacts" / "chunk_cache",
    REPO / "results"   / "evaluations",
    REPO / "results"   / "table4",
    REPO / "results"   / "optimization",
]


def _fmt_size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    if total >= 1 << 30:
        return f"{total / (1 << 30):.1f} GB"
    if total >= 1 << 20:
        return f"{total / (1 << 20):.1f} MB"
    return f"{total / (1 << 10):.1f} KB"


def _collect_targets(args) -> list[tuple[str, Path]]:
    targets: list[tuple[str, Path]] = []

    if args.onnx:
        onnx_root = REPO / "artifacts" / "onnx"
        if onnx_root.exists():
            for model_dir in sorted(onnx_root.iterdir()):
                if model_dir.is_dir():
                    targets.append(("variant-onnx", model_dir))
        else:
            print(f"  [skip] artifacts/onnx/ does not exist")

    if args.engines:
        engine_root = REPO / "artifacts" / "engines"
        if engine_root.exists():
            for model_dir in sorted(engine_root.iterdir()):
                if model_dir.is_dir():
                    targets.append(("variant-engine", model_dir))
        else:
            print(f"  [skip] artifacts/engines/ does not exist")

    if args.logs:
        logs_root = REPO / "artifacts" / "logs"
        if logs_root.exists():
            targets.append(("logs", logs_root))
        else:
            print(f"  [skip] artifacts/logs/ does not exist")

    return targets


def _is_safe(path: Path) -> bool:
    for safe in SAFE_DIRS:
        try:
            path.relative_to(safe)
            return False  # path is inside a protected dir
        except ValueError:
            pass
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove old variant-specific artifact copies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--onnx",    action="store_true", help="Remove artifacts/onnx/<model>/ dirs")
    parser.add_argument("--engines", action="store_true", help="Remove artifacts/engines/<model>/ dirs")
    parser.add_argument("--logs",    action="store_true", help="Remove artifacts/logs/ dir")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Print what would be deleted without deleting (default if --yes absent)")
    parser.add_argument("--yes",     action="store_true", help="Actually delete (required to act)")
    args = parser.parse_args()

    if not any([args.onnx, args.engines, args.logs]):
        parser.print_help()
        print("\nNothing selected.  Pass --onnx, --engines, and/or --logs.")
        sys.exit(0)

    dry_run = args.dry_run or not args.yes

    if dry_run:
        print("DRY RUN — no files will be deleted.  Pass --yes to actually delete.\n")

    targets = _collect_targets(args)
    if not targets:
        print("Nothing to clean.")
        return

    total_freed = 0
    for label, path in targets:
        # Safety check: never touch protected dirs
        if not _is_safe(path):
            print(f"  [PROTECTED] {path.relative_to(REPO)} — skipped")
            continue

        size_str = _fmt_size(path) if path.exists() else "0 B"
        print(f"  {'[dry-run] would remove' if dry_run else 'removing'} [{label}] {path.relative_to(REPO)}  ({size_str})")

        if not dry_run:
            shutil.rmtree(path)
            total_freed += 1

    if dry_run:
        print(f"\nRe-run with --yes to delete {len(targets)} target(s).")
    else:
        print(f"\nRemoved {total_freed} target(s).")
        print("Interval cache (artifacts/chunk_cache/) is intact.")


if __name__ == "__main__":
    main()
