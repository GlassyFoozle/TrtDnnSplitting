#!/usr/bin/env python3
"""Re-measure existing caches with CPU wall-clock timing, without building engines."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
MONOTONIC_EPSILON_MS = 1e-12

from src.integration.mask_applicator import _measured_evenness_score, _store_cached_k_split_mask
from src.optimization.config_evaluator import (
    _compute_merge_groups,
    _eval_json_path,
    _load_interval_timing,
    assemble_from_intervals,
    can_assemble_from_intervals,
    mask_to_variant_name,
)


def tqdm_wrap(items: Iterable[Any], *, desc: str, total: int):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc, total=total, unit="item", file=sys.stdout)
    except Exception:
        return _visible_progress(items, desc=desc, total=total)


def _visible_progress(items: Iterable[Any], *, desc: str, total: int):
    step = max(1, total // 100) if total else 1
    print(f"{desc}: 0/{total} (0.0%)", flush=True)
    for idx, item in enumerate(items, start=1):
        yield item
        if idx == total or idx % step == 0:
            pct = (100.0 * idx / total) if total else 100.0
            print(f"{desc}: {idx}/{total} ({pct:.1f}%)", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=None, help="Models to process (default: all)")
    ap.add_argument("--precisions", nargs="+", default=["fp16", "fp32"], choices=["fp16", "fp32"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-intervals", action="store_true")
    ap.add_argument("--skip-eval-caches", action="store_true")
    ap.add_argument("--skip-profiling-db", action="store_true")
    ap.add_argument("--force-intervals", action="store_true")
    ap.add_argument("--force-eval-caches", action="store_true")
    ap.add_argument(
        "--reprofile-monotonic-violations",
        action="store_true",
        help=(
            "Repeatedly re-profile direct child interval pairs whose wall-max "
            "sum is smaller than their direct parent interval wall-max."
        ),
    )
    ap.add_argument(
        "--inflate-monotonic-children-by-parent-ratio",
        action="store_true",
        help=(
            "Repair direct child monotonic violations without re-profiling by "
            "inflating child wall-max values top-down according to each violating "
            "child pair's current ratio."
        ),
    )
    ap.add_argument(
        "--max-monotonic-reprofile-rounds",
        type=int,
        default=20,
        help=(
            "Maximum extra monotonic-violation re-profile rounds after the "
            "initial interval pass (default: 20)."
        ),
    )
    ap.add_argument(
        "--monotonic-violation-report",
        default=str(REPO / "results" / "monotonic_interval_violations.json"),
        help="Where to write remaining monotonic violations after re-profiling.",
    )
    ap.add_argument(
        "--reprofile-violation-list",
        default=None,
        help=(
            "Re-profile only child intervals listed in a prior monotonic "
            "violation JSON report."
        ),
    )
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers for eval-cache JSON rewrite")
    return ap.parse_args()


def discover_models(requested: List[str] | None) -> List[str]:
    if requested:
        return sorted(set(requested))
    root = REPO / "artifacts" / "chunk_cache"
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def interval_engine_dirs(models: List[str], precision: str) -> List[Path]:
    out: List[Path] = []
    for model in models:
        root = REPO / "artifacts" / "chunk_cache" / model
        if not root.exists():
            continue
        out.extend(
            p for p in sorted(root.iterdir())
            if p.is_dir() and (p / f"chunk_{precision}.engine").exists()
        )
    return out


def remeasure_intervals(args: argparse.Namespace, models: List[str], precision: str) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "reprofile_intervals",
        REPO / "scripts" / "27_reprofile_cached_intervals.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/27_reprofile_cached_intervals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if args.reprofile_violation_list:
        intervals = _load_violation_child_intervals(
            Path(args.reprofile_violation_list),
            models,
            precision,
        )
    else:
        intervals = [
            path for path in interval_engine_dirs(models, precision)
            if args.force_intervals or not _interval_wall_done(path, precision, args.warmup, args.iters)
        ]
    _remeasure_interval_batch(
        args,
        mod,
        intervals,
        precision,
        desc=f"intervals {precision}",
        preserve_observed_max=bool(args.reprofile_violation_list),
    )

    if args.reprofile_monotonic_violations:
        _remeasure_monotonic_violations_until_clear(args, mod, models, precision)


def _remeasure_interval_batch(
    args: argparse.Namespace,
    mod,
    intervals: List[Path],
    precision: str,
    *,
    desc: str,
    preserve_observed_max: bool = False,
) -> None:
    for interval_dir in tqdm_wrap(intervals, desc=desc, total=len(intervals)):
        model = interval_dir.parent.name
        if args.dry_run:
            continue
        previous_timing = _load_timing(interval_dir / "timing.json")
        base_cfg = mod._load_base_config(model)
        with mod.tempfile.TemporaryDirectory(prefix="trtdnn_wall_") as td:
            cfg_path = Path(td) / f"{model}_{interval_dir.name}.json"
            cfg_path.write_text(
                json.dumps(mod._make_profile_config(model, interval_dir, precision, base_cfg), indent=2)
            )
            out_path, wall = _run_table4_quiet(mod, cfg_path, precision, args.warmup, args.iters)
            if out_path is None:
                raise RuntimeError(f"failed interval profile: {model}/{interval_dir.name}/{precision}")
            mod._update_timing(
                interval_dir / "timing.json",
                model,
                interval_dir,
                precision,
                out_path,
                wall,
                args.warmup,
                args.iters,
            )
            if preserve_observed_max:
                _preserve_observed_interval_max(
                    interval_dir / "timing.json",
                    previous_timing,
                    precision,
                )


def _remeasure_monotonic_violations_until_clear(
    args: argparse.Namespace,
    mod,
    models: List[str],
    precision: str,
) -> None:
    if args.dry_run:
        violation_children = _monotonic_violation_child_intervals(models, precision)
        print(
            f"{precision}: monotonic violation child intervals="
            f"{len(violation_children)} (dry-run; no retries)",
            flush=True,
        )
        return

    for round_idx in range(1, args.max_monotonic_reprofile_rounds + 1):
        violation_children = _monotonic_violation_child_intervals(models, precision)
        if not violation_children:
            print(f"{precision}: monotonic violations cleared", flush=True)
            return
        print(
            f"{precision}: monotonic violation round {round_idx} "
            f"child_intervals={len(violation_children)}",
            flush=True,
        )
        _remeasure_interval_batch(
            args,
            mod,
            violation_children,
            precision,
            desc=f"monotonic {precision} round {round_idx}",
            preserve_observed_max=True,
        )

    remaining = _monotonic_violation_child_intervals(models, precision)
    if remaining:
        _write_monotonic_violation_report(args, models, precision)
        raise RuntimeError(
            f"{precision}: monotonic violations remain after "
            f"{args.max_monotonic_reprofile_rounds} rounds "
            f"({len(remaining)} child intervals)"
        )
    _write_monotonic_violation_report(args, models, precision)


def _interval_wall_done(interval_dir: Path, precision: str, warmup: int, iters: int) -> bool:
    timing_path = interval_dir / "timing.json"
    try:
        timing = json.loads(timing_path.read_text())
    except Exception:
        return False
    return (
        timing.get(f"timing_basis_{precision}") == "cpu_wall_sync"
        and timing.get(f"wall_max_ms_{precision}") is not None
        and timing.get(f"profile_warmup_{precision}") == warmup
        and timing.get(f"profile_iters_{precision}") == iters
    )


def _load_timing(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _preserve_observed_interval_max(
    timing_path: Path,
    previous_timing: dict,
    precision: str,
) -> None:
    """
    Keep max timing fields at the greatest value observed across reprofiles.

    Monotonic repair uses wall_max. gpu_max mirrors wall_max in this wall-sync
    pipeline, so keep both aligned with the preserved maximum.
    """
    current = _load_timing(timing_path)
    old_wall = previous_timing.get(f"wall_max_ms_{precision}")
    new_wall = current.get(f"wall_max_ms_{precision}")
    try:
        preserved = max(float(old_wall), float(new_wall))
    except (TypeError, ValueError):
        return
    current[f"wall_max_ms_{precision}"] = preserved
    current[f"gpu_max_ms_{precision}"] = preserved
    current[f"wall_max_observed_preserved_{precision}"] = True
    current[f"latest_measured_wall_max_ms_{precision}"] = float(new_wall)
    timing_path.write_text(json.dumps(current, indent=2))


def _parse_interval_bounds(interval_dir: Path) -> tuple[int, int] | None:
    parts = interval_dir.name.split("_")
    if len(parts) != 3 or parts[0] != "int":
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


def _interval_wall_max(interval_dir: Path, precision: str) -> float | None:
    try:
        timing = json.loads((interval_dir / "timing.json").read_text())
    except Exception:
        return None
    value = timing.get(f"wall_max_ms_{precision}")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _monotonic_violation_child_intervals(models: List[str], precision: str) -> List[Path]:
    """
    Return direct child intervals whose wall-max sum is below their parent.

    A direct child pair for int_a_b is int_a_c + int_(c+1)_b for one split c.
    """
    violating_children: set[Path] = set()
    for model in models:
        by_bounds: dict[tuple[int, int], Path] = {}
        wall_max: dict[tuple[int, int], float] = {}
        for interval_dir in interval_engine_dirs([model], precision):
            bounds = _parse_interval_bounds(interval_dir)
            value = _interval_wall_max(interval_dir, precision)
            if bounds is None or value is None:
                continue
            by_bounds[bounds] = interval_dir
            wall_max[bounds] = value

        for (start, end), parent_dir in by_bounds.items():
            if start >= end:
                continue
            parent_time = wall_max[(start, end)]
            for split in range(start, end):
                left = (start, split)
                right = (split + 1, end)
                if left not in by_bounds or right not in by_bounds:
                    continue
                child_total = wall_max[left] + wall_max[right]
                if child_total + MONOTONIC_EPSILON_MS < parent_time:
                    violating_children.add(by_bounds[left])
                    violating_children.add(by_bounds[right])
    return sorted(violating_children)


def _monotonic_violation_rows(models: List[str], precision: str) -> List[dict]:
    rows: list[dict] = []
    for model in models:
        by_bounds: dict[tuple[int, int], Path] = {}
        wall_max: dict[tuple[int, int], float] = {}
        for interval_dir in interval_engine_dirs([model], precision):
            bounds = _parse_interval_bounds(interval_dir)
            value = _interval_wall_max(interval_dir, precision)
            if bounds is None or value is None:
                continue
            by_bounds[bounds] = interval_dir
            wall_max[bounds] = value

        for (start, end), parent_dir in by_bounds.items():
            if start >= end:
                continue
            parent_time = wall_max[(start, end)]
            for split in range(start, end):
                left = (start, split)
                right = (split + 1, end)
                if left not in by_bounds or right not in by_bounds:
                    continue
                child_total = wall_max[left] + wall_max[right]
                if child_total + MONOTONIC_EPSILON_MS < parent_time:
                    rows.append(
                        {
                            "model": model,
                            "precision": precision,
                            "parent": str(parent_dir.relative_to(REPO)),
                            "parent_wall_max_ms": parent_time,
                            "left_child": str(by_bounds[left].relative_to(REPO)),
                            "left_wall_max_ms": wall_max[left],
                            "right_child": str(by_bounds[right].relative_to(REPO)),
                            "right_wall_max_ms": wall_max[right],
                            "child_total_wall_max_ms": child_total,
                            "gap_ms": parent_time - child_total,
                        }
                    )
    rows.sort(key=lambda row: (row["model"], -row["gap_ms"], row["parent"], row["left_child"]))
    return rows


def _inflate_monotonic_children_top_down(
    args: argparse.Namespace,
    models: List[str],
    precision: str,
) -> None:
    total_updates = 0
    total_constraints = 0
    for model in models:
        by_bounds, wall_max = _load_model_interval_wall_maxes(model, precision)
        if not by_bounds:
            continue
        update_reasons: dict[tuple[int, int], list[dict]] = {}
        ordered_parents = sorted(
            by_bounds,
            key=lambda bounds: (-(bounds[1] - bounds[0] + 1), bounds[0], bounds[1]),
        )
        for parent in ordered_parents:
            start, end = parent
            if start >= end or parent not in wall_max:
                continue
            parent_time = wall_max[parent]
            child_lower_bounds: dict[tuple[int, int], float] = {}
            child_reasons: dict[tuple[int, int], list[dict]] = {}
            for split in range(start, end):
                left = (start, split)
                right = (split + 1, end)
                if left not in wall_max or right not in wall_max:
                    continue
                total_constraints += 1
                left_time = wall_max[left]
                right_time = wall_max[right]
                child_total = left_time + right_time
                if child_total + MONOTONIC_EPSILON_MS >= parent_time:
                    continue
                if child_total > 0.0:
                    left_target = parent_time * (left_time / child_total)
                    right_target = parent_time * (right_time / child_total)
                else:
                    left_target = parent_time / 2.0
                    right_target = parent_time / 2.0
                for child, current, target in (
                    (left, left_time, left_target),
                    (right, right_time, right_target),
                ):
                    child_lower_bounds[child] = max(child_lower_bounds.get(child, current), target)
                    child_reasons.setdefault(child, []).append(
                        {
                            "parent": f"int_{start}_{end}",
                            "split": split,
                            "parent_wall_max_ms": parent_time,
                            "sibling_pair_before_ms": child_total,
                            "current_wall_max_ms": current,
                            "ratio_target_wall_max_ms": target,
                        }
                    )
            for child, target in child_lower_bounds.items():
                current = wall_max[child]
                if target <= current:
                    continue
                wall_max[child] = target
                update_reasons.setdefault(child, []).extend(child_reasons.get(child, []))
                total_updates += 1
        remaining = _monotonic_violation_rows_from_maps(model, precision, by_bounds, wall_max)
        if remaining:
            raise RuntimeError(
                f"{precision}: top-down ratio inflation left {len(remaining)} "
                f"monotonic violations for {model}"
            )
        if not args.dry_run:
            _write_ratio_inflated_wall_maxes(
                by_bounds,
                wall_max,
                update_reasons,
                precision,
            )
    print(
        f"{precision}: ratio inflation constraints={total_constraints} "
        f"updated_intervals={total_updates}",
        flush=True,
    )
    if args.dry_run:
        print(
            f"{precision}: simulated remaining monotonic violations=0 "
            f"(dry-run; no timing.json files changed)",
            flush=True,
        )
    else:
        _write_monotonic_violation_report(args, models, precision)


def _load_model_interval_wall_maxes(
    model: str,
    precision: str,
) -> tuple[dict[tuple[int, int], Path], dict[tuple[int, int], float]]:
    by_bounds: dict[tuple[int, int], Path] = {}
    wall_max: dict[tuple[int, int], float] = {}
    for interval_dir in interval_engine_dirs([model], precision):
        bounds = _parse_interval_bounds(interval_dir)
        value = _interval_wall_max(interval_dir, precision)
        if bounds is None or value is None:
            continue
        by_bounds[bounds] = interval_dir
        wall_max[bounds] = value
    return by_bounds, wall_max


def _monotonic_violation_rows_from_maps(
    model: str,
    precision: str,
    by_bounds: dict[tuple[int, int], Path],
    wall_max: dict[tuple[int, int], float],
) -> List[dict]:
    rows: list[dict] = []
    for (start, end), parent_dir in by_bounds.items():
        if start >= end or (start, end) not in wall_max:
            continue
        parent_time = wall_max[(start, end)]
        for split in range(start, end):
            left = (start, split)
            right = (split + 1, end)
            if left not in wall_max or right not in wall_max:
                continue
            child_total = wall_max[left] + wall_max[right]
            if child_total + MONOTONIC_EPSILON_MS < parent_time:
                rows.append(
                    {
                        "model": model,
                        "precision": precision,
                        "parent": str(parent_dir.relative_to(REPO)),
                        "parent_wall_max_ms": parent_time,
                        "left_child": str(by_bounds[left].relative_to(REPO)),
                        "left_wall_max_ms": wall_max[left],
                        "right_child": str(by_bounds[right].relative_to(REPO)),
                        "right_wall_max_ms": wall_max[right],
                        "child_total_wall_max_ms": child_total,
                        "gap_ms": parent_time - child_total,
                    }
                )
    return rows


def _write_ratio_inflated_wall_maxes(
    by_bounds: dict[tuple[int, int], Path],
    wall_max: dict[tuple[int, int], float],
    update_reasons: dict[tuple[int, int], list[dict]],
    precision: str,
) -> None:
    for bounds, reasons in update_reasons.items():
        timing_path = by_bounds[bounds] / "timing.json"
        timing = _load_timing(timing_path)
        old_wall = timing.get(f"wall_max_ms_{precision}")
        new_wall = wall_max[bounds]
        if timing.get(f"measured_wall_max_ms_{precision}") is None:
            timing[f"measured_wall_max_ms_{precision}"] = old_wall
        timing[f"wall_max_ms_{precision}"] = new_wall
        timing[f"gpu_max_ms_{precision}"] = new_wall
        timing[f"wall_max_ratio_inflated_{precision}"] = True
        timing[f"wall_max_ratio_inflation_reasons_{precision}"] = reasons
        timing_path.write_text(json.dumps(timing, indent=2))


def _write_monotonic_violation_report(
    args: argparse.Namespace,
    models: List[str],
    precision: str,
) -> None:
    rows = _monotonic_violation_rows(models, precision)
    path = Path(args.monotonic_violation_report)
    if len(args.precisions) > 1:
        path = path.with_name(f"{path.stem}_{precision}{path.suffix}")
    payload = {
        "precision": precision,
        "models": models,
        "violation_count": len(rows),
        "violations": rows,
        "child_intervals": sorted(
            {
                row["left_child"]
                for row in rows
            }
            | {
                row["right_child"]
                for row in rows
            }
        ),
    }
    if not args.dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        txt = path.with_suffix(".txt")
        txt.write_text("\n".join(payload["child_intervals"]) + ("\n" if payload["child_intervals"] else ""))
    print(
        f"{precision}: remaining monotonic violations={len(rows)} "
        f"report={path}",
        flush=True,
    )


def _load_violation_child_intervals(
    report_path: Path,
    models: List[str],
    precision: str,
) -> List[Path]:
    data = json.loads(report_path.read_text())
    allowed_models = set(models)
    intervals: list[Path] = []
    for rel in data.get("child_intervals", []):
        path = REPO / str(rel)
        if path.parent.name not in allowed_models:
            continue
        if not (path / f"chunk_{precision}.engine").exists():
            continue
        intervals.append(path)
    return sorted(set(intervals))


def _run_table4_quiet(mod, config_path: Path, precision: str, warmup: int, iters: int):
    cmd = [
        str(mod.TABLE4_RUNNER),
        "--config",
        str(config_path),
        "--repo",
        str(REPO),
        "--precision",
        precision,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--sync-wall",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        print(detail, file=sys.stderr)
        return None, wall
    out = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    out_path = Path(out) if out else None
    return (out_path if out_path and out_path.exists() else None), wall


def eval_paths(models: List[str], precision: str) -> List[Path]:
    out: List[Path] = []
    for model in models:
        root = REPO / "results" / "evaluations" / model
        if root.exists():
            out.extend(sorted(root.glob(f"*_{precision}.json")))
    return [p for p in out if not p.stem.endswith("_cpp_raw")]


def rewrite_eval_caches(args: argparse.Namespace, models: List[str], precision: str) -> tuple[int, int]:
    paths = eval_paths(models, precision)
    def rewrite_one(path: Path) -> str:
        try:
            data = json.loads(path.read_text())
        except Exception:
            return "skipped"
        if data.get("timing_basis") == "cpu_wall_sync" and not args.force_eval_caches:
            return "already_done"
        mask = data.get("mask")
        model = data.get("model_name")
        if not isinstance(mask, list) or not model:
            return "skipped"
        groups = data.get("groups") or _compute_merge_groups(mask)
        means: List[float] = []
        p99s: List[float] = []
        maxs: List[float] = []
        ok = True
        for grp in groups:
            timing = _load_interval_timing(model, grp)
            vals = (
                timing.get(f"wall_mean_ms_{precision}"),
                timing.get(f"wall_p99_ms_{precision}"),
                timing.get(f"wall_max_ms_{precision}"),
            )
            if any(v is None for v in vals):
                return "skipped"
            means.append(float(vals[0]))
            p99s.append(float(vals[1]))
            maxs.append(float(vals[2]))
        data.update(
            {
                "per_chunk_gpu_mean_ms": means,
                "per_chunk_gpu_p99_ms": p99s,
                "per_chunk_gpu_max_ms": maxs,
                "per_chunk_cpu_wall_mean_ms": means,
                "per_chunk_cpu_wall_p99_ms": p99s,
                "per_chunk_cpu_wall_max_ms": maxs,
                "chunked_gpu_mean_ms": sum(means),
                "chunked_gpu_p99_ms": sum(p99s),
                "chunked_gpu_max_ms": sum(maxs),
                "timing_basis": "cpu_wall_sync",
            }
        )
        if not args.dry_run:
            path.write_text(json.dumps(data, indent=2))
        return "rewritten"

    rewritten = skipped = already_done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = pool.map(rewrite_one, paths)
        for status in tqdm_wrap(results, desc=f"eval caches {precision}", total=len(paths)):
            if status == "rewritten":
                rewritten += 1
            elif status == "already_done":
                already_done += 1
            else:
                skipped += 1
    if already_done:
        print(f"{precision}: eval caches already_done={already_done}")
    return rewritten, skipped


def rebuild_profiling_db(args: argparse.Namespace, models: List[str], precisions: List[str]) -> int:
    from src.optimization.profiling_db import ProfilingDB

    db_path = REPO / "results" / "optimization" / ".profiling_cache.json"
    db = ProfilingDB(db_path)
    count = 0
    paths = [(precision, path) for precision in precisions for path in eval_paths(models, precision)]
    for precision, path in tqdm_wrap(paths, desc="profiling db", total=len(paths)):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if data.get("timing_basis") != "cpu_wall_sync":
            continue
        model = data.get("model_name")
        variant = data.get("variant_name")
        if not model or not variant:
            continue
        if not args.dry_run:
            key = db.make_key(model, variant, precision)
            db._data[key] = {
                "model": model,
                "variant": variant,
                "precision": precision,
                "full_gpu_mean_ms": data.get("full_gpu_mean_ms"),
                "full_gpu_max_ms": data.get("full_gpu_max_ms"),
                "per_chunk_gpu_mean_ms": data.get("per_chunk_gpu_mean_ms"),
                "per_chunk_gpu_p99_ms": data.get("per_chunk_gpu_p99_ms"),
                "per_chunk_gpu_max_ms": data.get("per_chunk_gpu_max_ms"),
                "total_chunked_gpu_mean_ms": data.get("chunked_gpu_mean_ms"),
                "total_chunked_gpu_max_ms": data.get("chunked_gpu_max_ms"),
                "source_json": str(path),
                "timestamp": "walltime_remeasure",
            }
        count += 1
    if not args.dry_run:
        db._flush()
    return count


def refresh_k_cache(args: argparse.Namespace, models: List[str], precisions: List[str]) -> tuple[int, int]:
    path = REPO / "results" / "optimization" / "measured_k_split_cache.json"
    if not path.exists():
        return 0, 0
    data = json.loads(path.read_text())
    entries = data.get("entries", {})
    refreshed = 0
    skipped = 0
    ordered = sorted(entries.items(), key=lambda item: (item[1].get("model_name", ""), item[1].get("precision", ""), item[1].get("policy_name", ""), item[1].get("k", 0)))
    for key, entry in tqdm_wrap(ordered, desc="apply-k cache", total=len(ordered)):
        mask = entry.get("mask")
        model = entry.get("model_name")
        precision = entry.get("precision")
        if model not in models or precision not in precisions:
            continue
        if not isinstance(mask, list) or not model or not precision:
            skipped += 1
            continue
        variant = mask_to_variant_name(model, mask)
        eval_path = _eval_json_path(model, variant, precision)
        if not eval_path.exists():
            skipped += 1
            continue
        eval_data = json.loads(eval_path.read_text())
        times = eval_data.get("per_chunk_gpu_max_ms")
        if not times:
            skipped += 1
            continue
        score = _measured_evenness_score(list(times))
        if args.dry_run:
            refreshed += 1
            continue
        # Let the normal cache writer re-apply monotonic envelope semantics.
        _store_cached_k_split_mask(
            key,
            model_name=model,
            precision=precision,
            wcet_metric=entry.get("wcet_metric", "max"),
            policy_name=entry.get("policy_name", "all"),
            boundary_count=entry["boundary_count"],
            enabled_boundaries=entry["enabled_boundaries"],
            k=entry["k"],
            mask=list(mask),
            score=score,
            selected_chunk_times=list(times),
            variant_name=variant,
            profile_result_path=str(eval_path.relative_to(REPO)),
        )
        refreshed += 1
    return refreshed, skipped


def ensure_optimistic_k_cache(args: argparse.Namespace, models: List[str], precisions: List[str]) -> tuple[int, int]:
    from src.integration.mask_applicator import _k_split_cache_key
    from src.integration.split_point_policy import get_enabled_boundaries

    created = 0
    skipped = 0
    jobs = [(model, precision) for model in models for precision in precisions]
    for model, precision in tqdm_wrap(jobs, desc="optimistic endpoint cache", total=len(jobs)):
        if model == "vit_l_16":
            skipped += 1
            continue
        cfg_path = REPO / "artifacts" / "split_configs" / model / "dag_aligned_full.json"
        if not cfg_path.exists():
            skipped += 1
            continue
        cfg = json.loads(cfg_path.read_text())
        boundary_count = len(cfg.get("chunks", [])) - 1
        if boundary_count < 0:
            skipped += 1
            continue
        enabled = get_enabled_boundaries(model, "trt_fusion_safe", boundary_count)
        if not enabled:
            skipped += 1
            continue
        endpoint_masks = [
            (1, [0] * boundary_count),
            (len(enabled) + 1, [1 if i in set(enabled) else 0 for i in range(boundary_count)]),
        ]
        for k, mask in endpoint_masks:
            if not can_assemble_from_intervals(model, mask, precision):
                skipped += 1
                continue
            variant = mask_to_variant_name(model, mask)
            eval_path = _eval_json_path(model, variant, precision)
            if eval_path.exists():
                eval_data = json.loads(eval_path.read_text())
            else:
                eval_result = assemble_from_intervals(model, mask, precision)
                eval_data = eval_result.to_dict()
            times = eval_data.get("per_chunk_gpu_max_ms")
            if not times:
                skipped += 1
                continue
            cache_key = _k_split_cache_key(
                model_name=model,
                precision=precision,
                wcet_metric="max",
                policy_name="trt_fusion_safe",
                boundary_count=boundary_count,
                enabled_boundaries=enabled,
                k=k,
            )
            if not args.dry_run:
                _store_cached_k_split_mask(
                    cache_key,
                    model_name=model,
                    precision=precision,
                    wcet_metric="max",
                    policy_name="trt_fusion_safe",
                    boundary_count=boundary_count,
                    enabled_boundaries=enabled,
                    k=k,
                    mask=mask,
                    score=_measured_evenness_score(list(times)),
                    selected_chunk_times=list(times),
                    variant_name=variant,
                    profile_result_path=str(eval_path.relative_to(REPO)),
                )
            created += 1
    return created, skipped


def main() -> int:
    args = parse_args()
    if args.reprofile_monotonic_violations and args.inflate_monotonic_children_by_parent_ratio:
        raise SystemExit(
            "choose only one monotonic repair mode: "
            "--reprofile-monotonic-violations or "
            "--inflate-monotonic-children-by-parent-ratio"
        )
    models = discover_models(args.models)
    if not models:
        print("No models found.")
        return 0
    for precision in args.precisions:
        if not args.skip_intervals:
            remeasure_intervals(args, models, precision)
        if args.inflate_monotonic_children_by_parent_ratio:
            _inflate_monotonic_children_top_down(args, models, precision)
        if not args.skip_eval_caches:
            rewritten, skipped = rewrite_eval_caches(args, models, precision)
            print(f"{precision}: eval caches rewritten={rewritten} skipped={skipped}")
    db_count = 0 if args.skip_profiling_db else rebuild_profiling_db(args, models, args.precisions)
    print(f"profiling DB refreshed={db_count}{' (skipped)' if args.skip_profiling_db else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
