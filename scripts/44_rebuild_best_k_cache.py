#!/usr/bin/env python3
"""Rebuild best-K cache from existing eval masks using current interval wall-max."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.integration.mask_applicator import (
    _k_chunk_candidate_masks,
    _k_split_cache_key,
    _measured_evenness_score,
    _store_cached_k_split_mask,
)
from src.optimization.config_evaluator import mask_to_variant_name
from src.integration.split_point_policy import get_enabled_boundaries
from src.optimization.config_evaluator import evaluate_mask
from src.optimization.monotonic_interval_repair import inflate_children_by_parent_ratio


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--policy", default="trt_fusion_safe")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--max-k-curve-repair-rounds", type=int, default=20)
    ap.add_argument(
        "--no-profile-missing",
        action="store_true",
        help="Do not live-profile masks whose required interval timings are missing.",
    )
    return ap.parse_args()


def base_boundary_count(model: str) -> int:
    cfg = json.loads(
        (REPO / "artifacts" / "split_configs" / model / "dag_aligned_full.json").read_text()
    )
    return len(cfg["chunks"]) - 1


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


def policy_mask_count(enabled: list[int]) -> int:
    return sum(comb(len(enabled), cuts) for cuts in range(len(enabled) + 1))


def progress(iterable, *, total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        def _iter():
            for idx, item in enumerate(iterable, 1):
                if idx == 1 or idx == total or idx % 1000 == 0:
                    print(f"{desc}: {idx}/{total}", flush=True)
                yield item

        return _iter()


def load_interval_wall_maxes(model: str, precision: str) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    root = REPO / "artifacts" / "chunk_cache" / model
    for path in root.glob("int_*_*/timing.json"):
        try:
            _, start, end = path.parent.name.split("_")
            timing = json.loads(path.read_text())
            values[(int(start), int(end))] = float(timing[f"wall_max_ms_{precision}"])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return values


def current_interval_times(
    interval_wall_maxes: dict[tuple[int, int], float],
    mask: list[int],
) -> list[float] | None:
    times: list[float] = []
    for group in mask_groups(mask):
        try:
            times.append(interval_wall_maxes[(group[0], group[-1])])
        except KeyError:
            return None
    return times


def all_policy_candidates(
    model: str,
    precision: str,
    enabled: list[int],
    boundary_count: int,
):
    interval_wall_maxes = load_interval_wall_maxes(model, precision)
    for k in range(1, len(enabled) + 2):
        for mask in _k_chunk_candidate_masks(boundary_count, k, enabled):
            times = current_interval_times(interval_wall_maxes, mask)
            if not times:
                continue
            variant_name = mask_to_variant_name(model, mask)
            eval_path = REPO / "results" / "evaluations" / model / f"{variant_name}_{precision}.json"
            yield {
                "mask": mask,
                "times": times,
                "score": _measured_evenness_score(times),
                "variant_name": variant_name,
                "profile_result_path": (
                    str(eval_path.relative_to(REPO)) if eval_path.exists() else ""
                ),
            }


def best_candidates_by_k(
    model: str,
    precision: str,
    enabled: list[int],
    boundary_count: int,
) -> tuple[dict[int, dict], int]:
    best_by_k: dict[int, dict] = {}
    scanned = 0
    total = policy_mask_count(enabled)
    candidates = all_policy_candidates(model, precision, enabled, boundary_count)
    for candidate in progress(
        candidates,
        total=total,
        desc=f"{model}/{precision} scan masks",
    ):
        scanned += 1
        k = sum(candidate["mask"]) + 1
        prev = best_by_k.get(k)
        if prev is None or candidate["score"] < prev["score"]:
            best_by_k[k] = candidate
    return best_by_k, scanned


def required_policy_intervals(
    enabled: list[int],
    boundary_count: int,
) -> set[tuple[int, int]]:
    """Return every contiguous interval any policy-valid mask may require."""
    cuts = [-1, *sorted(enabled), boundary_count]
    return {
        (left + 1, right)
        for left_idx, left in enumerate(cuts[:-1])
        for right in cuts[left_idx + 1 :]
    }


def mask_for_interval(
    interval: tuple[int, int],
    boundary_count: int,
) -> list[int]:
    """Build a minimal whole-model mask that contains one target interval."""
    start, end = interval
    mask = [0] * boundary_count
    if start > 0:
        mask[start - 1] = 1
    if end < boundary_count:
        mask[end] = 1
    return mask


def fill_missing_intervals(
    model: str,
    precision: str,
    enabled: list[int],
    boundary_count: int,
    *,
    warmup: int,
    iters: int,
) -> int:
    """Build/profile required missing intervals before the full mask scan."""
    profiled = 0
    required = required_policy_intervals(enabled, boundary_count)
    present = set(load_interval_wall_maxes(model, precision))
    missing = sorted(required - present, key=lambda bounds: (bounds[1] - bounds[0], bounds))
    print(
        f"[{model}] required intervals={len(required)} "
        f"present={len(required) - len(missing)} missing={len(missing)}",
        flush=True,
    )
    for interval in progress(
        missing,
        total=len(missing),
        desc=f"{model}/{precision} build missing intervals",
    ):
        if interval in load_interval_wall_maxes(model, precision):
            continue
        mask = mask_for_interval(interval, boundary_count)
        result = evaluate_mask(
            model,
            mask,
            variant_name=f"{model}_fill_int_{interval[0]}_{interval[1]}_{precision}",
            precision=precision,
            warmup=warmup,
            iters=iters,
            use_cpp=True,
            export=True,
            build=True,
            profile=True,
            force=False,
        )
        if not result.ok():
            raise RuntimeError(
                f"{model}/{precision}: failed to profile missing interval "
                f"int_{interval[0]}_{interval[1]} via mask={''.join(map(str, mask))}: "
                f"{result.error}"
            )
        profiled += 1
    # Re-run a full top-down pass after all new timings are present so the
    # following mask scan sees a complete adjusted interval envelope.
    inflate_children_by_parent_ratio(model, precision)
    return profiled


def curve_violations(best_by_k: dict[int, dict]) -> list[dict]:
    rows: list[dict] = []
    ordered = sorted(best_by_k.items())
    for (prev_k, prev), (cur_k, cur) in zip(ordered, ordered[1:]):
        prev_max, prev_total = prev["score"][:2]
        cur_max, cur_total = cur["score"][:2]
        max_up = cur_max > prev_max + 1e-12
        total_down = cur_total + 1e-12 < prev_total
        if max_up or total_down:
            rows.append(
                {
                    "prev_k": prev_k,
                    "cur_k": cur_k,
                    "prev": prev,
                    "cur": cur,
                    "max_up": max_up,
                    "total_down": total_down,
                }
            )
    return rows


def print_curve_state(model: str, precision: str, best_by_k: dict[int, dict], round_idx: int) -> None:
    print(f"[{model}] {precision} K-curve state round {round_idx}", flush=True)
    print("  K    max_block_ms    total_wcet_ms    mask", flush=True)
    for k, candidate in sorted(best_by_k.items()):
        max_block, total_wcet = candidate["score"][:2]
        print(
            f"  {k:<4d} {max_block:>12.6f}    {total_wcet:>13.6f}    "
            f"{''.join(map(str, candidate['mask']))}",
            flush=True,
        )


def _load_reprofile_module():
    spec = importlib.util.spec_from_file_location(
        "reprofile_intervals",
        REPO / "scripts" / "27_reprofile_cached_intervals.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/27_reprofile_cached_intervals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _interval_dir(model: str, bounds: tuple[int, int]) -> Path:
    return REPO / "artifacts" / "chunk_cache" / model / f"int_{bounds[0]}_{bounds[1]}"


def reprofile_intervals(
    model: str,
    precision: str,
    intervals: set[tuple[int, int]],
    *,
    warmup: int,
    iters: int,
) -> None:
    mod = _load_reprofile_module()
    base_cfg = mod._load_base_config(model)
    ordered = sorted(intervals, key=lambda bounds: (bounds[1] - bounds[0], bounds))
    for bounds in progress(
        ordered,
        total=len(ordered),
        desc=f"{model}/{precision} reprofile broken curve intervals",
    ):
        interval_dir = _interval_dir(model, bounds)
        with tempfile.TemporaryDirectory(prefix="trtdnn_kcurve_") as td:
            cfg_path = Path(td) / f"{model}_{interval_dir.name}.json"
            cfg_path.write_text(
                json.dumps(mod._make_profile_config(model, interval_dir, precision, base_cfg), indent=2)
            )
            out_path, wall = mod._run_table4(cfg_path, precision, warmup, iters)
            if out_path is None:
                raise RuntimeError(f"failed interval profile: {model}/{interval_dir.name}/{precision}")
            mod._update_timing(
                interval_dir / "timing.json",
                model,
                interval_dir,
                precision,
                out_path,
                wall,
                warmup,
                iters,
            )


def repair_k_curve_by_reprofiling(
    model: str,
    precision: str,
    enabled: list[int],
    boundary_count: int,
    *,
    warmup: int,
    iters: int,
    max_rounds: int,
) -> tuple[dict[int, dict], int]:
    total_scanned = 0
    for round_idx in range(max_rounds + 1):
        best_by_k, scanned = best_candidates_by_k(model, precision, enabled, boundary_count)
        total_scanned = scanned
        print_curve_state(model, precision, best_by_k, round_idx)
        violations = curve_violations(best_by_k)
        if not violations:
            return best_by_k, total_scanned
        print(f"[{model}] K-curve violations round {round_idx}", flush=True)
        for row in violations:
            prev_score = row["prev"]["score"]
            cur_score = row["cur"]["score"]
            flags = []
            if row["max_up"]:
                flags.append("max_up")
            if row["total_down"]:
                flags.append("total_down")
            print(
                f"  K={row['prev_k']} -> K={row['cur_k']} "
                f"prev=(max={prev_score[0]:.6f}, total={prev_score[1]:.6f}) "
                f"cur=(max={cur_score[0]:.6f}, total={cur_score[1]:.6f}) "
                f"flags={','.join(flags)}",
                flush=True,
            )
        if round_idx >= max_rounds:
            detail = [
                {
                    "prev_k": row["prev_k"],
                    "cur_k": row["cur_k"],
                    "prev_score": row["prev"]["score"],
                    "cur_score": row["cur"]["score"],
                    "max_up": row["max_up"],
                    "total_down": row["total_down"],
                }
                for row in violations
            ]
            raise RuntimeError(
                f"{model}/{precision}: K-curve monotonicity violations remain "
                f"after {max_rounds} repair rounds: {detail}"
            )
        intervals: set[tuple[int, int]] = set()
        for row in violations:
            for candidate in (row["prev"], row["cur"]):
                intervals.update((group[0], group[-1]) for group in mask_groups(candidate["mask"]))
        print(
            f"[{model}] K-curve repair round {round_idx + 1}: "
            f"violations={len(violations)} intervals={len(intervals)}",
            flush=True,
        )
        reprofile_intervals(
            model,
            precision,
            intervals,
            warmup=warmup,
            iters=iters,
        )
        inflate_children_by_parent_ratio(model, precision)
    raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()
    for model in args.models:
        boundary_count = base_boundary_count(model)
        enabled = get_enabled_boundaries(model, args.policy, boundary_count)
        if not args.no_profile_missing:
            profiled = fill_missing_intervals(
                model,
                args.precision,
                enabled,
                boundary_count,
                warmup=args.warmup,
                iters=args.iters,
            )
            print(f"[{model}] live-profiled interval-fill masks={profiled}", flush=True)
        best_by_k, scanned = repair_k_curve_by_reprofiling(
            model,
            args.precision,
            enabled,
            boundary_count,
            warmup=args.warmup,
            iters=args.iters,
            max_rounds=args.max_k_curve_repair_rounds,
        )
        print(f"[{model}] scanned eval masks={scanned} best-K entries={len(best_by_k)}", flush=True)
        for k in sorted(best_by_k):
            best = best_by_k[k]
            key = _k_split_cache_key(
                model_name=model,
                precision=args.precision,
                wcet_metric="max",
                policy_name=args.policy,
                boundary_count=boundary_count,
                enabled_boundaries=enabled,
                k=k,
            )
            _store_cached_k_split_mask(
                key,
                model_name=model,
                precision=args.precision,
                wcet_metric="max",
                policy_name=args.policy,
                boundary_count=boundary_count,
                enabled_boundaries=enabled,
                k=k,
                mask=best["mask"],
                score=best["score"],
                selected_chunk_times=best["times"],
                variant_name=best["variant_name"],
                profile_result_path=best["profile_result_path"],
                enable_monotonic_k_split_cache=False,
                refresh_curve_artifacts=False,
            )
            print(
                f"  K={k} mask={''.join(map(str, best['mask']))} "
                f"max={best['score'][0]:.6f} total={best['score'][1]:.6f}",
                flush=True,
            )
    from src.integration.mask_applicator import _refresh_k_split_curve_artifacts_if_live_cache
    _refresh_k_split_curve_artifacts_if_live_cache(
        precision=args.precision,
        wcet_metric="max",
        policy_name=args.policy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
