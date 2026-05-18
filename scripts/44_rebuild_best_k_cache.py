#!/usr/bin/env python3
"""Rebuild best-K cache from existing eval masks using current interval wall-max."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.integration.mask_applicator import (
    _k_split_cache_key,
    _measured_evenness_score,
    _store_cached_k_split_mask,
)
from src.integration.split_point_policy import get_enabled_boundaries


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--policy", default="trt_fusion_safe")
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


def current_interval_times(model: str, precision: str, mask: list[int]) -> list[float] | None:
    times: list[float] = []
    for group in mask_groups(mask):
        path = (
            REPO
            / "artifacts"
            / "chunk_cache"
            / model
            / f"int_{group[0]}_{group[-1]}"
            / "timing.json"
        )
        try:
            timing = json.loads(path.read_text())
            times.append(float(timing[f"wall_max_ms_{precision}"]))
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
    return times


def existing_eval_candidates(model: str, precision: str, enabled: set[int], boundary_count: int):
    root = REPO / "results" / "evaluations" / model
    if not root.exists():
        return
    for path in root.glob(f"*_{precision}.json"):
        if path.stem.endswith("_cpp_raw"):
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        mask = data.get("mask")
        if not isinstance(mask, list) or len(mask) != boundary_count:
            continue
        mask = [int(bit) for bit in mask]
        if any(bit not in (0, 1) for bit in mask):
            continue
        if any(bit and idx not in enabled for idx, bit in enumerate(mask)):
            continue
        times = current_interval_times(model, precision, mask)
        if not times:
            continue
        yield {
            "mask": mask,
            "times": times,
            "score": _measured_evenness_score(times),
            "variant_name": str(data.get("variant_name") or path.stem[: -(len(precision) + 1)]),
            "profile_result_path": str(path.relative_to(REPO)),
        }


def main() -> int:
    args = parse_args()
    for model in args.models:
        boundary_count = base_boundary_count(model)
        enabled = get_enabled_boundaries(model, args.policy, boundary_count)
        enabled_set = set(enabled)
        best_by_k: dict[int, dict] = {}
        scanned = 0
        for candidate in existing_eval_candidates(model, args.precision, enabled_set, boundary_count):
            scanned += 1
            k = sum(candidate["mask"]) + 1
            prev = best_by_k.get(k)
            if prev is None or candidate["score"] < prev["score"]:
                best_by_k[k] = candidate
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
