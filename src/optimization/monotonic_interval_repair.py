"""Top-down monotonic repair for cached interval wall-max timings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent.parent
MONOTONIC_EPSILON_MS = 1e-12


def inflate_children_by_parent_ratio(
    model_name: str,
    precision: str,
    *,
    seed_intervals: Iterable[tuple[int, int]] | None = None,
) -> dict:
    by_bounds, wall_max = _load_model_intervals(model_name, precision)
    if seed_intervals is None:
        parents = set(by_bounds)
    else:
        parents = {bounds for bounds in seed_intervals if bounds in by_bounds}
    changed: set[tuple[int, int]] = set()
    reasons: dict[tuple[int, int], list[dict]] = {}
    constraints = 0

    # Any child raised by a parent must itself be checked later, so walk by
    # descending length. Parents cannot be raised by shorter intervals.
    ordered = sorted(
        by_bounds,
        key=lambda bounds: (-(bounds[1] - bounds[0] + 1), bounds[0], bounds[1]),
    )
    active = set(parents)
    for parent in ordered:
        if parent not in active:
            continue
        start, end = parent
        if start >= end or parent not in wall_max:
            continue
        parent_time = wall_max[parent]
        lower_bounds: dict[tuple[int, int], float] = {}
        lower_reasons: dict[tuple[int, int], list[dict]] = {}
        for split in range(start, end):
            left = (start, split)
            right = (split + 1, end)
            if left not in wall_max or right not in wall_max:
                continue
            constraints += 1
            left_time = wall_max[left]
            right_time = wall_max[right]
            child_total = left_time + right_time
            if child_total + MONOTONIC_EPSILON_MS >= parent_time:
                continue
            if child_total > 0.0:
                targets = (
                    parent_time * (left_time / child_total),
                    parent_time * (right_time / child_total),
                )
            else:
                targets = (parent_time / 2.0, parent_time / 2.0)
            for child, current, target in (
                (left, left_time, targets[0]),
                (right, right_time, targets[1]),
            ):
                lower_bounds[child] = max(lower_bounds.get(child, current), target)
                lower_reasons.setdefault(child, []).append(
                    {
                        "parent": f"int_{start}_{end}",
                        "split": split,
                        "parent_wall_max_ms": parent_time,
                        "sibling_pair_before_ms": child_total,
                        "current_wall_max_ms": current,
                        "ratio_target_wall_max_ms": target,
                    }
                )
        for child, target in lower_bounds.items():
            if target <= wall_max[child]:
                continue
            wall_max[child] = target
            changed.add(child)
            reasons.setdefault(child, []).extend(lower_reasons.get(child, []))
            active.add(child)

    remaining = violation_rows(model_name, precision, by_bounds=by_bounds, wall_max=wall_max)
    if seed_intervals is None and remaining:
        raise RuntimeError(
            f"{precision}: top-down ratio inflation left {len(remaining)} "
            f"monotonic violations for {model_name}"
        )
    _write_updates(by_bounds, wall_max, reasons, precision)
    _record_inflation_event(model_name, precision, changed)
    return {
        "constraints": constraints,
        "updated_intervals": len(changed),
        "changed_bounds": sorted(changed),
        "remaining_violations": remaining,
    }


def violation_rows(
    model_name: str,
    precision: str,
    *,
    by_bounds: dict[tuple[int, int], Path] | None = None,
    wall_max: dict[tuple[int, int], float] | None = None,
) -> list[dict]:
    if by_bounds is None or wall_max is None:
        by_bounds, wall_max = _load_model_intervals(model_name, precision)
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
            if child_total + MONOTONIC_EPSILON_MS >= parent_time:
                continue
            rows.append(
                {
                    "model": model_name,
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


def _load_model_intervals(model_name: str, precision: str):
    root = REPO / "artifacts" / "chunk_cache" / model_name
    by_bounds: dict[tuple[int, int], Path] = {}
    wall_max: dict[tuple[int, int], float] = {}
    if not root.exists():
        return by_bounds, wall_max
    for path in root.iterdir():
        bounds = _parse_bounds(path)
        if bounds is None or not (path / f"chunk_{precision}.engine").exists():
            continue
        try:
            timing = json.loads((path / "timing.json").read_text())
            value = float(timing[f"wall_max_ms_{precision}"])
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        by_bounds[bounds] = path
        wall_max[bounds] = value
    return by_bounds, wall_max


def _parse_bounds(path: Path):
    parts = path.name.split("_")
    if len(parts) != 3 or parts[0] != "int":
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


def _write_updates(by_bounds, wall_max, reasons, precision: str) -> None:
    for bounds, child_reasons in reasons.items():
        path = by_bounds[bounds] / "timing.json"
        timing = json.loads(path.read_text())
        old_wall = timing.get(f"wall_max_ms_{precision}")
        if timing.get(f"measured_wall_max_ms_{precision}") is None:
            timing[f"measured_wall_max_ms_{precision}"] = old_wall
        timing[f"wall_max_ms_{precision}"] = wall_max[bounds]
        timing[f"gpu_max_ms_{precision}"] = wall_max[bounds]
        timing[f"wall_max_ratio_inflated_{precision}"] = True
        timing[f"wall_max_ratio_inflation_reasons_{precision}"] = child_reasons
        path.write_text(json.dumps(timing, indent=2))


def _record_inflation_event(
    model_name: str,
    precision: str,
    changed: set[tuple[int, int]],
) -> None:
    marker = os.environ.get("MONOTONIC_INFLATION_MARKER")
    if not marker or not changed:
        return
    path = Path(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(
            json.dumps(
                {
                    "model": model_name,
                    "precision": precision,
                    "updated_intervals": len(changed),
                    "changed_bounds": sorted(changed),
                }
            )
            + "\n"
        )
