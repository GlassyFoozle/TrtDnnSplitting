#!/usr/bin/env python3
"""
Re-run split-heavy SS-tol-fb tasksets and plot per-taskset bar charts.

For each supplied source result directory, this script:
  1. reads per_taskset_results.csv,
  2. filters SS-tol-fb rows at selected utilizations (default: 0.80, 0.90),
  3. selects top-N tasksets per source directory and utilization by k_split_calls,
  4. re-runs SS-tol-fb and SS-tol-fb-off, optionally with per-split overhead,
  5. writes selected_tasksets.csv, comparison_results.csv, and plots.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = REPO / "results" / "dnn_experiments"
DEFAULT_METRICS = [
    "k_split_calls",
    "k_split_unique_model_chunks",
    "k_split_unique_model_masks",
    "k_split_unique_task_chunks",
    "k_split_unique_task_masks",
]

METRIC_LABELS = {
    "k_split_calls": "Split function calls (k_split_calls)",
    "k_split_unique_model_chunks": "Model-level chunk types (k_split_unique_model_chunks)",
    "k_split_unique_model_masks": "Model-level mask types (k_split_unique_model_masks)",
    "k_split_unique_task_chunks": "Task-level chunk types (k_split_unique_task_chunks)",
    "k_split_unique_task_masks": "Task-level mask types (k_split_unique_task_masks)",
}

ALGORITHM_ORDER = ["SS-tol-fb", "SS-tol-fb-off"]
ALGORITHM_COLORS = {
    "SS-tol-fb": "#7A3FB2",
    "SS-tol-fb-off": "#2F6FB3",
}


def _load_fig4_helpers():
    path = REPO / "scripts" / "internal_fig4_helpers.py"
    spec = importlib.util.spec_from_file_location("_fig4_helpers_for_top_compare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FIG4 = _load_fig4_helpers()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Select top split-heavy SS-tol-fb tasksets from one or more Fig.4 "
            "result directories, re-run tol-fb on/off, and plot taskset boxplots."
        )
    )
    ap.add_argument(
        "--run-dirs",
        nargs="+",
        default=[],
        help="Source result directories containing per_taskset_results.csv.",
    )
    # Backward-compatible aliases from the first version of this script.
    ap.add_argument("--fp32-run-dir", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--fp16-run-dir", default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--utilizations",
        nargs="+",
        type=float,
        default=[0.8, 0.9],
        help="Utilization values to select independently.",
    )
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument(
        "--rank-metric",
        default="k_split_calls",
        help="Column used to rank split-heavy rows within each source/utilization.",
    )
    ap.add_argument(
        "--precision-from",
        choices=["auto", "csv", "run-dir-name"],
        default="auto",
        help="How to infer precision for re-runs.",
    )
    ap.add_argument("--split-policy", default="trt_fusion_safe")
    ap.add_argument("--wcet-metric", default="max", choices=["max", "p99", "mean"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--max-candidates", type=int, default=1000000)
    ap.add_argument("--max-profiles", type=int, default=1000000)
    ap.add_argument("--max-iterations", type=int, default=10000)
    ap.add_argument("--live", action="store_true", help="Run live/cache-first profiling.")
    ap.add_argument("--force-profile", action="store_true", help="Force rebuild/reprofile.")
    ap.add_argument(
        "--per-split-overhead-ms",
        nargs="+",
        type=float,
        default=[0.0],
        help=(
            "Milliseconds to add at every active split boundary during RTA. "
            "Pass multiple values to compare overhead sensitivity."
        ),
    )
    ap.add_argument(
        "--allow-equal-wcet-fallback",
        action="store_true",
        help="Pass through to the DNN runner for development-only fallback timing.",
    )
    ap.add_argument("--run-name", default=None, help="Output run name under results/dnn_experiments.")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT_ROOT), help="Base output directory.")
    ap.add_argument("--plot-metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--plot-output-prefix", default="tolfb_on_off_top_split")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed runner logs instead of the compact progress bar.",
    )
    return ap.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def parse_int(value: object, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(str(value))
    except (TypeError, ValueError):
        return default


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def is_ss_tolfb_on(row: Dict[str, str]) -> bool:
    impl = (row.get("algorithm_impl") or "").strip().lower()
    label = (row.get("algorithm_label") or row.get("algorithm") or "").strip().lower()
    if impl:
        return impl == "ss:tol-fb"
    return label == "ss-tol-fb"


def infer_precision(run_dir: Path, rows: Sequence[Dict[str, str]], mode: str) -> str:
    if mode in ("auto", "csv"):
        values = sorted({(r.get("precision") or "").strip().lower() for r in rows if r.get("precision")})
        values = [v for v in values if v in ("fp32", "fp16")]
        if len(values) == 1:
            return values[0]
        if mode == "csv" and len(values) != 1:
            raise ValueError(f"cannot infer one precision from CSV in {run_dir}: {values}")
    name = run_dir.name.lower()
    if "fp16" in name:
        return "fp16"
    if "fp32" in name:
        return "fp32"
    cfg_path = run_dir / "run_config.json"
    if mode == "auto" and cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            precision = str(cfg.get("precision") or "").lower()
            if precision in ("fp32", "fp16"):
                return precision
        except Exception:
            pass
    raise ValueError(f"cannot infer precision for {run_dir}; include fp32/fp16 in run dir or CSV")


def util_matches(row: Dict[str, str], util: float) -> bool:
    return abs(parse_float(row.get("utilization"), default=-1.0) - util) < 1e-6


def select_top_split_rows(
    run_dir: Path,
    top_n: int,
    utilizations: Iterable[float],
    rank_metric: str,
    precision_mode: str,
) -> List[Dict[str, Any]]:
    csv_path = run_dir / "per_taskset_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    all_rows = read_rows(csv_path)
    precision = infer_precision(run_dir, all_rows, precision_mode)

    selected: List[Dict[str, Any]] = []
    for util in utilizations:
        rows = [r for r in all_rows if is_ss_tolfb_on(r) and util_matches(r, util)]
        rows.sort(
            key=lambda r: (
                parse_int(r.get(rank_metric)),
                parse_int(r.get("split_task_count")),
                parse_int(r.get("masks_evaluated")),
                parse_int(r.get("real_profiles")),
            ),
            reverse=True,
        )
        for rank, row in enumerate(rows[:top_n], start=1):
            taskset_path = resolve_path(row["taskset_path"])
            selected.append(
                {
                    "precision": precision,
                    "rank": rank,
                    "rank_metric": rank_metric,
                    "rank_metric_value": parse_int(row.get(rank_metric)),
                    "source_run_dir": rel(run_dir),
                    "source_utilization": parse_float(row.get("utilization")),
                    "source_taskset": row.get("taskset", taskset_path.stem),
                    "taskset_path": taskset_path,
                    "source_split_task_count": parse_int(row.get("split_task_count")),
                    "source_masks_evaluated": parse_int(row.get("masks_evaluated")),
                    "source_real_profiles": parse_int(row.get("real_profiles")),
                    "source_k_split_calls": parse_int(row.get("k_split_calls")),
                    "source_k_split_candidate_masks": parse_int(row.get("k_split_candidate_masks")),
                    "source_k_split_candidate_mask_profiles": parse_int(
                        row.get("k_split_candidate_mask_profiles")
                    ),
                }
            )
    return selected


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class ProgressBar:
    def __init__(self, total: int, enabled: bool = True, width: int = 34) -> None:
        self.total = max(0, int(total))
        self.enabled = bool(enabled and self.total > 0)
        self.width = width
        self.current = 0
        self.started = time.time()
        self._last_len = 0

    def update(self, label: str = "") -> None:
        if not self.enabled:
            return
        self.current = min(self.current + 1, self.total)
        ratio = self.current / self.total if self.total else 1.0
        filled = int(round(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.started
        msg = (
            f"\r[{bar}] {self.current}/{self.total} "
            f"({ratio * 100:5.1f}%) elapsed={elapsed:6.1f}s"
        )
        if label:
            msg += f"  {label}"
        pad = max(0, self._last_len - len(msg))
        sys.stderr.write(msg + (" " * pad))
        sys.stderr.flush()
        self._last_len = len(msg)

    def close(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


def run_one(
    taskset_path: Path,
    precision: str,
    algorithm: str,
    label: str,
    overhead_ms: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    from src.integration.dnn_algorithm_runner import run_dnn_rta_algorithm

    result = run_dnn_rta_algorithm(
        dnn_taskset_path=taskset_path,
        model="ss",
        algorithm=algorithm,
        precision=precision,
        wcet_metric=args.wcet_metric,
        use_cpp=True,
        force_profile=bool(args.force_profile),
        dry_run=not bool(args.live),
        max_iterations=args.max_iterations,
        warmup=args.warmup,
        iters=args.iters,
        policy_name=args.split_policy,
        max_profiles=args.max_profiles,
        max_candidates=args.max_candidates,
        allow_equal_wcet_fallback=args.allow_equal_wcet_fallback,
        per_splitting_overhead=float(overhead_ms),
    )
    row = _FIG4.summarize_result(
        None,
        taskset_path,
        "ss",
        algorithm,
        result,
        _FIG4.load_initial_masks(taskset_path),
    )
    row["algorithm_label"] = label
    row["algorithm"] = label
    row["algorithm_impl"] = f"ss:{algorithm}"
    row["per_split_overhead_ms"] = float(overhead_ms)
    return row


def run_one_quiet(
    taskset_path: Path,
    precision: str,
    algorithm: str,
    label: str,
    overhead_ms: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if args.verbose:
        return run_one(taskset_path, precision, algorithm, label, overhead_ms, args)

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            return run_one(taskset_path, precision, algorithm, label, overhead_ms, args)
    except Exception:
        text = captured.getvalue().strip()
        if text:
            print(text, file=sys.stderr)
        raise


def plot_taskset_boxplots(
    rows: List[Dict[str, Any]],
    metrics: Iterable[str],
    output_base: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in metrics if any(str(r.get(m, "")) not in ("", "None") for r in rows)]
    if not metrics:
        raise ValueError("no plottable metrics found")

    cases: List[str] = []
    for row in rows:
        case = str(row["case_label"])
        if case not in cases:
            cases.append(case)

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(max(10.0, 1.0 * len(cases)), 3.4 * len(metrics)),
        squeeze=False,
    )

    for ax, metric in zip(axes[:, 0], metrics):
        data = [
            [parse_float(r.get(metric)) for r in rows if r["case_label"] == case]
            for case in cases
        ]
        ax.boxplot(data, labels=cases, showmeans=True, patch_artist=True)
        ax.set_title(metric.replace("_", " "))
        ax.set_xticklabels(cases, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".boxplot.png")
    pdf = output_base.with_suffix(".boxplot.pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_algorithm_overhead_boxplots(
    rows: List[Dict[str, Any]],
    metrics: Iterable[str],
    output_base: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in metrics if any(str(r.get(m, "")) not in ("", "None") for r in rows)]
    labels: List[str] = []
    for row in rows:
        label = f"{row['algorithm_label']} / +{parse_float(row.get('per_split_overhead_ms')):g}ms"
        if label not in labels:
            labels.append(label)
    if not metrics or not labels:
        return

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(max(9.0, 1.25 * len(labels)), 3.4 * len(metrics)),
        squeeze=False,
    )
    for ax, metric in zip(axes[:, 0], metrics):
        data = [
            [
                parse_float(r.get(metric))
                for r in rows
                if f"{r['algorithm_label']} / +{parse_float(r.get('per_split_overhead_ms')):g}ms" == label
            ]
            for label in labels
        ]
        ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
        ax.set_title(metric.replace("_", " "))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    png = output_base.with_suffix(".by_algorithm_overhead.png")
    pdf = output_base.with_suffix(".by_algorithm_overhead.pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def _overhead_token(overhead_ms: float) -> str:
    text = f"{overhead_ms:g}".replace("-", "m").replace(".", "p")
    return f"{text}ms"


def _mean_metric(rows: List[Dict[str, Any]], metric: str) -> float:
    values = [parse_float(r.get(metric)) for r in rows]
    return sum(values) / len(values) if values else 0.0


def plot_taskset_bars_by_overhead(
    rows: List[Dict[str, Any]],
    metrics: Iterable[str],
    output_base: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in metrics if any(str(r.get(m, "")) not in ("", "None") for r in rows)]
    if not metrics:
        raise ValueError("no plottable metrics found")

    cases: List[str] = []
    for row in rows:
        case = str(row["case_label"])
        if case not in cases:
            cases.append(case)

    overheads = sorted({parse_float(r.get("per_split_overhead_ms")) for r in rows})
    present_algorithms: List[str] = []
    for preferred in ALGORITHM_ORDER:
        if any(str(r.get("algorithm_label")) == preferred for r in rows):
            present_algorithms.append(preferred)
    for row in rows:
        label = str(row.get("algorithm_label"))
        if label and label not in present_algorithms:
            present_algorithms.append(label)
    if not overheads or not present_algorithms:
        return

    output_base.parent.mkdir(parents=True, exist_ok=True)

    for overhead in overheads:
        fig, axes = plt.subplots(
            len(metrics),
            1,
            figsize=(max(12.0, 1.1 * len(cases)), 3.2 * len(metrics)),
            squeeze=False,
        )
        bar_width = min(0.38, 0.78 / max(1, len(present_algorithms)))
        start = -bar_width * (len(present_algorithms) - 1) / 2.0
        x_positions = list(range(len(cases)))

        for ax, metric in zip(axes[:, 0], metrics):
            for alg_idx, algorithm in enumerate(present_algorithms):
                offset = start + alg_idx * bar_width
                values = []
                for case in cases:
                    matching = [
                        r
                        for r in rows
                        if str(r.get("case_label")) == case
                        and str(r.get("algorithm_label")) == algorithm
                        and abs(parse_float(r.get("per_split_overhead_ms")) - overhead) < 1e-9
                    ]
                    values.append(_mean_metric(matching, metric))
                color = ALGORITHM_COLORS.get(algorithm)
                ax.bar(
                    [x + offset for x in x_positions],
                    values,
                    width=bar_width,
                    label=algorithm,
                    color=color,
                    alpha=0.86,
                )

            ax.set_title(_metric_label(metric))
            ax.set_ylabel("Count")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(cases, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)

        axes[0, 0].legend(loc="upper right")
        fig.suptitle(f"Per-task counters by algorithm (+{overhead:g} ms/split)", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        stem = f"{output_base.name}_overhead_{_overhead_token(overhead)}_bars"
        png = output_base.parent / f"{stem}.png"
        pdf = output_base.parent / f"{stem}.pdf"
        fig.savefig(png, dpi=dpi)
        fig.savefig(pdf)
        plt.close(fig)
        print(f"Saved: {png}")
        print(f"Saved: {pdf}")


def plot_unschedulable_bars_by_overhead(
    rows: List[Dict[str, Any]],
    output_base: Path,
    dpi: int,
) -> None:
    """Plot unschedulable counts separately from the split-counter bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases: List[str] = []
    for row in rows:
        case = str(row["case_label"])
        if case not in cases:
            cases.append(case)
    overheads = sorted({parse_float(r.get("per_split_overhead_ms")) for r in rows})
    algorithms = [a for a in ALGORITHM_ORDER if any(str(r.get("algorithm_label")) == a for r in rows)]
    if not cases or not overheads or not algorithms:
        return

    for overhead in overheads:
        fig, ax = plt.subplots(figsize=(max(12.0, 1.1 * len(cases)), 4.2))
        bar_width = min(0.38, 0.78 / max(1, len(algorithms)))
        start = -bar_width * (len(algorithms) - 1) / 2.0
        x_positions = list(range(len(cases)))
        for alg_idx, algorithm in enumerate(algorithms):
            values = []
            for case in cases:
                matching = [
                    r for r in rows
                    if str(r.get("case_label")) == case
                    and str(r.get("algorithm_label")) == algorithm
                    and abs(parse_float(r.get("per_split_overhead_ms")) - overhead) < 1e-9
                ]
                values.append(sum(1 for r in matching if not bool(r.get("schedulable"))))
            ax.bar(
                [x + start + alg_idx * bar_width for x in x_positions],
                values,
                width=bar_width,
                label=algorithm,
                color=ALGORITHM_COLORS.get(algorithm),
                alpha=0.86,
            )
        ax.set_title(f"Unschedulable tasksets (+{overhead:g} ms/split)")
        ax.set_ylabel("Count")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(cases, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right")
        fig.tight_layout()
        stem = f"{output_base.name}_overhead_{_overhead_token(overhead)}_unsched_bars"
        png = output_base.parent / f"{stem}.png"
        pdf = output_base.parent / f"{stem}.pdf"
        fig.savefig(png, dpi=dpi)
        fig.savefig(pdf)
        plt.close(fig)
        print(f"Saved: {png}")
        print(f"Saved: {pdf}")


def build_schedulability_ratio_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Summarize only primary SS-tol-fb rows; exclude fb-off by design."""
    grouped: Dict[tuple[str, float], List[Dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("algorithm_label")) != "SS-tol-fb":
            continue
        key = (str(row["case_label"]), parse_float(row.get("per_split_overhead_ms")))
        grouped.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for (case, overhead), group in grouped.items():
        total = len(group)
        sched = sum(1 for row in group if bool(row.get("schedulable")))
        out.append(
            {
                "case_label": case,
                "per_split_overhead_ms": overhead,
                "algorithm_label": "SS-tol-fb",
                "total_tasksets": total,
                "schedulable_count": sched,
                "unschedulable_count": total - sched,
                "schedulability_ratio": sched / total if total else 0.0,
            }
        )
    return sorted(out, key=lambda r: (parse_float(r["per_split_overhead_ms"]), str(r["case_label"])))


def plot_schedulability_ratio(
    rows: List[Dict[str, Any]],
    output_base: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    cases: List[str] = []
    for row in rows:
        case = str(row["case_label"])
        if case not in cases:
            cases.append(case)
    overheads = sorted({parse_float(r["per_split_overhead_ms"]) for r in rows})
    for overhead in overheads:
        selected = [r for r in rows if abs(parse_float(r["per_split_overhead_ms"]) - overhead) < 1e-9]
        by_case = {str(r["case_label"]): r for r in selected}
        values = [parse_float((by_case.get(case) or {}).get("schedulability_ratio")) for case in cases]
        fig, ax = plt.subplots(figsize=(max(12.0, 1.1 * len(cases)), 4.2))
        ax.bar(cases, values, color=ALGORITHM_COLORS["SS-tol-fb"], alpha=0.86)
        ax.set_title(f"Schedulability ratio, SS-tol-fb only (+{overhead:g} ms/split)")
        ax.set_ylabel("Schedulability ratio")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels(cases, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        fig.tight_layout()
        stem = f"{output_base.name}_overhead_{_overhead_token(overhead)}_sched_ratio"
        png = output_base.parent / f"{stem}.png"
        pdf = output_base.parent / f"{stem}.pdf"
        fig.savefig(png, dpi=dpi)
        fig.savefig(pdf)
        plt.close(fig)
        print(f"Saved: {png}")
        print(f"Saved: {pdf}")


def main() -> int:
    args = parse_args()
    source_dirs = [resolve_path(p) for p in args.run_dirs]
    if args.fp32_run_dir:
        source_dirs.append(resolve_path(args.fp32_run_dir))
    if args.fp16_run_dir:
        source_dirs.append(resolve_path(args.fp16_run_dir))
    if not source_dirs:
        print("[error] pass --run-dirs <result_dir> [...]", file=sys.stderr)
        return 2

    run_name = args.run_name or f"tolfb_top_split_boxplots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root = resolve_path(args.output_dir)
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    selected: List[Dict[str, Any]] = []
    for run_dir in source_dirs:
        selected.extend(
            select_top_split_rows(
                run_dir=run_dir,
                top_n=args.top_n,
                utilizations=args.utilizations,
                rank_metric=args.rank_metric,
                precision_mode=args.precision_from,
            )
        )

    selected_rows = [
        {**{k: v for k, v in item.items() if k != "taskset_path"}, "taskset_path": rel(item["taskset_path"])}
        for item in selected
    ]
    write_csv(out_dir / "selected_tasksets.csv", selected_rows)

    result_rows: List[Dict[str, Any]] = []
    total_runs = len(selected) * len(args.per_split_overhead_ms) * 2
    progress = ProgressBar(total_runs, enabled=not args.verbose)
    if args.verbose:
        print(f"Selected {len(selected)} tasksets; running {total_runs} experiments.", flush=True)
    for item in selected:
        base_case = (
            f"{item['precision'].upper()} {Path(item['source_run_dir']).name} "
            f"U={item['source_utilization']:.2f} #{item['rank']}"
        )
        if args.verbose:
            print(
                f"\n=== {base_case}: {item['taskset_path']} "
                f"({item['rank_metric']}={item['rank_metric_value']}) ===",
                flush=True,
            )
        for overhead_ms in args.per_split_overhead_ms:
            for algorithm, label in (("tol-fb", "SS-tol-fb"), ("tol-fb-off", "SS-tol-fb-off")):
                if args.verbose:
                    print(
                        f"  running {label} ({item['precision']}, overhead={overhead_ms:g}ms/split)",
                        flush=True,
                    )
                row = run_one_quiet(
                    item["taskset_path"],
                    item["precision"],
                    algorithm,
                    label,
                    overhead_ms,
                    args,
                )
                row.update(
                    {
                        "precision": item["precision"],
                        "source_rank": item["rank"],
                        "case_label": base_case,
                        "source_run_dir": item["source_run_dir"],
                        "source_utilization": item["source_utilization"],
                        "source_split_task_count": item["source_split_task_count"],
                        "source_masks_evaluated": item["source_masks_evaluated"],
                        "source_real_profiles": item["source_real_profiles"],
                        "source_k_split_calls": item["source_k_split_calls"],
                        "source_k_split_candidate_masks": item["source_k_split_candidate_masks"],
                        "source_k_split_candidate_mask_profiles": item[
                            "source_k_split_candidate_mask_profiles"
                        ],
                    }
                )
                result_rows.append(row)
                write_csv(out_dir / "comparison_results.csv", result_rows)
                progress.update(
                    f"{item['precision']} U={item['source_utilization']:.2f} "
                    f"#{item['rank']} {label} +{overhead_ms:g}ms"
                )
    progress.close()

    metadata = {
        "run_name": run_name,
        "live": bool(args.live),
        "split_policy": args.split_policy,
        "wcet_metric": args.wcet_metric,
        "top_n": args.top_n,
        "utilizations": args.utilizations,
        "rank_metric": args.rank_metric,
        "per_split_overhead_ms": args.per_split_overhead_ms,
        "sources": [rel(p) for p in source_dirs],
        "plot_metrics": args.plot_metrics,
    }
    (out_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))
    ratio_rows = build_schedulability_ratio_rows(result_rows)
    write_csv(out_dir / "schedulability_ratio.csv", ratio_rows)

    if not args.no_plot:
        plot_taskset_bars_by_overhead(
            result_rows,
            args.plot_metrics,
            out_dir / args.plot_output_prefix,
            args.dpi,
        )
        plot_unschedulable_bars_by_overhead(
            result_rows,
            out_dir / args.plot_output_prefix,
            args.dpi,
        )
        plot_schedulability_ratio(
            ratio_rows,
            out_dir / args.plot_output_prefix,
            args.dpi,
        )

    print(f"\nSaved: {out_dir / 'selected_tasksets.csv'}")
    print(f"Saved: {out_dir / 'comparison_results.csv'}")
    print(f"Saved: {out_dir / 'schedulability_ratio.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
