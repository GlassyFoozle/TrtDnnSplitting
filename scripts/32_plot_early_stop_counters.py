#!/usr/bin/env python3
"""
32_plot_early_stop_counters.py — Plot SS-tol vs SS-tol-fb early-stop counters.

Reads per_taskset_results.csv from a script-30 run and writes:
  - early_stop_counters_boxplot.{png,pdf}
  - early_stop_counters_histogram.{png,pdf}

Default run:
  results/dnn_experiments/early_stop_ss_tol_vs_tolfb
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RUN_NAME = "early_stop_ss_tol_vs_tolfb"
DEFAULT_RUN_DIR = REPO / "results" / "dnn_experiments" / DEFAULT_RUN_NAME
FIGURE_HEIGHT_IN = 3.8
FIGURE_WIDTH_PER_METRIC_IN = 4.2

ALGORITHM_ORDER = ["SS-tol-fb-off", "SS-tol-fb"]
STYLE = {
    "SS-tol": {"color": "#2F6FB3", "label": "offload-tol"},
    "SS-tol-fb-off": {"color": "#2F6FB3", "label": "offload-tol-fb (early stop=off)"},
    "SS-tol-fb": {"color": "#8000FF", "label": "offload-tol-fb"},
}
DEFAULT_X_AXIS_LABEL = "Total utilization U"

DEFAULT_METRICS = [
    "k_split_calls",
    "k_split_candidate_mask_profiles",
    "k_split_candidate_masks",
]

METRIC_LABELS = {
    "masks_evaluated": "Mask evaluations",
    "real_profiles": "Actual new profiles",
    "cache_hits": "Cache hits",
    "k_split_calls": "K-split calls",
    "k_split_cache_hits": "K-split cache hits",
    "k_split_candidate_masks": "Candidate mask profiles (no cache)",
    "k_split_candidate_mask_profiles": "Candidate mask profiles",
    "k_split_candidate_mask_inference_runs": "Candidate mask inference runs",
    "k_split_candidate_chunk_profiles": "Candidate chunk profiles",
    "k_split_candidate_inference_runs": "Candidate inference runs",
    "early_stop_optimistic_checks": "Optimistic R checks",
    "early_stop_optimistic_deadline_misses": "Optimistic deadline misses",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot early-stop counter boxplots and histograms from per_taskset_results.csv"
    )
    ap.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="Run directory name under results/dnn_experiments",
    )
    ap.add_argument(
        "--run-dir",
        default=None,
        help="Explicit run directory containing per_taskset_results.csv",
    )
    ap.add_argument(
        "--csv",
        default=None,
        dest="csv_path",
        help="Explicit per_taskset_results.csv path",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the run directory.",
    )
    ap.add_argument(
        "--output-prefix",
        default="early_stop_counters",
        help="Output filename prefix",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help=(
            "Metric columns to plot. Default: k_split_calls, "
            "k_split_candidate_mask_profiles, and k_split_candidate_masks."
        ),
    )
    ap.add_argument(
        "--algorithms",
        nargs="+",
        default=ALGORITHM_ORDER,
        help="Algorithm labels to include, in plotting order.",
    )
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--hist-bins",
        type=int,
        default=24,
        help="Number of histogram bins per metric.",
    )
    ap.add_argument(
        "--max-utilization",
        type=float,
        default=0.9,
        help="Drop rows above this utilization before plotting. Default: 0.9.",
    )
    ap.add_argument(
        "--x-axis-label",
        default=None,
        help="Override x-axis label; otherwise inferred from run_config.json when available.",
    )
    return ap.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        path = Path(args.run_dir)
    elif args.csv_path:
        path = Path(args.csv_path).parent
    else:
        path = REPO / "results" / "dnn_experiments" / args.run_name
    return path if path.is_absolute() else REPO / path


def resolve_csv(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.csv_path:
        path = Path(args.csv_path)
        path = path if path.is_absolute() else REPO / path
    else:
        path = run_dir / "per_taskset_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def resolve_output_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.output_dir:
        path = Path(args.output_dir)
        return path if path.is_absolute() else REPO / path
    return run_dir


def infer_x_axis_label(run_dir: Path, override: str | None) -> str:
    if override:
        return override
    run_config = run_dir / "run_config.json"
    try:
        data = json.loads(run_config.read_text())
        mapped = data.get("mapped_values", {})
        if mapped.get("utilization_kind") == "dnn_gpu":
            return "GPU utilization U"
    except Exception:
        pass
    return DEFAULT_X_AXIS_LABEL


def load_rows(
    csv_path: Path,
    metrics: Sequence[str],
    algorithms: Sequence[str],
    max_utilization: float | None,
) -> List[dict]:
    allowed = set(algorithms)
    rows: List[dict] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [m for m in metrics if m not in (reader.fieldnames or [])]
        if missing:
            raise KeyError(
                f"{csv_path} does not contain metric columns: {', '.join(missing)}"
            )
        for raw in reader:
            algorithm = raw.get("algorithm_label") or raw.get("algorithm") or ""
            if algorithm not in allowed:
                continue
            try:
                util = float(raw["utilization"])
            except (KeyError, TypeError, ValueError):
                continue
            if max_utilization is not None and util > max_utilization + 1e-12:
                continue
            row = {"utilization": util, "algorithm": algorithm}
            for metric in metrics:
                row[metric] = parse_float(raw.get(metric, "0"))
            rows.append(row)
    if not rows:
        raise ValueError("No matching rows found. Check --algorithms and input CSV.")
    return rows


def parse_float(value: object) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm
        from matplotlib.ticker import MaxNLocator
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required. Run with the project/conda environment that has matplotlib."
        ) from exc

    font_family = "serif"
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    if "Times New Roman" in available_fonts:
        font_family = "Times New Roman"
    else:
        liberation = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
        if liberation.exists():
            fm.fontManager.addfont(str(liberation))
            font_family = fm.FontProperties(fname=str(liberation)).get_name()
    return plt, MaxNLocator, font_family


def figure_size(metric_count: int) -> tuple[float, float]:
    width = max(5.2, FIGURE_WIDTH_PER_METRIC_IN * max(1, metric_count))
    return width, FIGURE_HEIGHT_IN


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def should_symlog(values: Iterable[float]) -> bool:
    vals = [abs(v) for v in values if v is not None]
    positives = [v for v in vals if v > 0]
    if not positives:
        return False
    return max(positives) / max(min(positives), 1.0) >= 100.0


def plot_boxplots(
    rows: List[dict],
    metrics: Sequence[str],
    algorithms: Sequence[str],
    output_base: Path,
    dpi: int,
    x_axis_label: str,
) -> None:
    plt, MaxNLocator, font_family = load_matplotlib()
    utils = sorted({r["utilization"] for r in rows})

    with plt.rc_context({"font.family": font_family, "mathtext.fontset": "stix"}):
        fig, axes = plt.subplots(
            1, len(metrics), figsize=figure_size(len(metrics)), squeeze=False
        )
        axes_flat = list(axes[0])
        width = 0.28
        offsets = centered_offsets(len(algorithms), width)

        for ax, metric in zip(axes_flat, metrics):
            all_values: List[float] = []
            annotation_items = []
            for alg_idx, algorithm in enumerate(algorithms):
                color = STYLE.get(algorithm, {}).get("color", "#333333")
                grouped = [
                    [r[metric] for r in rows if r["algorithm"] == algorithm and r["utilization"] == util]
                    for util in utils
                ]
                all_values.extend(v for group in grouped for v in group)
                positions = [idx + offsets[alg_idx] for idx in range(len(utils))]
                bp = ax.boxplot(
                    grouped,
                    positions=positions,
                    widths=width * 0.82,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                for box in bp["boxes"]:
                    box.set(facecolor=color, alpha=0.34, edgecolor=color, linewidth=1.2)
                for key in ("whiskers", "caps", "medians"):
                    for artist in bp[key]:
                        artist.set(color=color, linewidth=1.2)
                medians = [median(group) for group in grouped]
                ax.plot(
                    positions,
                    medians,
                    color=color,
                    marker="o",
                    markersize=3.2,
                    linewidth=1.3,
                    label=STYLE.get(algorithm, {}).get("label", algorithm),
                )
                for pos, group in zip(positions, grouped):
                    if not group:
                        continue
                    annotation_items.append(
                        {
                            "x": pos,
                            "max": max(group),
                            "mean": sum(group) / len(group),
                            "color": color,
                            "text": (
                                f"{format_stat(sum(group) / len(group))}\n"
                                f"{format_stat(max(group))}"
                            ),
                        }
                    )

            ax.set_title(metric_label(metric), fontsize=12)
            ax.set_xticks(range(len(utils)))
            ax.set_xticklabels([f"{u:.2f}" for u in utils], fontsize=10)
            ax.tick_params(axis="y", labelsize=10)
            ax.grid(True, axis="y", linestyle=(0, (5, 4)), linewidth=0.9, alpha=0.65)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.set_ylim(bottom=0.0)
            annotate_box_stats(ax, annotation_items)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(algorithms),
            fontsize=9,
            frameon=True,
            framealpha=0.88,
        )
        fig.supxlabel(x_axis_label, fontsize=13)
        fig.supylabel("Counter value", fontsize=13)
        fig.tight_layout(pad=0.8, rect=(0.03, 0.02, 1.0, 0.94))
        save_both(fig, output_base.with_name(output_base.name + "_boxplot"), dpi)
        plt.close(fig)


def plot_histograms(
    rows: List[dict],
    metrics: Sequence[str],
    algorithms: Sequence[str],
    output_base: Path,
    dpi: int,
    bins: int,
) -> None:
    plt, MaxNLocator, font_family = load_matplotlib()

    with plt.rc_context({"font.family": font_family, "mathtext.fontset": "stix"}):
        fig, axes = plt.subplots(
            1, len(metrics), figsize=figure_size(len(metrics)), squeeze=False
        )
        axes_flat = list(axes[0])

        for ax, metric in zip(axes_flat, metrics):
            all_values: List[float] = [r[metric] for r in rows]
            common_bins = make_bins(all_values, bins)
            for algorithm in algorithms:
                values = [r[metric] for r in rows if r["algorithm"] == algorithm]
                color = STYLE.get(algorithm, {}).get("color", "#333333")
                ax.hist(
                    values,
                    bins=common_bins,
                    histtype="step",
                    linewidth=1.7,
                    color=color,
                    label=STYLE.get(algorithm, {}).get("label", algorithm),
                )
            ax.set_title(metric_label(metric), fontsize=12)
            ax.set_xlabel("Counter value", fontsize=10)
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(True, axis="y", linestyle=(0, (5, 4)), linewidth=0.9, alpha=0.65)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.ticklabel_format(axis="both", style="plain", useOffset=False)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(algorithms),
            fontsize=9,
            frameon=True,
            framealpha=0.88,
        )
        fig.supylabel("Taskset count", fontsize=13)
        fig.tight_layout(pad=0.8, rect=(0.03, 0.02, 1.0, 0.94))
        save_both(fig, output_base.with_name(output_base.name + "_histogram"), dpi)
        plt.close(fig)


def plot_histograms_by_utilization(
    rows: List[dict],
    metrics: Sequence[str],
    algorithms: Sequence[str],
    output_base: Path,
    dpi: int,
    bins: int,
) -> None:
    for util in sorted({r["utilization"] for r in rows}):
        util_rows = [r for r in rows if r["utilization"] == util]
        if not util_rows:
            continue
        util_tag = f"u{util:.2f}".replace(".", "p")
        util_base = output_base.with_name(output_base.name + f"_histogram_{util_tag}")
        plot_histogram_figure(
            util_rows,
            metrics,
            algorithms,
            util_base,
            dpi,
            bins,
            title_suffix=f"U={util:.2f}",
        )


def plot_histogram_figure(
    rows: List[dict],
    metrics: Sequence[str],
    algorithms: Sequence[str],
    output_base: Path,
    dpi: int,
    bins: int,
    title_suffix: str | None = None,
) -> None:
    plt, MaxNLocator, font_family = load_matplotlib()

    with plt.rc_context({"font.family": font_family, "mathtext.fontset": "stix"}):
        fig, axes = plt.subplots(
            1, len(metrics), figsize=figure_size(len(metrics)), squeeze=False
        )
        axes_flat = list(axes[0])

        for ax, metric in zip(axes_flat, metrics):
            all_values: List[float] = [r[metric] for r in rows]
            common_bins = make_bins(all_values, bins)
            for algorithm in algorithms:
                values = [r[metric] for r in rows if r["algorithm"] == algorithm]
                color = STYLE.get(algorithm, {}).get("color", "#333333")
                ax.hist(
                    values,
                    bins=common_bins,
                    histtype="step",
                    linewidth=1.7,
                    color=color,
                    label=STYLE.get(algorithm, {}).get("label", algorithm),
                )
            title = metric_label(metric)
            if title_suffix:
                title = f"{title} ({title_suffix})"
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Counter value", fontsize=10)
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(True, axis="y", linestyle=(0, (5, 4)), linewidth=0.9, alpha=0.65)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.ticklabel_format(axis="both", style="plain", useOffset=False)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(algorithms),
            fontsize=9,
            frameon=True,
            framealpha=0.88,
        )
        fig.supylabel("Taskset count", fontsize=13)
        fig.tight_layout(pad=0.8, rect=(0.03, 0.02, 1.0, 0.94))
        save_both(fig, output_base, dpi)
        plt.close(fig)


def centered_offsets(count: int, width: float) -> List[float]:
    center = (count - 1) / 2.0
    return [(idx - center) * width for idx in range(count)]


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def format_stat(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def annotate_box_stats(ax, items: Sequence[dict]) -> None:
    if not items:
        return
    y0, y1 = ax.get_ylim()
    if ax.get_yscale() == "log":
        multiplier = 1.16
        for item in items:
            y = max(item["max"], 1e-9) * multiplier
            ax.text(
                item["x"], y, item["text"],
                ha="center", va="bottom", fontsize=5.8,
                rotation=0, color=item["color"],
            )
        ax.set_ylim(top=max(y1, max(max(i["max"], 1e-9) * 3.2 for i in items)))
        return
    if ax.get_yscale() == "symlog":
        for item in items:
            y = max(item["max"], 1.0) * 1.18
            ax.text(
                item["x"], y, item["text"],
                ha="center", va="bottom", fontsize=5.8,
                rotation=0, color=item["color"],
            )
        ax.set_ylim(top=max(y1, max(max(i["max"], 1.0) * 8.0 for i in items)))
        return

    span = max(y1 - y0, 1.0)
    for item in items:
        y = item["max"] + 0.035 * span
        ax.text(
            item["x"], y, item["text"],
            ha="center", va="bottom", fontsize=5.8,
            rotation=0, color=item["color"],
        )
    ax.set_ylim(top=max(y1, max(i["max"] for i in items) + 0.28 * span))


def make_bins(values: Sequence[float], bin_count: int) -> Sequence[float] | int:
    if not values:
        return bin_count
    lo = min(values)
    hi = max(values)
    if lo == hi:
        return max(1, min(bin_count, 10))
    return bin_count


def save_both(fig, output_base: Path, dpi: int) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args)
    csv_path = resolve_csv(args, run_dir)
    out_dir = resolve_output_dir(args, run_dir)
    rows = load_rows(csv_path, args.metrics, args.algorithms, args.max_utilization)
    output_base = out_dir / args.output_prefix
    x_axis_label = infer_x_axis_label(run_dir, args.x_axis_label)

    plot_boxplots(rows, args.metrics, args.algorithms, output_base, args.dpi, x_axis_label)
    plot_histograms(rows, args.metrics, args.algorithms, output_base, args.dpi, args.hist_bins)
    plot_histograms_by_utilization(
        rows, args.metrics, args.algorithms, output_base, args.dpi, args.hist_bins
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
