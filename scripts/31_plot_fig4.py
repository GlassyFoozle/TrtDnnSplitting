#!/usr/bin/env python3
"""
66_plot_yaml_fig4.py — Plot YAML-driven Fig.4 schedulability curves.

Reads schedulability_ratio.csv from a script-65 run and writes PNG/PDF under
results/dnn_experiments/plots/.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
_PLOTS_DIR = REPO / "results" / "dnn_experiments" / "plots"

_ORDER = ["SS_ours", "UNI_ours", "UNI_Buttazzo", "SS_Buttazzo"]
_STYLE = {
    "SS_ours": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "UNI_ours": {"color": "#2ca02c", "marker": "s", "linestyle": "-"},
    "UNI_Buttazzo": {"color": "#d62728", "marker": "^", "linestyle": "--"},
    "SS_Buttazzo": {"color": "#ff7f0e", "marker": "D", "linestyle": "--"},
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot YAML Fig.4-style schedulability ratio curves"
    )
    ap.add_argument(
        "--run-dir",
        default=None,
        help="Run directory containing schedulability_ratio.csv",
    )
    ap.add_argument(
        "--csv",
        default=None,
        dest="csv_path",
        help="Explicit schedulability_ratio.csv path",
    )
    ap.add_argument("--output", default=None, help="Output basename without extension")
    ap.add_argument(
        "--output-dir",
        default=str(_PLOTS_DIR),
        help="Directory for PNG/PDF output",
    )
    ap.add_argument(
        "--title",
        default="Fig.4-style Schedulability Ratio",
        help="Plot title",
    )
    ap.add_argument("--dpi", type=int, default=150)
    return ap.parse_args()


def resolve_csv(args: argparse.Namespace) -> Path:
    if args.csv_path:
        path = Path(args.csv_path)
    elif args.run_dir:
        path = Path(args.run_dir) / "schedulability_ratio.csv"
    else:
        raise ValueError("Provide --run-dir or --csv")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_series(csv_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    series: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                util = float(row["utilization"])
                ratio = float(row["schedulability_ratio"])
            except (KeyError, TypeError, ValueError):
                continue
            algorithm = row.get("algorithm", "unknown")
            series[algorithm].append((util, ratio))
    return {k: sorted(v) for k, v in series.items()}


def plot_with_matplotlib(
    series: Dict[str, List[Tuple[float, float]]],
    title: str,
    output_base: Path,
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for label in _ORDER + sorted(k for k in series if k not in _ORDER):
        points = series.get(label, [])
        if not points:
            continue
        xs, ys = zip(*points)
        style = _STYLE.get(label, {"color": None, "marker": "o", "linestyle": "-"})
        ax.plot(
            xs, ys,
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=5,
        )

    ax.set_xlabel("Total U across CPUs")
    ax.set_ylabel("Schedulability Ratio")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_with_pillow(
    series: Dict[str, List[Tuple[float, float]]],
    title: str,
    output_base: Path,
) -> None:
    """Minimal fallback used only if matplotlib is unavailable."""
    from PIL import Image, ImageDraw

    width, height = 1000, 680
    margin_l, margin_r, margin_t, margin_b = 90, 40, 70, 90
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    all_x = [x for pts in series.values() for x, _ in pts]
    x_min = min(all_x) if all_x else 0.0
    x_max = max(all_x) if all_x else 1.0
    if x_min == x_max:
        x_max = x_min + 1.0

    def xp(x: float) -> int:
        return int(margin_l + (x - x_min) / (x_max - x_min) * (width - margin_l - margin_r))

    def yp(y: float) -> int:
        return int(height - margin_b - y * (height - margin_t - margin_b))

    # axes
    draw.line([(margin_l, margin_t), (margin_l, height - margin_b)], fill="black", width=2)
    draw.line([(margin_l, height - margin_b), (width - margin_r, height - margin_b)], fill="black", width=2)
    draw.text((margin_l, 25), title, fill="black")
    draw.text((width // 2 - 80, height - 35), "Total U across CPUs", fill="black")
    draw.text((10, height // 2), "Schedulability Ratio", fill="black")

    colors = {
        "SS_ours": (31, 119, 180),
        "UNI_ours": (44, 160, 44),
        "UNI_Buttazzo": (214, 39, 40),
        "SS_Buttazzo": (255, 127, 14),
    }
    legend_y = margin_t
    for label in _ORDER + sorted(k for k in series if k not in _ORDER):
        points = series.get(label, [])
        if not points:
            continue
        color = colors.get(label, (80, 80, 80))
        coords = [(xp(x), yp(y)) for x, y in points]
        if len(coords) >= 2:
            draw.line(coords, fill=color, width=3)
        for x, y in coords:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.rectangle((width - 230, legend_y + 4, width - 210, legend_y + 14), fill=color)
        draw.text((width - 200, legend_y), label, fill="black")
        legend_y += 24

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    img.save(png)
    img.save(pdf, "PDF")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main() -> int:
    args = parse_args()
    csv_path = resolve_csv(args)
    series = load_series(csv_path)
    if not series:
        print(f"[error] no series found in {csv_path}", file=sys.stderr)
        return 1
    output_name = args.output
    if not output_name:
        if args.run_dir:
            output_name = Path(args.run_dir).name
        else:
            output_name = csv_path.parent.name
    output_base = Path(args.output_dir) / output_name

    try:
        plot_with_matplotlib(series, args.title, output_base, args.dpi)
    except ModuleNotFoundError:
        try:
            plot_with_pillow(series, args.title, output_base)
        except ModuleNotFoundError as exc:
            print(
                "[error] plotting requires matplotlib or Pillow. "
                "Try: conda run -n trt python scripts/66_plot_yaml_fig4.py ...",
                file=sys.stderr,
            )
            print(str(exc), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
