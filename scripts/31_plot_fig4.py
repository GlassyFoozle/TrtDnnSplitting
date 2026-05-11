#!/usr/bin/env python3
"""
31_plot_fig4.py — Plot YAML-driven Fig.4 schedulability curves.

Reads schedulability_ratio.csv from a script-30 run and writes PNG/PDF.
By default it renders the six paper-facing curves:
Offload/Uni × Tolerance/Heuristic/Optimal.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parent.parent
_PLOTS_DIR = REPO / "results" / "plots"

# ── Render order (lower index = front of legend) ─────────────────────────────
_ORDER_MAIN6 = [
    "SS-tol-fb", "SS-heu", "SS-opt",
    "UNI-tol-fb", "UNI-heu", "UNI-opt",
]

_ORDER_FULL8 = [
    "SS-tol-fb", "SS-heu", "SS-tol", "SS-opt",
    "UNI-tol-fb", "UNI-heu", "UNI-tol", "UNI-opt",
]

# Backward-compatible aliases from the old main4 label scheme.
_LEGACY_ALIASES = {
    "SS_ours":      "SS-tol-fb",
    "UNI_ours":     "UNI-tol-fb",
    "SS_Buttazzo":  "SS-opt",
    "UNI_Buttazzo": "UNI-opt",
}

_STYLE: Dict[str, dict] = {
    # Styles mirror simulation.py plot_bw_schedulability_ratio.
    # Open markers (markerfacecolor="none") except x and * (filled with line color).
    "SS-tol-fb":  {"color": "#8000FF", "marker": "*", "linestyle": "-",  "zorder": 8},
    "SS-heu":     {"color": "#00FF00", "marker": "v", "linestyle": "--", "zorder": 5},
    "SS-tol":     {"color": "#aec7e8", "marker": "p", "linestyle": "-.", "zorder": 3},
    "SS-opt":     {"color": "#008000", "marker": "D", "linestyle": "-",  "zorder": 4},
    "UNI-tol-fb": {"color": "#FF0000", "marker": "x", "linestyle": "-",  "zorder": 7},
    "UNI-heu":    {"color": "#00BFFF", "marker": "^", "linestyle": "--", "zorder": 5},
    "UNI-tol":    {"color": "#98df8a", "marker": "X", "linestyle": "-.", "zorder": 3},
    "UNI-opt":    {"color": "#0000FF", "marker": "s", "linestyle": "-",  "zorder": 4},
}

_LEGEND_LABELS: Dict[str, str] = {
    "SS-tol-fb":  "Offload-Tolerance",
    "SS-heu":     "Offload-Heuristic",
    "SS-tol":     "Offload-Tol",
    "SS-opt":     "Offload-Optimal",
    "UNI-tol-fb": "Uni-Tolerance",
    "UNI-heu":    "Uni-Heuristic",
    "UNI-tol":    "Uni-Tol",
    "UNI-opt":    "Uni-Optimal",
}

_PLOT_MODE_FILTERS: Dict[str, Optional[Set[str]]] = {
    "all":      set(_ORDER_MAIN6),
    "full8":    None,
    "main4":    {"SS-tol-fb", "UNI-tol-fb", "SS-opt", "UNI-opt",
                 "SS_ours", "UNI_ours", "SS_Buttazzo", "UNI_Buttazzo"},
    "ss_only":  {"SS-tol-fb", "SS-heu", "SS-tol", "SS-opt"},
    "uni_only": {"UNI-tol-fb", "UNI-heu", "UNI-tol", "UNI-opt"},
    "heu_tol_fb_only": {"SS-heu", "SS-tol-fb", "UNI-heu", "UNI-tol-fb"},
}

_PLOT_MODE_ORDER: Dict[str, List[str]] = {
    "all":      _ORDER_MAIN6,
    "full8":    _ORDER_FULL8,
    "main4":    ["SS-tol-fb", "UNI-tol-fb", "SS-opt", "UNI-opt"],
    "ss_only":  ["SS-tol-fb", "SS-heu", "SS-tol", "SS-opt"],
    "uni_only": ["UNI-tol-fb", "UNI-heu", "UNI-tol", "UNI-opt"],
    "heu_tol_fb_only": ["SS-heu", "SS-tol-fb", "UNI-heu", "UNI-tol-fb"],
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
    ap.add_argument(
        "--plot-mode",
        default="all",
        choices=list(_PLOT_MODE_FILTERS.keys()),
        dest="plot_mode",
        help="Which algorithms to include: all|full8|main4|ss_only|uni_only|heu_tol_fb_only",
    )
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
            # Normalize legacy labels to canonical names.
            algorithm = _LEGACY_ALIASES.get(algorithm, algorithm)
            series[algorithm].append((util, ratio))
    return {k: sorted(v) for k, v in series.items()}


def filter_series(
    series: Dict[str, List[Tuple[float, float]]],
    plot_mode: str,
) -> Dict[str, List[Tuple[float, float]]]:
    allowed = _PLOT_MODE_FILTERS.get(plot_mode)
    if allowed is None:
        return series
    return {k: v for k, v in series.items() if k in allowed}


def render_order(series: Dict[str, List[Tuple[float, float]]], plot_mode: str) -> List[str]:
    primary = _PLOT_MODE_ORDER.get(plot_mode, _ORDER_FULL8)
    extras = sorted(k for k in series if k not in primary)
    return primary + extras


def legend_label(label: str) -> str:
    return _LEGEND_LABELS.get(label, label)


def plot_with_matplotlib(
    series: Dict[str, List[Tuple[float, float]]],
    title: str,
    output_base: Path,
    dpi: int,
    plot_mode: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator

    # ── Font: prefer Times New Roman, fall back to Liberation Serif, then serif ─
    font_family = "serif"
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    if "Times New Roman" in available_fonts:
        font_family = "Times New Roman"
    else:
        liberation = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
        if liberation.exists():
            fm.fontManager.addfont(str(liberation))
            font_family = fm.FontProperties(fname=str(liberation)).get_name()

    # ── X-axis range derived from data ────────────────────────────────────────
    all_x = sorted({x for pts in series.values() for x, _ in pts})
    if not all_x:
        return
    x_pad = 0.025
    xlim = (all_x[0] - x_pad, all_x[-1] + x_pad)
    order = render_order(series, plot_mode)

    def _draw(include_legend: bool):
        fig, ax = plt.subplots(figsize=(6.6, 6.6))
        for label in order:
            points = series.get(label, [])
            if not points:
                continue
            xs, ys = zip(*points)
            style = _STYLE.get(label, {"color": "#333333", "marker": "o",
                                       "linestyle": "-", "zorder": 3})
            color = style["color"]
            marker = style["marker"]
            mfc = color if marker in {"x", "*"} else "none"
            ax.plot(
                xs, ys,
                color=color,
                linestyle=style["linestyle"],
                linewidth=1.55,
                marker=marker,
                markerfacecolor=mfc,
                markeredgecolor=color,
                markeredgewidth=2.0,
                markersize=15.75,
                label=legend_label(label),
                zorder=style.get("zorder", 3),
            )
        ax.set_xlabel("Utilization", fontsize=24)
        ax.set_ylabel("Schedulability Ratio", fontsize=24)
        ax.tick_params(axis="both", which="major", labelsize=24)
        ax.grid(True, linestyle=(0, (5, 4)), linewidth=1.4, alpha=0.7)
        ax.set_xlim(xlim)
        ax.set_ylim(0.0, 1.1)
        ax.set_xticks(all_x)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        if include_legend:
            ax.legend(loc="lower left", frameon=True, framealpha=0.88, markerscale=0.5)
        fig.tight_layout()
        return fig

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")

    with plt.rc_context({"font.family": font_family, "mathtext.fontset": "stix"}):
        png_fig = _draw(include_legend=True)
        png_fig.savefig(png, dpi=dpi)
        plt.close(png_fig)

        pdf_fig = _draw(include_legend=False)
        pdf_fig.savefig(pdf)
        plt.close(pdf_fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_with_pillow(
    series: Dict[str, List[Tuple[float, float]]],
    title: str,
    output_base: Path,
    plot_mode: str,
) -> None:
    """Minimal fallback used only if matplotlib is unavailable."""
    from PIL import Image, ImageDraw

    width, height = 1100, 720
    margin_l, margin_r, margin_t, margin_b = 90, 260, 70, 90
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

    draw.line([(margin_l, margin_t), (margin_l, height - margin_b)], fill="black", width=2)
    draw.line([(margin_l, height - margin_b), (width - margin_r, height - margin_b)], fill="black", width=2)
    draw.text((margin_l, 25), title, fill="black")
    draw.text((width // 2 - 200, height - 35), "Total U across CPUs", fill="black")
    draw.text((10, height // 2), "Schedulability Ratio", fill="black")

    _pil_colors = {
        "SS-tol-fb":  (31, 119, 180),
        "SS-heu":     (23, 190, 207),
        "SS-tol":     (174, 199, 232),
        "SS-opt":     (255, 127, 14),
        "UNI-tol-fb": (44, 160, 44),
        "UNI-heu":    (148, 103, 189),
        "UNI-tol":    (152, 223, 138),
        "UNI-opt":    (214, 39, 40),
    }

    legend_y = margin_t
    for label in render_order(series, plot_mode):
        points = series.get(label, [])
        if not points:
            continue
        color = _pil_colors.get(label, (80, 80, 80))
        coords = [(xp(x), yp(y)) for x, y in points]
        if len(coords) >= 2:
            draw.line(coords, fill=color, width=3)
        for x, y in coords:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.rectangle((width - 250, legend_y + 4, width - 230, legend_y + 14), fill=color)
        draw.text((width - 220, legend_y), legend_label(label), fill="black")
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

    series = filter_series(series, args.plot_mode)
    if not series:
        print(
            f"[error] no series left after applying --plot-mode={args.plot_mode}",
            file=sys.stderr,
        )
        return 1

    output_name = args.output
    if not output_name:
        if args.run_dir:
            output_name = Path(args.run_dir).name
        else:
            output_name = csv_path.parent.name
    if args.plot_mode != "all":
        output_name = f"{output_name}_{args.plot_mode}"
    output_base = Path(args.output_dir) / output_name

    try:
        plot_with_matplotlib(series, args.title, output_base, args.dpi, args.plot_mode)
    except ModuleNotFoundError:
        try:
            plot_with_pillow(series, args.title, output_base, args.plot_mode)
        except ModuleNotFoundError as exc:
            print(
                "[error] plotting requires matplotlib or Pillow. "
                "Try: conda run -n trt python scripts/31_plot_fig4.py ...",
                file=sys.stderr,
            )
            print(str(exc), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
