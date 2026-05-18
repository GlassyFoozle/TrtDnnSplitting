#!/usr/bin/env python3
"""Draw per-configuration chunk-ratio bar charts and check longest chunk flips."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


REPO = Path(__file__).resolve().parent.parent
PLOT_SCRIPT = REPO / "scripts" / "42_plot_finest_chunk_wcet_prediction.py"


def load_plot_module() -> Any:
    spec = importlib.util.spec_from_file_location("finest_chunk_plot", PLOT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PLOT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result-roots",
        nargs="+",
        default=["results_0512_no_vit/table4", "results/table4"],
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            "alexnet",
            "resnet18",
            "vgg19",
            "mobilenet_v3_small",
            "inception_v3",
            "vit_b_16",
            "vit_l_16",
        ],
    )
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--wcet-metric", default="p99", choices=["mean", "p99", "max"])
    ap.add_argument("--seed", type=int, default=20260516)
    ap.add_argument(
        "--out-dir",
        default="results/finest_chunk_ratio_barplots",
    )
    return ap.parse_args()


def svg_text(x: float, y: float, text: str, size: int = 13, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" fill="#27323a">'
        f"{escape(text)}</text>"
    )


def draw_ratio_svg(
    title: str,
    labels: list[str],
    actual_ratio: list[float],
    predicted_ratio: list[float],
    actual_longest: int,
    predicted_longest: int,
    out_path: Path,
) -> None:
    width = max(760, 140 + len(labels) * 112)
    height = 470
    left = 76
    right = 30
    top = 66
    bottom = 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = max(actual_ratio + predicted_ratio + [0.01])
    max_value = min(1.0, max(0.2, max_value * 1.18))
    group_w = plot_w / len(labels)
    bar_w = min(28, group_w * 0.28)

    def y(value: float) -> float:
        return top + plot_h - (value / max_value) * plot_h

    changed = actual_longest != predicted_longest
    subtitle = (
        f"longest changed: actual c{actual_longest}, predicted c{predicted_longest}"
        if changed
        else f"longest unchanged: c{actual_longest}"
    )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 28, title, 18),
        svg_text(width / 2, 50, subtitle, 12),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#88939b" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#88939b" stroke-width="1"/>',
    ]

    for i in range(6):
        value = max_value * i / 5
        yy = y(value)
        lines.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="#e5e9ec" stroke-width="1"/>'
        )
        lines.append(svg_text(left - 10, yy + 4, f"{value * 100:.1f}%", 11, "end"))

    for idx, label in enumerate(labels):
        cx = left + group_w * idx + group_w / 2
        ax = cx - bar_w - 3
        px = cx + 3
        ay = y(actual_ratio[idx])
        py = y(predicted_ratio[idx])
        actual_stroke = '#1b5fb8' if idx == actual_longest else 'none'
        pred_stroke = '#b95f00' if idx == predicted_longest else 'none'
        lines.append(
            f'<rect x="{ax:.1f}" y="{ay:.1f}" width="{bar_w:.1f}" height="{top + plot_h - ay:.1f}" '
            f'fill="#2f80ed" stroke="{actual_stroke}" stroke-width="2"/>'
        )
        lines.append(
            f'<rect x="{px:.1f}" y="{py:.1f}" width="{bar_w:.1f}" height="{top + plot_h - py:.1f}" '
            f'fill="#f2994a" stroke="{pred_stroke}" stroke-width="2"/>'
        )
        lines.append(svg_text(ax + bar_w / 2, ay - 6, f"{actual_ratio[idx] * 100:.1f}", 10))
        lines.append(svg_text(px + bar_w / 2, py - 6, f"{predicted_ratio[idx] * 100:.1f}", 10))
        lines.append(svg_text(cx, top + plot_h + 24, label, 12))

    lines.extend(
        [
            svg_text(left + plot_w / 2, height - 18, "chunk", 13),
            svg_text(18, top + plot_h / 2, "ratio of total WCET", 13, "middle").replace(
                "<text ", '<text transform="rotate(-90 18 {:.1f})" '.format(top + plot_h / 2), 1
            ),
            f'<rect x="{width - 228}" y="64" width="14" height="14" fill="#2f80ed"/>',
            svg_text(width - 207, 76, "actual ratio", 12, "start"),
            f'<rect x="{width - 118}" y="64" width="14" height="14" fill="#f2994a"/>',
            svg_text(width - 97, 76, "predicted ratio", 12, "start"),
            "</svg>",
        ]
    )
    out_path.write_text("\n".join(lines))


def write_outputs(rows: list[dict[str, Any]], notes: list[str], out_dir: Path, wcet_metric: str) -> None:
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for row in rows:
        actual_total = row["actual"]
        predicted_total = row["predicted"]
        actual_values = [chunk["actual"] for chunk in row["chunks"]]
        predicted_values = [chunk["predicted"] for chunk in row["chunks"]]
        actual_longest_pos = max(range(len(actual_values)), key=actual_values.__getitem__)
        predicted_longest_pos = max(range(len(predicted_values)), key=predicted_values.__getitem__)
        actual_longest_chunk = row["chunks"][actual_longest_pos]["chunk"]
        predicted_longest_chunk = row["chunks"][predicted_longest_pos]["chunk"]

        actual_ratio = [value / actual_total for value in actual_values]
        predicted_ratio = [value / predicted_total for value in predicted_values]
        filename = f"{row['model']}_k{row['k']}_ratio.svg"
        draw_ratio_svg(
            f"{row['model']} k={row['k']} chunk ratio",
            [f"c{chunk['chunk']}" for chunk in row["chunks"]],
            actual_ratio,
            predicted_ratio,
            actual_longest_pos,
            predicted_longest_pos,
            out_dir / filename,
        )

        summary_rows.append(
            {
                "model": row["model"],
                "k": row["k"],
                "variant": row["variant"],
                "actual_longest_chunk": actual_longest_chunk,
                "predicted_longest_chunk": predicted_longest_chunk,
                "longest_changed": actual_longest_chunk != predicted_longest_chunk,
                "plot": filename,
            }
        )
        for chunk, ar, pr in zip(row["chunks"], actual_ratio, predicted_ratio):
            detail_rows.append(
                {
                    "model": row["model"],
                    "k": row["k"],
                    "variant": row["variant"],
                    "chunk": chunk["chunk"],
                    "actual_ratio": f"{ar:.8f}",
                    "predicted_ratio": f"{pr:.8f}",
                    "actual_pct": f"{ar * 100:.4f}",
                    "predicted_pct": f"{pr * 100:.4f}",
                }
            )

    with (out_dir / "longest_chunk_check.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model",
                "k",
                "variant",
                "actual_longest_chunk",
                "predicted_longest_chunk",
                "longest_changed",
                "plot",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with (out_dir / "chunk_ratios.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model",
                "k",
                "variant",
                "chunk",
                "actual_ratio",
                "predicted_ratio",
                "actual_pct",
                "predicted_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    changed = [row for row in summary_rows if row["longest_changed"]]
    lines = [
        "# Chunk ratio bar plots and longest chunk check",
        "",
        f"- Blue: measured chunk {wcet_metric} WCET divided by measured total {wcet_metric} WCET.",
        f"- Orange: predicted chunk {wcet_metric} WCET divided by predicted total {wcet_metric} WCET.",
        "- A stroked bar marks the longest chunk in each series.",
        "",
        "## Longest Chunk Changes",
        "",
    ]
    if changed:
        lines.append("| model | k | actual longest | predicted longest | plot |")
        lines.append("|---|---:|---:|---:|---|")
        for row in changed:
            lines.append(
                f"| {row['model']} | {row['k']} | {row['actual_longest_chunk']} | "
                f"{row['predicted_longest_chunk']} | `{row['plot']}` |"
            )
    else:
        lines.append("모든 configuration에서 actual과 predicted의 longest chunk가 동일했다.")
    lines.append("")

    if notes:
        lines.append("## Data Notes")
        lines.append("")
        lines.extend(notes)
        lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("- `longest_chunk_check.csv`")
    lines.append("- `chunk_ratios.csv`")
    for row in summary_rows:
        lines.append(f"- `{row['plot']}`")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    plot_module = load_plot_module()
    rows, notes = plot_module.collect_rows(args, plot_module.load_report_module())
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(rows, notes, out_dir, args.wcet_metric)
    print(out_dir.relative_to(REPO))


if __name__ == "__main__":
    main()
