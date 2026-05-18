#!/usr/bin/env python3
"""Draw dependency-free SVG bar plots for finest-chunk WCET prediction."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


REPO = Path(__file__).resolve().parent.parent
REPORT_SCRIPT = REPO / "scripts" / "41_report_finest_chunk_wcet_prediction.py"


def load_report_module() -> Any:
    spec = importlib.util.spec_from_file_location("finest_chunk_report", REPORT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {REPORT_SCRIPT}")
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
        default="results/finest_chunk_wcet_prediction_barplots",
    )
    return ap.parse_args()


def fmt(value: float) -> str:
    return f"{value:.4f}"


def svg_text(x: float, y: float, text: str, size: int = 13, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" fill="#27323a">'
        f"{escape(text)}</text>"
    )


def draw_grouped_bar_svg(
    title: str,
    labels: list[str],
    actual: list[float],
    predicted: list[float],
    ylabel: str,
    out_path: Path,
    metric_label: str = "WCET",
) -> None:
    width = max(760, 120 + len(labels) * 110)
    height = 470
    left = 78
    right = 28
    top = 62
    bottom = 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = max(actual + predicted + [0.001]) * 1.12
    group_w = plot_w / len(labels)
    bar_w = min(28, group_w * 0.28)

    def y(value: float) -> float:
        return top + plot_h - (value / max_value) * plot_h

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 30, title, 18),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#88939b" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#88939b" stroke-width="1"/>',
    ]

    for i in range(6):
        value = max_value * i / 5
        yy = y(value)
        lines.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="#e5e9ec" stroke-width="1"/>'
        )
        lines.append(svg_text(left - 10, yy + 4, fmt(value), 11, "end"))

    for idx, label in enumerate(labels):
        cx = left + group_w * idx + group_w / 2
        ax = cx - bar_w - 3
        px = cx + 3
        ay = y(actual[idx])
        py = y(predicted[idx])
        lines.append(
            f'<rect x="{ax:.1f}" y="{ay:.1f}" width="{bar_w:.1f}" height="{top + plot_h - ay:.1f}" fill="#2f80ed"/>'
        )
        lines.append(
            f'<rect x="{px:.1f}" y="{py:.1f}" width="{bar_w:.1f}" height="{top + plot_h - py:.1f}" fill="#f2994a"/>'
        )
        lines.append(svg_text(ax + bar_w / 2, ay - 6, fmt(actual[idx]), 10))
        lines.append(svg_text(px + bar_w / 2, py - 6, fmt(predicted[idx]), 10))
        lines.append(svg_text(cx, top + plot_h + 24, label, 12))

    lines.extend(
        [
            svg_text(left + plot_w / 2, height - 18, "configuration", 13),
            svg_text(18, top + plot_h / 2, ylabel, 13, "middle").replace(
                "<text ", '<text transform="rotate(-90 18 {:.1f})" '.format(top + plot_h / 2), 1
            ),
            f'<rect x="{width - 205}" y="42" width="14" height="14" fill="#2f80ed"/>',
            svg_text(width - 184, 54, f"actual {metric_label}", 12, "start"),
            f'<rect x="{width - 105}" y="42" width="14" height="14" fill="#f2994a"/>',
            svg_text(width - 84, 54, "predicted sum", 12, "start"),
            "</svg>",
        ]
    )
    out_path.write_text("\n".join(lines))


def collect_rows(args: argparse.Namespace, report: Any) -> tuple[list[dict[str, Any]], list[str]]:
    roots = [REPO / root for root in args.result_roots]
    selected, notes, base_paths = report.load_selected(roots, args.models, args.precision, args.seed)
    base_by_model = {model: report.load_json(path) for model, path in base_paths.items()}
    rows: list[dict[str, Any]] = []

    for item in selected:
        base_chunks = base_by_model[item.model]["chunks"]
        actual_chunks = sorted(item.result["chunks"], key=lambda x: int(x["id"]))
        cfg_chunks = sorted(item.config["chunks"], key=lambda x: int(x["id"]))
        total_actual = 0.0
        total_predicted = 0.0
        chunk_rows = []
        for actual_chunk, cfg_chunk in zip(actual_chunks, cfg_chunks):
            source_ids = report.chunk_source_ids(cfg_chunk)
            actual = report.metric(actual_chunk, args.wcet_metric)
            predicted = sum(report.metric(base_chunks[source_id], args.wcet_metric) for source_id in source_ids)
            total_actual += actual
            total_predicted += predicted
            chunk_rows.append(
                {
                    "chunk": int(actual_chunk["id"]),
                    "actual": actual,
                    "predicted": predicted,
                }
            )
        rows.append(
            {
                "model": item.model,
                "k": item.k,
                "variant": item.result["variant"],
                "actual": total_actual,
                "predicted": total_predicted,
                "chunks": chunk_rows,
            }
        )
    return rows, notes


def write_csv(rows: list[dict[str, Any]], out_dir: Path, wcet_metric: str) -> None:
    actual_field = f"actual_{wcet_metric}_ms"
    predicted_field = f"predicted_{wcet_metric}_ms"
    with (out_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "k", "variant", actual_field, predicted_field, "error_ms", "error_pct"],
        )
        writer.writeheader()
        for row in rows:
            error = row["actual"] - row["predicted"]
            error_pct = 0.0 if row["predicted"] == 0 else error / row["predicted"] * 100.0
            writer.writerow(
                {
                    "model": row["model"],
                    "k": row["k"],
                    "variant": row["variant"],
                    actual_field: f"{row['actual']:.6f}",
                    predicted_field: f"{row['predicted']:.6f}",
                    "error_ms": f"{error:.6f}",
                    "error_pct": f"{error_pct:.4f}",
                }
            )


def write_index(rows: list[dict[str, Any]], notes: list[str], out_dir: Path, wcet_metric: str) -> None:
    lines = [
        "# Finest chunk WCET prediction bar plots",
        "",
        f"- Blue bars are measured chunked {wcet_metric} WCET.",
        f"- Orange bars are predicted {wcet_metric} WCET from the sum of finest chunks.",
        "- `summary_by_model.svg` compares total WCET for all selected configurations.",
        "- `<model>_k<k>_chunks.svg` compares chunk-level WCET within one configuration.",
        "",
    ]
    if notes:
        lines.append("## Data notes")
        lines.append("")
        lines.extend(notes)
        lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `summary.csv`")
    lines.append("- `summary_by_model.svg`")
    for row in rows:
        lines.append(f"- `{row['model']}_k{row['k']}_chunks.svg`")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    report = load_report_module()
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, notes = collect_rows(args, report)
    write_csv(rows, out_dir, args.wcet_metric)
    write_index(rows, notes, out_dir, args.wcet_metric)

    labels = [f"{row['model']} k={row['k']}" for row in rows]
    draw_grouped_bar_svg(
        f"Total {args.wcet_metric} WCET: measured vs finest-sum prediction",
        labels,
        [row["actual"] for row in rows],
        [row["predicted"] for row in rows],
        f"{args.wcet_metric} WCET (ms)",
        out_dir / "summary_by_model.svg",
        args.wcet_metric,
    )

    for row in rows:
        chunk_labels = [f"c{chunk['chunk']}" for chunk in row["chunks"]]
        draw_grouped_bar_svg(
            f"{row['model']} k={row['k']} chunk WCET",
            chunk_labels,
            [chunk["actual"] for chunk in row["chunks"]],
            [chunk["predicted"] for chunk in row["chunks"]],
            f"{args.wcet_metric} WCET (ms)",
            out_dir / f"{row['model']}_k{row['k']}_chunks.svg",
            args.wcet_metric,
        )

    print(out_dir.relative_to(REPO))


if __name__ == "__main__":
    main()
