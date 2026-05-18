#!/usr/bin/env python3
"""Generate a Korean markdown report comparing measured chunk WCETs with
predictions from the finest-granularity chunk profile.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SelectedConfig:
    model: str
    k: int
    result_path: Path
    config_path: Path
    result: dict[str, Any]
    config: dict[str, Any]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result-roots",
        nargs="+",
        default=["results_0512_no_vit/table4", "results/table4"],
        help="table4 JSON directories searched in order",
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
        "--out",
        default="docs/report_finest_chunk_wcet_prediction_ko.md",
        help="output markdown path",
    )
    return ap.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def metric(chunk: dict[str, Any], name: str) -> float:
    key = f"gpu_{name}_ms"
    if key not in chunk:
        raise KeyError(f"{key} missing in chunk {chunk.get('id')}")
    return float(chunk[key])


def fmt_ms(value: float) -> str:
    return f"{value:.6f}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}%"


def variant_from_path(path: Path, model: str, precision: str) -> str | None:
    prefix = f"{model}_cpp_"
    suffix = f"_{precision}.json"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    return name[len(prefix) : -len(suffix)]


def k_from_variant(variant: str) -> int | None:
    m = re.search(r"_k(\d+)$", variant)
    if m:
        return int(m.group(1))
    return None


def find_result_files(root: Path, model: str, precision: str) -> list[Path]:
    return sorted(root.glob(f"{model}_cpp_{model}_mask_*_k*_{precision}.json"))


def find_base_profile(root: Path, model: str, precision: str) -> Path | None:
    path = root / f"{model}_cpp_dag_aligned_full_{precision}.json"
    return path if path.exists() else None


def load_selected(
    roots: list[Path],
    models: list[str],
    precision: str,
    seed: int,
) -> tuple[list[SelectedConfig], list[str], dict[str, Path]]:
    rng = random.Random(seed)
    selected: list[SelectedConfig] = []
    notes: list[str] = []
    base_profiles: dict[str, Path] = {}

    for model in models:
        root_with_base = next((root for root in roots if find_base_profile(root, model, precision)), None)
        if root_with_base is None:
            notes.append(f"- `{model}`: finest profile JSON이 없어 제외함.")
            continue
        base_profiles[model] = find_base_profile(root_with_base, model, precision)  # type: ignore[assignment]

        files_by_k: dict[int, list[Path]] = {k: [] for k in range(1, 6)}
        for path in find_result_files(root_with_base, model, precision):
            variant = variant_from_path(path, model, precision)
            if not variant:
                continue
            k = k_from_variant(variant)
            if k in files_by_k:
                cfg_path = REPO / "artifacts" / "split_configs" / model / f"{variant}.json"
                if cfg_path.exists():
                    files_by_k[k].append(path)

        for k in range(1, 6):
            candidates = sorted(set(files_by_k[k]))
            if not candidates:
                notes.append(f"- `{model}`, k={k}: 측정 JSON 또는 split config가 없어 제외함.")
                continue
            result_path = rng.choice(candidates)
            variant = variant_from_path(result_path, model, precision)
            assert variant is not None
            config_path = REPO / "artifacts" / "split_configs" / model / f"{variant}.json"
            selected.append(
                SelectedConfig(
                    model=model,
                    k=k,
                    result_path=result_path,
                    config_path=config_path,
                    result=load_json(result_path),
                    config=load_json(config_path),
                )
            )

    return selected, notes, base_profiles


def chunk_source_ids(config_chunk: dict[str, Any]) -> list[int]:
    if "source_chunk_ids" in config_chunk:
        return [int(x) for x in config_chunk["source_chunk_ids"]]
    chunk_id = int(config_chunk["id"])
    return [chunk_id]


def make_report(args: argparse.Namespace) -> str:
    roots = [REPO / root for root in args.result_roots]
    selected, notes, base_paths = load_selected(roots, args.models, args.precision, args.seed)
    base_by_model = {
        model: load_json(path)
        for model, path in base_paths.items()
    }

    lines: list[str] = []
    lines.append("# Finest chunk WCET 합 기반 chunk 길이 예측 실험")
    lines.append("")
    lines.append("## 실험 방법")
    lines.append("")
    lines.append(
        f"- precision은 `{args.precision}`이고, WCET은 각 JSON의 `gpu_{args.wcet_metric}_ms`를 사용했다."
    )
    lines.append(
        "- finest granularity 기준은 각 모델의 `dag_aligned_full` profile이다."
    )
    lines.append(
        "- 특정 splitting configuration의 chunk 예측값은 해당 chunk가 포함하는 `source_chunk_ids`의 finest chunk WCET 합으로 계산했다."
    )
    lines.append(
        f"- random configuration 선택은 seed `{args.seed}`로 고정했다. 후보는 이미 측정된 table4 JSON 중 k=1..5를 사용했다."
    )
    lines.append("")

    if notes:
        lines.append("## 데이터 가용성")
        lines.append("")
        lines.extend(notes)
        lines.append("")

    for model in args.models:
        model_rows = [row for row in selected if row.model == model]
        if not model_rows:
            continue
        base = base_by_model[model]
        base_chunks = base["chunks"]
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"- finest chunk 수: {len(base_chunks)}"
        )
        lines.append(
            f"- finest profile: `{base_paths[model].relative_to(REPO)}`"
        )
        lines.append("")

        for item in sorted(model_rows, key=lambda x: x.k):
            split_name = "no-split" if item.k == 1 else f"{item.k} split"
            variant = item.result["variant"]
            lines.append(f"### {split_name}: `{variant}`")
            lines.append("")
            lines.append(
                f"- 측정 JSON: `{item.result_path.relative_to(REPO)}`"
            )
            lines.append(
                f"- split config: `{item.config_path.relative_to(REPO)}`"
            )
            lines.append("")
            lines.append(
                f"| chunk | finest source chunk | 실제 WCET {args.wcet_metric} (ms) | finest 합 예측 (ms) | 오차 (ms) | 오차율 |"
            )
            lines.append("|---:|---|---:|---:|---:|---:|")

            actual_chunks = sorted(item.result["chunks"], key=lambda x: int(x["id"]))
            cfg_chunks = sorted(item.config["chunks"], key=lambda x: int(x["id"]))
            total_actual = 0.0
            total_pred = 0.0
            for actual_chunk, cfg_chunk in zip(actual_chunks, cfg_chunks):
                source_ids = chunk_source_ids(cfg_chunk)
                actual = metric(actual_chunk, args.wcet_metric)
                pred = sum(metric(base_chunks[source_id], args.wcet_metric) for source_id in source_ids)
                err = actual - pred
                err_pct = None if pred == 0 else (err / pred) * 100.0
                total_actual += actual
                total_pred += pred
                source = (
                    str(source_ids[0])
                    if len(source_ids) == 1
                    else f"{source_ids[0]}..{source_ids[-1]} ({len(source_ids)}개)"
                )
                lines.append(
                    f"| {actual_chunk['id']} | {source} | {fmt_ms(actual)} | "
                    f"{fmt_ms(pred)} | {fmt_ms(err)} | {fmt_pct(err_pct)} |"
                )

            total_err = total_actual - total_pred
            total_err_pct = None if total_pred == 0 else (total_err / total_pred) * 100.0
            lines.append(
                f"| **합계** | - | **{fmt_ms(total_actual)}** | **{fmt_ms(total_pred)}** | "
                f"**{fmt_ms(total_err)}** | **{fmt_pct(total_err_pct)}** |"
            )
            lines.append("")

    lines.append("## 해석")
    lines.append("")
    lines.append(
        "finest chunk WCET 합은 split으로 인한 TensorRT fusion 변화, kernel launch 구성 변화, 중간 텐서 materialization 비용을 반영하지 않는다. "
        "따라서 실제 chunk WCET과 차이가 날 수 있으며, 위 표의 오차는 그 차이를 정량화한 값이다."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(make_report(args))
    print(out_path.relative_to(REPO))


if __name__ == "__main__":
    main()
