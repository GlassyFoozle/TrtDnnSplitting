#!/usr/bin/env python3
"""Measure paper-like CPU wall-clock WCET without rebuilding TensorRT engines."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
RUNNER = REPO / "cpp_runtime" / "build_sync_wall" / "table4_runner"
MODELS = ("alexnet", "resnet18", "vgg19")
PAPER_TABLE_IV = {
    "alexnet": {"non-split": 4.469, "full-split": 4.802},
    "resnet18": {"non-split": 2.533, "full-split": 3.751},
    "vgg19": {"non-split": 6.615, "full-split": 11.425},
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--precision", default="fp32", choices=("fp32", "fp16"))
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    return ap.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def k1_config(model: str, precision: str) -> Dict[str, Any]:
    base_path = REPO / "artifacts" / "split_configs" / model / "dag_aligned_full.json"
    base = read_json(base_path)
    n = len(base["chunks"])
    interval = REPO / "artifacts" / "chunk_cache" / model / f"int_0_{n - 1}"
    engine = interval / f"chunk_{precision}.engine"
    onnx = interval / "chunk.onnx"
    if not engine.exists():
        raise FileNotFoundError(f"missing existing K=1 engine: {engine}")
    return {
        "model": model,
        "variant": "paper_like_k1",
        "base_variant": "dag_aligned_full",
        "n_chunks": 1,
        "input_shape": base.get("input_shape"),
        "full_model": {
            "engine_fp32": str(engine.relative_to(REPO)),
            "engine_fp16": str(engine.relative_to(REPO)),
        },
        "chunks": [
            {
                "id": 0,
                "description": f"all base chunks 0..{n - 1}",
                "source_chunk_ids": list(range(n)),
                "onnx": str(onnx.relative_to(REPO)),
                "engine_fp32": str(engine.relative_to(REPO)),
                "engine_fp16": str(engine.relative_to(REPO)),
            }
        ],
    }


def trt_fusion_safe_full_config(model: str, precision: str) -> Path:
    curve = (
        REPO
        / "results"
        / "best_k_split_curves"
        / f"{precision}_trt_fusion_safe_cache"
        / "best_k_split_curve.csv"
    )
    rows = [row for row in csv.DictReader(curve.open()) if row["model"] == model]
    if not rows:
        raise FileNotFoundError(f"missing trt_fusion_safe curve row for {model}: {curve}")
    row = max(rows, key=lambda item: int(item["k"]))

    from src.optimization.config_evaluator import mask_to_variant_name

    mask = [int(bit) for bit in row["best_mask"]]
    variant = mask_to_variant_name(model, mask)
    cfg = REPO / "artifacts" / "split_configs" / model / f"{variant}.json"
    if not cfg.exists():
        raise FileNotFoundError(f"missing existing max-K config: {cfg}")
    return cfg


def run_profile(config: Path, precision: str, warmup: int, iters: int) -> Path:
    cmd = [
        str(RUNNER),
        "--config",
        str(config),
        "--repo",
        str(REPO),
        "--precision",
        precision,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--sync-wall",
    ]
    subprocess.run(cmd, check=True)
    cfg = read_json(config)
    return REPO / "results" / "table4" / f"{cfg['model']}_cpp_{cfg['variant']}_{precision}.json"


def fmt(value: float) -> str:
    return f"{value:.3f}"


def chunk_maxes(result: Dict[str, Any]) -> List[float]:
    return [float(chunk["cpu_max_ms"]) for chunk in result["chunks"]]


def render_table(rows: List[Dict[str, Any]]) -> str:
    out = [
        "# Paper-like CPU wall-clock WCET",
        "",
        "측정 구간: `enqueue 직전 -> cudaStreamSynchronize()` 완료 직후.",
        "TensorRT engine은 기존 파일을 그대로 재사용했고 새로 빌드하지 않았다.",
        "full-split은 `trt_fusion_safe` 정책에서 허용되는 max-K split을 사용했다.",
        "",
        "## 새 측정 방식: paper-like CPU wall time",
        "",
        "| model | config | chunks | per-chunk CPU wall max (ms) | sum of chunk WCETs (ms) | measured total CPU wall max (ms) |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in rows:
        chunk_text = ", ".join(fmt(v) for v in row["chunks"])
        out.append(
            f"| {row['model']} | {row['config']} | {len(row['chunks'])} | "
            f"{chunk_text} | {fmt(row['sum'])} | {fmt(row['total'])} |"
        )
    out.append("")
    out.append("`sum of chunk WCETs`가 논문 Table IV의 Sum과 가장 직접적으로 대응한다.")
    out.extend(
        [
            "",
            "## 논문 Table IV와의 비교",
            "",
            "| model | metric | paper Table IV (ms) | current paper-like wall time (ms) |",
            "|---|---|---:|---:|",
        ]
    )
    for model in MODELS:
        model_rows = {row["config"]: row for row in rows if row["model"] == model}
        out.append(
            f"| {model} | non-split | {fmt(PAPER_TABLE_IV[model]['non-split'])} | "
            f"{fmt(model_rows['non-split']['sum'])} |"
        )
        out.append(
            f"| {model} | full-split sum | {fmt(PAPER_TABLE_IV[model]['full-split'])} | "
            f"{fmt(model_rows['full-split']['sum'])} |"
        )
    out.extend(
        [
            "",
            "## 짧은 해석",
            "",
            "- wall-clock + per-chunk sync를 넣자 full-split 합은 논문과 훨씬 가까워졌다. 특히 `resnet18`은 거의 일치한다.",
            "- non-split은 아직 크게 다르다. 측정 방식 외에도 Caffe 대 torchvision/ONNX 그래프 차이와 TensorRT engine/tactic 차이가 남아 있다는 뜻이다.",
            "- `vgg19`는 현재 full-split이 논문 6 chunks가 아니라 `trt_fusion_safe` max-K 12 chunks라서, chunk별 dispatch/sync overhead가 더 많이 누적된다.",
        ]
    )
    return "\n".join(out) + "\n"


def main() -> None:
    args = parse_args()
    if not RUNNER.exists():
        raise FileNotFoundError(f"missing runner binary: {RUNNER}")

    out_dir = REPO / "results" / "paper_like_walltime"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="paper-wall-", dir=REPO) as tmp:
        tmpdir = Path(tmp)
        for model in MODELS:
            non_split_cfg = tmpdir / f"{model}_paper_like_k1.json"
            non_split_cfg.write_text(json.dumps(k1_config(model, args.precision), indent=2) + "\n")
            full_split_cfg = trt_fusion_safe_full_config(model, args.precision)

            for label, cfg in (("non-split", non_split_cfg), ("full-split", full_split_cfg)):
                result_path = run_profile(cfg, args.precision, args.warmup, args.iters)
                result = read_json(result_path)
                copied = out_dir / f"{model}_{label.replace('-', '_')}_{args.precision}.json"
                shutil.copy2(result_path, copied)
                chunks = chunk_maxes(result)
                rows.append(
                    {
                        "model": model,
                        "config": label,
                        "chunks": chunks,
                        "sum": sum(chunks),
                        "total": float(result["total_chunked_cpu_wall_max_ms"]),
                    }
                )

    report = render_table(rows)
    report_path = REPO / "docs" / f"report_paper_like_walltime_{args.precision}_ko.md"
    report_path.write_text(report)
    print(report, end="")
    print(f"saved: {report_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
