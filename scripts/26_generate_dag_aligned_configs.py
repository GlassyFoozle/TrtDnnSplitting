#!/usr/bin/env python3
"""
26_generate_dag_aligned_configs.py — Generate dag_aligned_full split configs.

This materializes the max-granularity split universe consumed by
scripts/21_profile_base_chunks.py and the optimization evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate dag_aligned_full split configs.")
    ap.add_argument("--models", nargs="+", default=["alexnet", "resnet18", "vgg19", "vit_l_16"])
    ap.add_argument("--force", action="store_true", help="Overwrite existing configs")
    return ap.parse_args()


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _chunk_to_config(model_name: str, spec) -> dict:
    onnx_dir = REPO / "artifacts" / "onnx" / model_name / "dag_aligned_full"
    engine_dir = REPO / "artifacts" / "engines" / model_name / "dag_aligned_full"
    idx = int(spec.chunk_id)
    return {
        "id": idx,
        "name": f"chunk{idx}",
        "chunk_name": spec.chunk_name,
        "description": spec.description,
        "source_graph": spec.source_graph,
        "source_fx_node_name": spec.source_fx_node_name,
        "source_module_target": spec.source_module_target,
        "op_type": spec.op_type,
        "covered_fx_nodes": list(spec.covered_fx_nodes),
        "covered_module_targets": list(spec.covered_module_targets),
        "is_full_dag_critical_candidate": bool(spec.is_full_dag_critical_candidate),
        "materialization_status": spec.materialization_status,
        "boundary_reason": spec.boundary_reason,
        "starts_at_critical_candidate": bool(spec.starts_at_critical_candidate),
        "ends_at_critical_candidate": bool(spec.ends_at_critical_candidate),
        "dag_indices": str(idx + 1),
        "notes": spec.notes,
        "input_shape": list(spec.input_shape),
        "output_shape": list(spec.output_shape),
        "onnx": _rel(onnx_dir / f"chunk{idx}.onnx"),
        "engine_fp32": _rel(engine_dir / f"chunk{idx}_fp32.engine"),
        "engine_fp16": _rel(engine_dir / f"chunk{idx}_fp16.engine"),
    }


def generate_config(model_name: str) -> dict:
    from src.models.registry import build_model, get_model_info
    from src.splitting.dag_aligned_split import make_dag_aligned_chunks

    info = get_model_info(model_name)
    model = build_model(model_name)
    specs = make_dag_aligned_chunks(model_name, model)
    chunks = [_chunk_to_config(model_name, spec) for spec in specs]
    n_chunks = len(chunks)

    return {
        "model": model_name,
        "variant": "dag_aligned_full",
        "n_chunks": n_chunks,
        "input_shape": list(info.input_shape),
        "dag_summary": {
            "full_fx": {
                "n_nodes": n_chunks + 2,
                "n_edges": n_chunks + 1,
                "n_skip_edges": 0,
                "sync_counts": {
                    "e2e": 1,
                    "critical": n_chunks + 1,
                    "wo_skip": n_chunks + 1,
                    "max": n_chunks + 2,
                },
                "critical_indices": list(range(1, n_chunks + 1)),
            }
        },
        "dag_aligned_summary": {
            "candidate_count": n_chunks,
            "boundary_count": max(0, n_chunks - 1),
            "materialized_count": n_chunks,
            "notes": (
                "Generated from src.splitting.dag_aligned_split. "
                "ViT models use architecture-defined encoder-block chunks."
            ),
        },
        "chunks": chunks,
        "full_model": {
            "onnx": _rel(REPO / "artifacts" / "onnx" / model_name / "dag_aligned_full" / "full.onnx"),
            "engine_fp32": _rel(REPO / "artifacts" / "engines" / model_name / "dag_aligned_full" / "full_fp32.engine"),
            "engine_fp16": _rel(REPO / "artifacts" / "engines" / model_name / "dag_aligned_full" / "full_fp16.engine"),
        },
    }


def main() -> int:
    args = parse_args()
    for model_name in args.models:
        out_dir = REPO / "artifacts" / "split_configs" / model_name
        out_path = out_dir / "dag_aligned_full.json"
        if out_path.exists() and not args.force:
            print(f"{model_name}: exists, skip ({out_path.relative_to(REPO)})")
            continue
        cfg = generate_config(model_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(cfg, indent=2))
        print(f"{model_name}: wrote {out_path.relative_to(REPO)} ({cfg['n_chunks']} chunks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
