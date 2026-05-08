"""
selective_split.py — mask-driven split-point ON/OFF configuration generator.

A *selected split configuration* is derived from the max-granularity
`dag_aligned_full` base variant by toggling N-1 candidate split boundaries.

Boundary semantics
------------------
Given N base chunks numbered 0 … N-1 there are N-1 boundaries:
  boundary i = the split between base chunk i and base chunk i+1 (i in 0 … N-2)

A boundary mask is a binary vector of length N-1:
  1 = keep split active (boundary preserved → two separate chunks)
  0 = merge across this boundary (chunks fused into one)

Example (N=4):
  base:    [A, B, C, D]
  mask:    [1, 0, 1]       (boundaries 0 and 2 active, boundary 1 merged)
  result:  [A], [B+C], [D]

This directly mirrors the paper's active/inactive split-point vector over the
set of materializable candidate boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent.parent


# ── Mask parsing ──────────────────────────────────────────────────────────────

def parse_boundary_mask(
    raw: str | Sequence[int],
    n_base_chunks: int,
) -> List[int]:
    """
    Parse a mask specification into a binary list of length n_base_chunks - 1.

    Accepted forms:
      "all"       → all 1s (every boundary active = max granularity)
      "none"      → all 0s (no boundaries = single merged chunk)
      "10110…"    → binary string of length n_base_chunks - 1
      list[int]   → already-parsed binary list; validated for length
    """
    n_boundaries = n_base_chunks - 1
    if n_boundaries < 0:
        raise ValueError("n_base_chunks must be >= 1")

    if isinstance(raw, str):
        if raw.strip().lower() == "all":
            return [1] * n_boundaries
        if raw.strip().lower() == "none":
            return [0] * n_boundaries
        raw = raw.strip()
        if len(raw) != n_boundaries:
            raise ValueError(
                f"Binary mask string has length {len(raw)}, expected {n_boundaries}"
            )
        for ch in raw:
            if ch not in ("0", "1"):
                raise ValueError(f"Mask string contains non-binary character: {ch!r}")
        return [int(c) for c in raw]

    # Sequence[int]
    mask = list(raw)
    if len(mask) != n_boundaries:
        raise ValueError(
            f"Mask list has length {len(mask)}, expected {n_boundaries}"
        )
    if any(b not in (0, 1) for b in mask):
        raise ValueError("Mask list may only contain 0 or 1 values")
    return mask


def active_boundaries_to_mask(
    active_indices: Sequence[int],
    n_base_chunks: int,
) -> List[int]:
    """
    Build a mask from a list of active boundary indices.

    Boundary index i is active (1) if i appears in active_indices.
    All other boundaries are 0 (merged).
    """
    n_boundaries = n_base_chunks - 1
    mask = [0] * n_boundaries
    for idx in active_indices:
        if not (0 <= idx < n_boundaries):
            raise ValueError(
                f"Boundary index {idx} is out of range [0, {n_boundaries - 1}]"
            )
        mask[idx] = 1
    return mask


# ── Merge logic ───────────────────────────────────────────────────────────────

def compute_merge_groups(mask: List[int]) -> List[List[int]]:
    """
    Given a boundary mask of length N-1, return a list of groups.
    Each group is a list of consecutive base-chunk indices.

    boundary i connects chunk i and chunk i+1.
    If mask[i] == 1  → keep split → end current group, start new one.
    If mask[i] == 0  → merge    → continue current group.
    """
    n_chunks = len(mask) + 1
    groups: List[List[int]] = []
    current: List[int] = [0]
    for boundary_idx, bit in enumerate(mask):
        if bit == 1:
            groups.append(current)
            current = [boundary_idx + 1]
        else:
            current.append(boundary_idx + 1)
    groups.append(current)
    return groups


# ── Module reconstruction ─────────────────────────────────────────────────────

class _MergedChunk(nn.Module):
    """Sequential composition of multiple base-chunk modules."""
    def __init__(self, modules: List[nn.Module]):
        super().__init__()
        # Register as a flat ModuleList for clean state_dict / ONNX export.
        self.seq = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


def build_merged_module(
    base_modules: List[nn.Module],
    input_shape: Tuple[int, ...],
) -> Tuple[nn.Module, Tuple[int, ...]]:
    """
    Compose base_modules into a single nn.Module and infer output shape.
    Returns (merged_module, output_shape).
    """
    if len(base_modules) == 1:
        m = base_modules[0].eval()
    else:
        m = _MergedChunk([mod.eval() for mod in base_modules]).eval()

    with torch.no_grad():
        out = m(torch.zeros(*input_shape))
    return m, tuple(out.shape)


# ── Config generation ─────────────────────────────────────────────────────────

def load_base_config(model_name: str, base_variant: str = "dag_aligned_full") -> dict:
    """Load the base split config JSON."""
    path = REPO / "artifacts" / "split_configs" / model_name / f"{base_variant}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Base config not found: {path}\n"
            f"Run script 26 first to generate {base_variant} configs."
        )
    return json.loads(path.read_text())


def make_selected_split_config(
    model_name: str,
    base_variant: str,
    mask: List[int],
    selected_name: str,
    base_cfg: dict | None = None,
) -> dict:
    """
    Build the config dict for a selected split configuration.

    Parameters
    ----------
    model_name      : e.g. "alexnet"
    base_variant    : "dag_aligned_full"
    mask            : binary list of length N-1 (N = base chunk count)
    selected_name   : variant name for this config (e.g. "alexnet_poolstage")
    base_cfg        : pre-loaded base config (loaded from disk if None)

    Returns
    -------
    A config dict ready to be saved as <model>/<selected_name>.json.
    Engine/ONNX paths are relative to REPO.
    """
    if base_cfg is None:
        base_cfg = load_base_config(model_name, base_variant)

    base_chunks = base_cfg["chunks"]
    n_base = len(base_chunks)
    assert len(mask) == n_base - 1, f"mask length {len(mask)} != n_base-1={n_base-1}"

    groups = compute_merge_groups(mask)

    onnx_dir   = REPO / "artifacts" / "onnx"    / model_name / selected_name
    engine_dir = REPO / "artifacts" / "engines"  / model_name / selected_name

    def rel(p: Path) -> str:
        return str(p.relative_to(REPO))

    chunk_cfgs = []
    for grp_id, grp_chunk_ids in enumerate(groups):
        first_bc = base_chunks[grp_chunk_ids[0]]
        last_bc  = base_chunks[grp_chunk_ids[-1]]

        # Flatten metadata from all base chunks in this group
        src_ids    = grp_chunk_ids
        src_names  = [base_chunks[i]["chunk_name"] for i in grp_chunk_ids]
        src_descs  = [base_chunks[i]["description"] for i in grp_chunk_ids]
        cov_mod    = []
        cov_fx     = []
        for i in grp_chunk_ids:
            cov_mod.extend(base_chunks[i].get("covered_module_targets", []))
            cov_fx.extend(base_chunks[i].get("covered_fx_nodes", []))

        if len(grp_chunk_ids) == 1:
            desc = first_bc["description"]
        else:
            desc = f"{first_bc['chunk_name']}..{last_bc['chunk_name']} ({len(grp_chunk_ids)} merged)"

        chunk_cfgs.append({
            "id":                    grp_id,
            "name":                  f"chunk{grp_id}",
            "chunk_name":            f"chunk{grp_id}",
            "description":           desc,
            "source_chunk_ids":      src_ids,
            "source_chunk_names":    src_names,
            "source_descriptions":   src_descs,
            "covered_module_targets": cov_mod,
            "covered_fx_nodes":      cov_fx,
            "input_shape":           first_bc["input_shape"],
            "output_shape":          last_bc["output_shape"],
            "onnx":                  rel(onnx_dir / f"chunk{grp_id}.onnx"),
            "engine_fp32":           rel(engine_dir / f"chunk{grp_id}_fp32.engine"),
            "engine_fp16":           rel(engine_dir / f"chunk{grp_id}_fp16.engine"),
            "notes":                 "; ".join(
                bc.get("notes", "") for bc in [base_chunks[i] for i in grp_chunk_ids]
                if bc.get("notes", "")
            ),
        })

    cfg = {
        "model":                model_name,
        "variant":              selected_name,
        "base_variant":         base_variant,
        "n_chunks":             len(groups),
        "input_shape":          base_cfg["input_shape"],
        "candidate_count":      n_base,
        "boundary_count":       n_base - 1,
        "active_boundary_count": sum(mask),
        "mask":                 mask,
        "merged_groups":        groups,
        "chunks":               chunk_cfgs,
        "full_model": {
            "onnx":        rel(onnx_dir / "full.onnx"),
            "engine_fp32": rel(engine_dir / "full_fp32.engine"),
            "engine_fp16": rel(engine_dir / "full_fp16.engine"),
        },
    }
    return cfg


def save_selected_config(cfg: dict, model_name: str, selected_name: str) -> Path:
    """Save config to artifacts/split_configs/<model>/<selected_name>.json."""
    out_dir = REPO / "artifacts" / "split_configs" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{selected_name}.json"
    out_path.write_text(json.dumps(cfg, indent=2))
    return out_path


# ── Validation ────────────────────────────────────────────────────────────────

def validate_selected_config(
    model_name: str,
    selected_cfg: dict,
    base_variant: str = "dag_aligned_full",
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> dict:
    """
    Run each merged chunk in PyTorch and compare the final output against
    the full model. Returns a validation summary dict.

    Imports model and base chunks at call time to avoid heavy imports at module level.
    """
    import numpy as np
    from src.models.registry import build_model, get_model_info
    from src.splitting.dag_aligned_split import make_dag_aligned_chunks

    info = get_model_info(model_name)
    model = build_model(model_name)
    base_specs = make_dag_aligned_chunks(model_name, model)
    groups = selected_cfg["merged_groups"]

    # Full model reference
    x = torch.zeros(*info.input_shape)
    model.eval()
    with torch.no_grad():
        ref = model(x).numpy()

    # Chunked pipeline
    cur = x.clone()
    for grp in groups:
        base_modules = [base_specs[i].module for i in grp]
        merged_mod, _ = build_merged_module(base_modules, tuple(cur.shape))
        merged_mod.eval()
        with torch.no_grad():
            cur = merged_mod(cur)

    out = cur.numpy()
    max_abs_err = float(np.max(np.abs(out - ref)))
    top1_ref  = int(np.argmax(ref))
    top1_out  = int(np.argmax(out))
    top1_ok   = (top1_ref == top1_out)

    return {
        "max_abs_err": max_abs_err,
        "top1_agreement": top1_ok,
        "top1_ref": top1_ref,
        "top1_out": top1_out,
        "pass": max_abs_err < atol * 100 and top1_ok,
    }
