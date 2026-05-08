"""
Resolve paths relative to the project root (the directory containing this src/).
Everything is expressed relative to REPO_ROOT so the project can be relocated.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def repo(rel: str) -> Path:
    """Return an absolute path from a repo-relative string."""
    return REPO_ROOT / rel


def load_config(config_rel: str = "configs/resnet18_manual3.json") -> dict:
    path = repo(config_rel)
    with open(path) as f:
        cfg = json.load(f)
    return cfg


def chunk_onnx(cfg: dict, chunk_id: int, precision: str = "fp32") -> Path:
    return repo(cfg["chunks"][chunk_id]["onnx"])


def chunk_engine(cfg: dict, chunk_id: int, precision: str = "fp32") -> Path:
    key = f"engine_{precision}"
    return repo(cfg["chunks"][chunk_id][key])


def full_onnx(cfg: dict) -> Path:
    return repo(cfg["full_model"]["onnx"])


def full_engine(cfg: dict, precision: str = "fp32") -> Path:
    key = f"engine_{precision}"
    return repo(cfg["full_model"][key])
