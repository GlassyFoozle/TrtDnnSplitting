"""
profiling_db.py — Lightweight profile-result cache.

Avoids re-profiling the same (model, variant, precision) configuration on every run.
Cache is a flat JSON file at results/optimization/.profiling_cache.json.

Key: "<model>|<variant>|<precision>"

Stored per entry:
  full_gpu_mean_ms      float | null
  per_chunk_gpu_mean_ms list[float] | null
  per_chunk_gpu_p99_ms  list[float] | null
  per_chunk_gpu_max_ms  list[float] | null
  total_chunked_gpu_mean_ms float | null
  source_json           str   (path to the C++ or Python result JSON)
  timestamp             str

Design notes:
- Written on every update to avoid losing data across runs.
- NOT thread-safe (single-process use only).
- Entries from existing C++ result JSONs are automatically imported on first lookup.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional


_CACHE_VERSION = 1


class ProfilingDB:
    def __init__(self, cache_path: str | Path) -> None:
        self._path = Path(cache_path)
        self._data: Dict[str, dict] = {}
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                self._data = raw.get("entries", {})
            except Exception:
                self._data = {}

    # ── Key format ────────────────────────────────────────────────────────────

    @staticmethod
    def make_key(model: str, variant: str, precision: str) -> str:
        return f"{model}|{variant}|{precision}"

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, model: str, variant: str, precision: str) -> Optional[dict]:
        return self._data.get(self.make_key(model, variant, precision))

    def has(self, model: str, variant: str, precision: str) -> bool:
        return self.make_key(model, variant, precision) in self._data

    def get_full_mean(self, model: str, variant: str, precision: str) -> Optional[float]:
        entry = self.get(model, variant, precision)
        if entry is None:
            return None
        return entry.get("full_gpu_mean_ms")

    def get_per_chunk_means(
        self, model: str, variant: str, precision: str
    ) -> Optional[List[float]]:
        entry = self.get(model, variant, precision)
        if entry is None:
            return None
        return entry.get("per_chunk_gpu_mean_ms")

    def get_total_chunked_mean(
        self, model: str, variant: str, precision: str
    ) -> Optional[float]:
        entry = self.get(model, variant, precision)
        if entry is None:
            return None
        return entry.get("total_chunked_gpu_mean_ms")

    # ── Write ─────────────────────────────────────────────────────────────────

    def put(
        self,
        model: str,
        variant: str,
        precision: str,
        *,
        full_gpu_mean_ms: Optional[float] = None,
        per_chunk_gpu_mean_ms: Optional[List[float]] = None,
        per_chunk_gpu_p99_ms: Optional[List[float]] = None,
        per_chunk_gpu_max_ms: Optional[List[float]] = None,
        total_chunked_gpu_mean_ms: Optional[float] = None,
        total_chunked_gpu_max_ms: Optional[float] = None,
        full_gpu_max_ms: Optional[float] = None,
        source_json: str = "",
    ) -> None:
        key = self.make_key(model, variant, precision)
        self._data[key] = {
            "model": model,
            "variant": variant,
            "precision": precision,
            "full_gpu_mean_ms": full_gpu_mean_ms,
            "full_gpu_max_ms": full_gpu_max_ms,
            "per_chunk_gpu_mean_ms": per_chunk_gpu_mean_ms,
            "per_chunk_gpu_p99_ms": per_chunk_gpu_p99_ms,
            "per_chunk_gpu_max_ms": per_chunk_gpu_max_ms,
            "total_chunked_gpu_mean_ms": total_chunked_gpu_mean_ms,
            "total_chunked_gpu_max_ms": total_chunked_gpu_max_ms,
            "source_json": source_json,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._flush()

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"version": _CACHE_VERSION, "entries": self._data}, indent=2)
        )

    # ── Import from existing result JSONs ─────────────────────────────────────

    def import_from_cpp_result(self, result_json_path: str | Path) -> bool:
        """
        Import timing data from a C++ table4_runner result JSON.
        Returns True if an entry was added/updated.
        """
        p = Path(result_json_path)
        if not p.exists():
            return False
        try:
            d = json.loads(p.read_text())
        except Exception:
            return False

        model = d.get("model", "")
        variant = d.get("variant", "")
        precision = d.get("precision", "fp32")
        if not model or not variant:
            return False

        full_mean = d.get("full_engine_gpu_mean_ms")
        full_max = d.get("full_engine_gpu_max_ms")
        chunks = d.get("chunks", [])
        per_chunk = [c["gpu_mean_ms"] for c in chunks] if chunks else None
        per_chunk_p99 = [c["gpu_p99_ms"] for c in chunks] if chunks else None
        per_chunk_max = [c["gpu_max_ms"] for c in chunks if "gpu_max_ms" in c] if chunks else None
        if per_chunk_max is not None and len(per_chunk_max) != len(chunks):
            per_chunk_max = None
        total = d.get("total_chunked_gpu_mean_ms")
        total_max = d.get("total_chunked_gpu_max_ms")

        self.put(
            model, variant, precision,
            full_gpu_mean_ms=full_mean,
            full_gpu_max_ms=full_max,
            per_chunk_gpu_mean_ms=per_chunk,
            per_chunk_gpu_p99_ms=per_chunk_p99,
            per_chunk_gpu_max_ms=per_chunk_max,
            total_chunked_gpu_mean_ms=total,
            total_chunked_gpu_max_ms=total_max,
            source_json=str(p),
        )
        return True

    def import_all_cpp_results(self, repo: Path) -> int:
        """Scan results/table4/ for C++ result JSONs and import all."""
        table4_dir = repo / "results" / "table4"
        count = 0
        if table4_dir.exists():
            for p in sorted(table4_dir.glob("*_cpp_*_fp*.json")):
                if self.import_from_cpp_result(p):
                    count += 1
        return count

    # ── Debug ─────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [f"ProfilingDB: {len(self._data)} entries  ({self._path})"]
        for key, entry in sorted(self._data.items()):
            full = entry.get("full_gpu_mean_ms")
            total = entry.get("total_chunked_gpu_mean_ms")
            n = len(entry.get("per_chunk_gpu_mean_ms") or [])
            lines.append(
                f"  {key:50s}  full={full:.4f}ms  total={total:.4f}ms  chunks={n}"
                if full and total else f"  {key}"
            )
        return "\n".join(lines)
