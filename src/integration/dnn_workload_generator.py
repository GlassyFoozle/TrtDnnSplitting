"""
dnn_workload_generator.py — Workload generation following paper Section VIII setup.

Generates synthetic DNN tasksets for schedulability experiments.

Two modes are supported:

legacy:

  1. Sample n_tasks models uniformly from available set.
  2. Get non-split GPU WCET G_i (K=1 from profiling DB or dag_aligned_full sum).
  3. UUniFast: distribute total utilization U_total across n_tasks tasks.
  4. T_i = G_i / U_i  (deadline-monotonic rate: T proportional to WCET).
  5. D_i = T_i (implicit deadlines).
  6. Assign priorities by deadline-monotonic order (shorter deadline = higher priority).
  7. cpu_pre/post sampled uniformly from configurable ranges.

dnnsplitting:

  Mirrors DNNSplitting/generate_task_set.py more closely. Utilization is first
  distributed across CPU partitions, then across tasks on each CPU. The
  original generator samples a period T and G_ratio, then computes
  total_work = U_i * T, G = total_work * G_ratio, and C = total_work - G.
  With real DNNs, G is fixed by TensorRT measurements, so this mode samples
  G_ratio and derives total_work = G / G_ratio and T = total_work / U_i.
  Period bounds are an optional validity filter rather than the driver.

UUniFast reference:
  Bini & Buttazzo (2005), "Measuring the Performance of Schedulability Tests."
  Guarantees mathematically unbiased utilization distribution.

Output format:
  configs/dnn_tasksets/generated/<run_name>/u<util>/taskset_NNN.json
  (same JSON schema as hand-crafted tasksets like mixed_two_dnn_demo.json)
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

# Known models and their dag_aligned_full chunk counts (for base-sum fallback)
_MODEL_N_CHUNKS: Dict[str, int] = {
    "alexnet":  22,
    "resnet18": 14,
    "vgg19":    46,
}

# Measured FP32 dag_aligned_full WCETs on Jetson AGX Orin (p99, ms).
# Used as final fallback in dry-run mode when no profiling cache is present.
_DRY_RUN_BASE_WCET_MS: Dict[str, float] = {
    "alexnet":  1.754,
    "resnet18": 1.037,
    "vgg19":    7.562,
}


# ── UUniFast ──────────────────────────────────────────────────────────────────

def uunifast(n: int, u_total: float, rng: random.Random) -> List[float]:
    """
    Generate n utilization values summing to u_total using UUniFast.

    Bini & Buttazzo (2005). Produces unbiased distribution over the
    n-dimensional simplex.
    """
    utils = []
    remaining = u_total
    for i in range(1, n):
        next_sum = remaining * (rng.random() ** (1.0 / (n - i)))
        utils.append(remaining - next_sum)
        remaining = next_sum
    utils.append(remaining)
    rng.shuffle(utils)
    return utils


# ── Base WCET lookup ──────────────────────────────────────────────────────────

def _get_base_gpu_wcet_ms(
    model_name: str,
    precision: str = "fp32",
    wcet_metric: str = "p99",
    profiling_db=None,
) -> Optional[float]:
    """
    Get the K=1 (no-split) GPU WCET for a model.

    Strategy (in order):
    1. ProfilingDB.get(model, "dag_aligned_full", precision) — sum of per-chunk times.
       This is the standard pre-profiled baseline variant; no K=1 evaluation is needed.
    2. Read .profiling_cache.json directly (same data, for callers without a DB object).
    3. Scan results/evaluations/<model>/ for any *.json with all-zeros mask.
    """
    metric_key = "per_chunk_gpu_p99_ms" if wcet_metric == "p99" else "per_chunk_gpu_mean_ms"

    # 1. ProfilingDB object (preferred — already loaded in memory)
    if profiling_db is not None:
        try:
            entry = profiling_db.get(model_name.lower(), "dag_aligned_full", precision)
            if entry:
                times = entry.get(metric_key) or entry.get("per_chunk_gpu_mean_ms")
                if times:
                    return float(sum(times))
        except Exception:
            pass

    # 2. Read cache file directly
    cache_path = REPO / "results" / "optimization" / ".profiling_cache.json"
    if cache_path.exists():
        try:
            cache_data = json.loads(cache_path.read_text())
            key = f"{model_name.lower()}|dag_aligned_full|{precision}"
            entry = cache_data.get("entries", {}).get(key)
            if entry:
                times = entry.get(metric_key) or entry.get("per_chunk_gpu_mean_ms")
                if times:
                    return float(sum(times))
        except Exception:
            pass

    # 3. Scan evaluation JSON files for an all-zeros mask (actual K=1 profile)
    eval_dir = REPO / "results" / "evaluations" / model_name.lower()
    if eval_dir.exists():
        for candidate in sorted(eval_dir.glob("*.json")):
            try:
                data = json.loads(candidate.read_text())
                mask = data.get("mask", [])
                if mask and sum(mask) == 0:
                    times = data.get(metric_key) or data.get("per_chunk_gpu_mean_ms")
                    if times:
                        return float(sum(times))
            except Exception:
                continue

    # 4. Dry-run fallback: use measured Jetson Orin reference values
    key_lower = model_name.lower()
    if key_lower in _DRY_RUN_BASE_WCET_MS:
        return _DRY_RUN_BASE_WCET_MS[key_lower]

    return None


# ── Task generation ───────────────────────────────────────────────────────────

@dataclass
class WorkloadConfig:
    """Configuration for workload generation."""
    models: List[str]                           # e.g. ["alexnet", "resnet18", "vgg19"]
    n_tasks: int                                # number of tasks in each taskset
    utilization: float                          # total utilization (sum of U_i)
    n_tasksets: int = 10                        # how many tasksets to generate
    precision: str = "fp32"
    wcet_metric: str = "p99"
    cpu_pre_range: Tuple[float, float] = (0.0, 0.0)   # (min, max) ms
    cpu_post_range: Tuple[float, float] = (0.0, 0.0)  # (min, max) ms
    period_min_ms: float = 5.0                 # minimum task period
    period_max_ms: float = 5000.0              # maximum task period
    cpu_id: int = 0                             # CPU for all tasks (single-CPU)
    seed: int = 42
    utilization_basis: str = "gpu"              # "gpu" or "total"
    taskgen_mode: str = "legacy"                # "legacy" or "dnnsplitting"
    num_cpus: int = 1                           # used by dnnsplitting mode
    g_ratio_range: Tuple[float, float] = (0.1, 0.8)
    uniform_cpu_utilization: bool = True
    uniform_task_utilization: bool = False
    tasks_per_cpu: Optional[int] = None
    g_utilization_threshold: float = 1.0
    number_of_inference_segments: int = 1
    max_block_count: Optional[int] = None
    per_splitting_overhead: float = 0.0
    max_retries: int = 200


def generate_tasksets(
    config: WorkloadConfig,
    output_dir: Optional[Path] = None,
    profiling_db=None,
) -> List[Path]:
    """
    Generate n_tasksets tasksets and write them to output_dir.

    Returns list of written JSON paths.
    """
    rng = random.Random(config.seed)
    paths = []

    # Collect base GPU WCETs for each requested model
    base_wcets: Dict[str, float] = {}
    for model in config.models:
        wcet = _get_base_gpu_wcet_ms(
            model, config.precision, config.wcet_metric, profiling_db
        )
        if wcet is not None and wcet > 0:
            base_wcets[model] = wcet
        else:
            # Fallback: use estimated sum from profile JSON metadata
            wcet = _estimate_base_wcet_from_metadata(model)
            if wcet is not None:
                base_wcets[model] = wcet

    if not base_wcets:
        raise RuntimeError(
            f"No base GPU WCET found for any model in {config.models}. "
            "Run K=1 profiling first or use --dry-run estimates."
        )

    available_models = list(base_wcets.keys())

    for ts_idx in range(config.n_tasksets):
        tasks = _generate_single_taskset(
            config=config,
            available_models=available_models,
            base_wcets=base_wcets,
            rng=rng,
            taskset_idx=ts_idx,
        )

        if tasks is None:
            continue  # Failed to generate a valid taskset

        if output_dir is not None:
            p = output_dir / f"taskset_{ts_idx:03d}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(tasks, indent=2))
            paths.append(p)

    return paths


def _generate_single_taskset(
    config: WorkloadConfig,
    available_models: List[str],
    base_wcets: Dict[str, float],
    rng: random.Random,
    taskset_idx: int,
    max_retries: int = 20,
) -> Optional[dict]:
    """
    Generate a single valid taskset dict.

    Retries up to max_retries times if any task ends up with period out of range.
    """
    if config.taskgen_mode == "legacy":
        return _generate_legacy_taskset(
            config=config,
            available_models=available_models,
            base_wcets=base_wcets,
            rng=rng,
            taskset_idx=taskset_idx,
            max_retries=max_retries,
        )
    if config.taskgen_mode == "dnnsplitting":
        return _generate_dnnsplitting_taskset(
            config=config,
            available_models=available_models,
            base_wcets=base_wcets,
            rng=rng,
            taskset_idx=taskset_idx,
            max_retries=max(max_retries, config.max_retries),
        )
    raise ValueError(
        f"Unsupported taskgen_mode={config.taskgen_mode!r}; "
        "expected 'legacy' or 'dnnsplitting'"
    )


def _generate_legacy_taskset(
    config: WorkloadConfig,
    available_models: List[str],
    base_wcets: Dict[str, float],
    rng: random.Random,
    taskset_idx: int,
    max_retries: int = 20,
) -> Optional[dict]:
    for attempt in range(max_retries):
        # Sample model names
        model_names = [rng.choice(available_models) for _ in range(config.n_tasks)]

        # UUniFast utilization distribution
        utils = uunifast(config.n_tasks, config.utilization, rng)

        tasks = []
        valid = True
        for i, (model, u_i) in enumerate(zip(model_names, utils)):
            G_i = base_wcets[model]
            if u_i <= 0:
                valid = False
                break

            cpu_pre = rng.uniform(*config.cpu_pre_range)
            cpu_post = rng.uniform(*config.cpu_post_range)
            cpu_total = cpu_pre + cpu_post
            if config.utilization_basis == "total":
                utilization_cost = G_i + cpu_total
            elif config.utilization_basis == "gpu":
                utilization_cost = G_i
            else:
                raise ValueError(
                    f"Unsupported utilization_basis={config.utilization_basis!r}; "
                    "expected 'gpu' or 'total'"
                )

            T_i = utilization_cost / u_i  # period from selected utilization basis

            if not (config.period_min_ms <= T_i <= config.period_max_ms):
                valid = False
                break

            tasks.append({
                "task_name": f"tau{i+1}_{model}",
                "model_name": model,
                "precision": config.precision,
                "period_ms": round(T_i, 6),
                "deadline_ms": round(T_i, 6),
                "priority": 0,  # will be assigned after sorting
                "cpu_id": config.cpu_id,
                "cpu_pre_ms": round(cpu_pre, 6),
                "cpu_post_ms": round(cpu_post, 6),
                "target_chunks": 1,
                "wcet_metric": config.wcet_metric,
                "notes": (
                    f"generated U={u_i:.4f} basis={config.utilization_basis} "
                    f"G={G_i:.4f}ms C={cpu_total:.4f}ms T={T_i:.4f}ms"
                ),
            })

        if not valid:
            continue

        # Assign priorities: deadline-monotonic (shorter deadline = higher priority)
        tasks.sort(key=lambda t: t["deadline_ms"])
        for prio, t in enumerate(tasks, start=1):
            t["priority"] = prio

        gpu_u = sum(base_wcets[t["model_name"]] / float(t["period_ms"]) for t in tasks)
        cpu_u = sum(
            (float(t["cpu_pre_ms"]) + float(t["cpu_post_ms"])) / float(t["period_ms"])
            for t in tasks
        )
        total_u = gpu_u + cpu_u
        per_cpu: Dict[str, float] = {}
        for t in tasks:
            key = str(t["cpu_id"])
            per_cpu[key] = per_cpu.get(key, 0.0) + (
                (float(t["cpu_pre_ms"]) + float(t["cpu_post_ms"])) / float(t["period_ms"])
            )

        return {
            "name": f"generated_u{config.utilization:.2f}_taskset_{taskset_idx:03d}",
            "description": "Generated DNN taskset for Fig.4-style schedulability experiments",
            "precision": config.precision,
            "wcet_metric": config.wcet_metric,
            "utilization_basis": config.utilization_basis,
            "taskgen_mode": config.taskgen_mode,
            "_generated": True,
            "_seed": config.seed,
            "_taskset_idx": taskset_idx,
            "_utilization": config.utilization,
            "_n_tasks": config.n_tasks,
            "_actual_gpu_utilization": round(gpu_u, 6),
            "_actual_cpu_utilization": round(cpu_u, 6),
            "_actual_total_utilization": round(total_u, 6),
            "_actual_cpu_partition_utilization": {
                cpu: round(util, 6) for cpu, util in sorted(per_cpu.items())
            },
            "tasks": tasks,
        }

    return None  # Could not generate valid taskset


def _generate_dnnsplitting_taskset(
    config: WorkloadConfig,
    available_models: List[str],
    base_wcets: Dict[str, float],
    rng: random.Random,
    taskset_idx: int,
    max_retries: int = 200,
) -> Optional[dict]:
    """
    Generate a DNN taskset using the same control structure as
    DNNSplitting/generate_task_set.py, adapted for real DNN GPU WCETs.

    DNNSplitting samples T first, derives total C+G work from U_i*T, samples
    G_ratio, and sets G = total_work*G_ratio. Since real DNN G is fixed, this
    function samples G_ratio and derives the period:

        total_work = real_G / sampled_g_ratio
        C          = total_work - real_G
        T          = total_work / U_i

    Thus G_ratio and U_i are preserved exactly, while period_range is used only
    as a validity filter. This is the closest real-DNN analogue of the
    synthetic DNNSplitting generator.
    """
    if config.num_cpus <= 0:
        raise ValueError("num_cpus must be positive in dnnsplitting taskgen mode")
    g_min, g_max = config.g_ratio_range
    if not (0.0 < g_min <= g_max <= 1.0):
        raise ValueError("g_ratio_range must satisfy 0.0 < min <= max <= 1.0")
    if config.tasks_per_cpu is not None and config.tasks_per_cpu <= 0:
        raise ValueError("tasks_per_cpu must be positive when set")

    expected_tasks = (
        config.tasks_per_cpu * config.num_cpus
        if config.tasks_per_cpu is not None
        else config.n_tasks
    )
    rejection_reasons: Dict[str, int] = {}

    def reject(reason: str) -> None:
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    for attempt in range(max_retries):
        if config.tasks_per_cpu is not None:
            cpu_task_counts = [config.tasks_per_cpu] * config.num_cpus
        else:
            cpu_task_counts = _distribute_task_count(config.n_tasks, config.num_cpus)
        if config.uniform_cpu_utilization:
            cpu_utils = [config.utilization / config.num_cpus] * config.num_cpus
        else:
            cpu_utils = uunifast(config.num_cpus, config.utilization, rng)

        tasks: List[dict] = []
        task_idx = 0
        valid = True
        model_counts: Dict[str, int] = {}
        actual_ratios: List[float] = []

        for cpu_id, count in enumerate(cpu_task_counts):
            if count <= 0:
                continue
            if config.uniform_task_utilization:
                task_utils = [cpu_utils[cpu_id] / count] * count
            else:
                task_utils = uunifast(count, cpu_utils[cpu_id], rng)

            for u_i in task_utils:
                if u_i <= 0:
                    reject("non_positive_task_utilization")
                    valid = False
                    break
                model = rng.choice(available_models)
                G_i = base_wcets[model]
                sampled_g_ratio = rng.uniform(g_min, g_max)
                if sampled_g_ratio <= 0:
                    reject("non_positive_g_ratio")
                    valid = False
                    break

                if config.utilization_basis == "total":
                    total_budget = G_i / sampled_g_ratio
                    cpu_total = total_budget - G_i
                    T_i = total_budget / u_i
                    actual_g_ratio = sampled_g_ratio
                    if not (config.period_min_ms <= T_i <= config.period_max_ms):
                        reject("derived_period_out_of_range")
                        valid = False
                        break
                elif config.utilization_basis == "gpu":
                    # Backward-compatible interpretation: U_i is GPU-only.
                    # CPU is still derived from sampled G_ratio, so total
                    # utilization is U_i / G_ratio. Prefer "total" for paper-
                    # style real-DNN tasksets.
                    T_i = G_i / u_i
                    if not (config.period_min_ms <= T_i <= config.period_max_ms):
                        reject("derived_period_out_of_range")
                        valid = False
                        break
                    total_budget = G_i / sampled_g_ratio
                    cpu_total = total_budget - G_i
                    actual_g_ratio = sampled_g_ratio
                else:
                    raise ValueError(
                        f"Unsupported utilization_basis={config.utilization_basis!r}; "
                        "expected 'gpu' or 'total'"
                    )

                cpu_pre, cpu_post = _split_cpu_budget(cpu_total, rng)
                task_idx += 1
                model_counts[model] = model_counts.get(model, 0) + 1
                actual_ratios.append(actual_g_ratio)
                tasks.append({
                    "task_name": f"tau{task_idx}_{model}",
                    "model_name": model,
                    "precision": config.precision,
                    "period_ms": round(T_i, 6),
                    "deadline_ms": round(T_i, 6),
                    "priority": 0,
                    "cpu_id": cpu_id,
                    "cpu_pre_ms": round(cpu_pre, 6),
                    "cpu_post_ms": round(cpu_post, 6),
                    "target_chunks": 1,
                    "wcet_metric": config.wcet_metric,
                    "target_utilization": round(u_i, 8),
                    "real_gpu_wcet_ms": round(G_i, 6),
                    "sampled_g_ratio": round(sampled_g_ratio, 6),
                    "actual_g_ratio": round(actual_g_ratio, 6),
                    "notes": (
                        f"generated taskgen=dnnsplitting U={u_i:.4f} "
                        f"basis={config.utilization_basis} G={G_i:.4f}ms "
                        f"C={cpu_total:.4f}ms T={T_i:.4f}ms "
                        f"G_ratio_sampled={sampled_g_ratio:.4f}"
                    ),
                })
            if not valid:
                break

        if not valid or len(tasks) != expected_tasks:
            continue

        tasks.sort(key=lambda t: t["deadline_ms"])
        for prio, t in enumerate(tasks, start=1):
            t["priority"] = prio

        gpu_u = sum(base_wcets[t["model_name"]] / float(t["period_ms"]) for t in tasks)
        cpu_u = sum(
            (float(t["cpu_pre_ms"]) + float(t["cpu_post_ms"])) / float(t["period_ms"])
            for t in tasks
        )
        total_u = gpu_u + cpu_u
        per_cpu: Dict[str, float] = {}
        for t in tasks:
            key = str(t["cpu_id"])
            per_cpu[key] = per_cpu.get(key, 0.0) + (
                (float(t["cpu_pre_ms"]) + float(t["cpu_post_ms"])) / float(t["period_ms"])
            )

        return {
            "name": f"generated_u{config.utilization:.2f}_taskset_{taskset_idx:03d}",
            "description": (
                "Generated DNN taskset using DNNSplitting-compatible utilization "
                "and CPU-assignment logic with real TensorRT DNN GPU WCETs"
            ),
            "precision": config.precision,
            "wcet_metric": config.wcet_metric,
            "utilization_basis": config.utilization_basis,
            "taskgen_mode": config.taskgen_mode,
            "_generated": True,
            "_seed": config.seed,
            "_taskset_idx": taskset_idx,
            "_utilization": config.utilization,
            "_n_tasks": expected_tasks,
            "_num_cpus": config.num_cpus,
            "_tasks_per_cpu": config.tasks_per_cpu,
            "_g_ratio_range": [g_min, g_max],
            "_g_utilization_threshold": config.g_utilization_threshold,
            "_number_of_inference_segments": config.number_of_inference_segments,
            "_max_block_count": config.max_block_count,
            "_per_splitting_overhead": config.per_splitting_overhead,
            "_uniform_cpu_utilization": config.uniform_cpu_utilization,
            "_uniform_task_utilization": config.uniform_task_utilization,
            "_generation_formula": (
                "sample U_i as DNNSplitting; sample G_ratio; set real_G=model "
                "WCET; total_exec=real_G/G_ratio; CPU=total_exec-real_G; "
                "period=deadline=total_exec/U_i for utilization_basis=total"
            ),
            "_generation_attempt": attempt + 1,
            "_rejection_reasons": dict(sorted(rejection_reasons.items())),
            "_actual_gpu_utilization": round(gpu_u, 6),
            "_actual_cpu_utilization": round(cpu_u, 6),
            "_actual_total_utilization": round(total_u, 6),
            "_actual_cpu_partition_utilization": {
                cpu: round(util, 6) for cpu, util in sorted(per_cpu.items())
            },
            "_model_distribution": dict(sorted(model_counts.items())),
            "_cpu_segment_distribution_ms": {
                "pre_min": round(min(float(t["cpu_pre_ms"]) for t in tasks), 6),
                "pre_max": round(max(float(t["cpu_pre_ms"]) for t in tasks), 6),
                "post_min": round(min(float(t["cpu_post_ms"]) for t in tasks), 6),
                "post_max": round(max(float(t["cpu_post_ms"]) for t in tasks), 6),
                "total_min": round(min(
                    float(t["cpu_pre_ms"]) + float(t["cpu_post_ms"]) for t in tasks
                ), 6),
                "total_max": round(max(
                    float(t["cpu_pre_ms"]) + float(t["cpu_post_ms"]) for t in tasks
                ), 6),
            },
            "_actual_g_ratio": {
                "min": round(min(actual_ratios), 6),
                "max": round(max(actual_ratios), 6),
                "avg": round(sum(actual_ratios) / len(actual_ratios), 6),
            },
            "_actual_period_ms": {
                "min": round(min(float(t["period_ms"]) for t in tasks), 6),
                "max": round(max(float(t["period_ms"]) for t in tasks), 6),
                "avg": round(
                    sum(float(t["period_ms"]) for t in tasks) / len(tasks), 6
                ),
            },
            "tasks": tasks,
        }

    return None


def _distribute_task_count(n_tasks: int, num_cpus: int) -> List[int]:
    counts = [n_tasks // num_cpus] * num_cpus
    for i in range(n_tasks % num_cpus):
        counts[i] += 1
    return counts


def _split_cpu_budget(cpu_total: float, rng: random.Random) -> Tuple[float, float]:
    if cpu_total <= 0:
        return 0.0, 0.0
    cut = rng.random()
    cpu_pre = cpu_total * cut
    return cpu_pre, cpu_total - cpu_pre


def _estimate_base_wcet_from_metadata(model_name: str) -> Optional[float]:
    """
    Estimate K=1 WCET from dag_aligned_full profiling cache (mean metric).
    Used as fallback when p99 is unavailable.
    """
    cache_path = REPO / "results" / "optimization" / ".profiling_cache.json"
    try:
        cache_data = json.loads(cache_path.read_text())
        for precision in ("fp32", "fp16"):
            key = f"{model_name.lower()}|dag_aligned_full|{precision}"
            entry = cache_data.get("entries", {}).get(key)
            if entry:
                times = entry.get("per_chunk_gpu_mean_ms") or entry.get("per_chunk_gpu_p99_ms")
                if times:
                    return float(sum(times))
    except Exception:
        pass
    return None


# ── Single-taskset writer ─────────────────────────────────────────────────────

def save_taskset(tasks_dict: dict, output_path: Path) -> None:
    """Write a taskset dict to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks_dict, indent=2))


def load_generated_taskset(json_path: Path) -> dict:
    """Load a generated taskset JSON."""
    return json.loads(json_path.read_text())
