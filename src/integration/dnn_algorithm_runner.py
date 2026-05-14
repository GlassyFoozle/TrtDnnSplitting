"""
dnn_algorithm_runner.py — DNN-aware splitting algorithm runner.

Connects DNNSplitting's analytical RTA algorithms to TensorRTServer's
real profiling backend.

Algorithms implemented
----------------------
SS model:
  single   — no-split baseline (K=1 per task, all boundaries off)
  max      — max-split baseline (K=N per task, all boundaries on)
  tol      — tolerance-fit splitting (follows DNNSplitting RTA_SS_tol)
  tol-fb   — tolerance-fit with fallback + optimistic early stop
  tol-fb-off — tolerance-fit with fallback, optimistic early stop disabled
  heu      — paper-style greedy: add one boundary at a time, pick best each round
  heu-k    — measured K-search greedy: increment K per lower task
  opt      — paper-style BFS-OPT: enumerate enabled-boundary subsets, min WCET
  opt-k    — measured K-search OPT: sweep K=1..N measured K masks

UNI model:
  single   — no-split baseline, UNI RTA
  max      — max-split baseline, UNI RTA
  tol      — tolerance-fit splitting in UNI space
  tol-fb   — UNI tol-fb with TRT GPU mask evaluation
  heu      — paper-style greedy in UNI space
  opt      — paper-style BFS-OPT in UNI space

Key hook: wherever the original DNNSplitting algorithm calls
  task.split_segment(seg_idx, K)
we call instead:
  _dnn_apply_k_split(dnn_task, seg_task, seg_idx, K, ...)
which:
  1. enumerates every policy-allowed mask with exactly K chunks
  2. calls evaluate_mask() for each candidate (cache-first)
  3. patches seg.G_block_list with MEASURED per-chunk max/WCET times
  4. returns MaskApplicationResult + updates profiling stats

DNNSplitting analysis.py functions reused without modification:
  get_SS_R, get_SS_tolerance, does_all_lower_meet_tolerance,
  find_splitting_target, update_SS_R_list_and_tolerance_list,
  update_UNI_R_list_and_tolerance_list,
  sort_task_set, get_UNI_R_and_K, get_UNI_tolerance,
  convert_task_SS_to_UNI, convert_task_list_to_SS, convert_task_list_to_UNI

No modification to DNNSplitting source files.

Split-point policy
------------------
The `policy_name` parameter restricts which boundaries may be active. Policies
are defined per model in configs/split_point_policies.json:
  "all"        — all N-1 boundaries enabled (default, no restriction)
  "paper_like" — skip fine-grained Conv→ReLU within-pair splits
  "stage"      — only at major architectural transitions (MaxPool + FC)
  "five_points" / "ten_points" — bounded design-phase search policies

Paper-style vs measured K-search algorithms
--------------------------------------
opt / heu (paper-style):
  Enumerate actual boundary-subset configs via BFS or greedy probing.
  Each profile_count += 1 = one evaluate_and_apply_mask call.
  Respects split-point policy (only enabled boundaries can be active).

opt-k / heu-k (measured K-search):
  For each requested K, enumerate all enabled boundary subsets with K-1 active
  boundaries, profile/cache each candidate, and choose the measured mask with
  the lowest max chunk time (tie: spread, then total GPU time).

UNI implementation note
-----------------------
UNI algorithms convert SS→UNI before analysis. After conversion, the UNI
InferenceSegment's base_block_list = [cpu_pre?, GPU_0..N-1, cpu_post?].
GPU boundaries in UNI space (positions 1..N-1) map directly to TRT boundaries
(TRT_boundary[k] = UNI_boundary[k+1] for k in 0..N-2).

For UNI algorithms that need splitting: extract TRT mask from UNI config,
call evaluate_mask, reconstruct UNI G_block_list as [cpu_pre?, g_0..g_K-1, cpu_post?].

UNI-single and UNI-max also use measured evaluator/cache timing before RTA.
"""

from __future__ import annotations

import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.integration.mask_applicator import (
    MaskApplicationResult, evaluate_and_apply_mask,
    apply_no_split_mask, apply_full_split_mask, apply_k_chunks,
)
from src.integration.dnn_taskset_generator import generate_dnn_taskset
from src.integration.dnnsplitting_adapter import build_task_set_dict, get_dnnsplitting_dir

from src.rta.analysis import (
    sort_task_set, get_SS_R, get_SS_tolerance,
    does_all_lower_meet_tolerance, find_splitting_target,
    update_SS_R_list_and_tolerance_list, update_UNI_R_list_and_tolerance_list,
    get_UNI_R_and_K, get_UNI_tolerance,
    get_optimistic_SS_R, get_optimistic_UNI_R,
    get_max_lower_blocking,
    convert_task_SS_to_UNI, convert_task_list_to_SS, convert_task_list_to_UNI,
)
import src.rta.analysis as _analysis_module

# Patch: ceil_div_with_context returns float when inputs are floats (due to Python
# float // semantics). UNI algorithms call range(1, K_i+1) which requires int.
# This module-level patch converts the return value to int, which is safe for
# all scheduling arithmetic (int * float = float, still correct).
_orig_ceil_div = _analysis_module.ceil_div_with_context

def _ceil_div_int(numerator, denominator, where, **context):
    result = _orig_ceil_div(numerator, denominator, where, **context)
    return int(result)

_analysis_module.ceil_div_with_context = _ceil_div_int


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ProfilingStats:
    masks_evaluated: int = 0
    baseline_k1_hits: int = 0        # Legacy counter; measured K=1 uses cache/real counters.
    cache_hits: int = 0
    real_profiles: int = 0
    builds_triggered: int = 0
    exports_triggered: int = 0
    dry_run_evaluations: int = 0
    skipped_cache_misses: int = 0   # attempt count (same mask may be counted >1×)
    # Unique-mask counters (set-based, not attempt-count)
    unique_masks_evaluated: int = 0
    unique_mask_cache_hits: int = 0
    unique_skipped_masks: int = 0
    interval_timing_cache_hits: int = 0   # served from interval GPU timing (no re-profile)
    k_split_calls: int = 0                # apply_k_chunks() invocations
    k_split_cache_hits: int = 0           # best-K cache hits inside apply_k_chunks()
    k_split_candidate_masks: int = 0      # total candidate masks implied by K-search
    k_split_candidate_mask_profiles: int = 0  # cold interval-cache e2e mask profiles
    k_split_candidate_mask_inference_runs: int = 0
    k_split_candidate_chunk_profiles: int = 0
    k_split_candidate_inference_runs: int = 0
    early_stop_optimistic_checks: int = 0
    early_stop_optimistic_deadline_misses: int = 0

    # Private set-based tracking for unique-mask counters (excluded from to_dict repr)
    _seen_evaluated: set = field(default_factory=set, repr=False, compare=False)
    _seen_cache_hits: set = field(default_factory=set, repr=False, compare=False)
    _seen_skipped: set = field(default_factory=set, repr=False, compare=False)
    _seen_k_split_candidate_intervals: set = field(default_factory=set, repr=False, compare=False)
    # Per-model skipped masks — {model: [mask_list, ...]} for diagnostic use
    _skipped_by_model: dict = field(default_factory=dict, repr=False, compare=False)

    # Interval-level cache accounting
    total_interval_cache_hits: int = 0
    total_interval_cache_misses: int = 0
    total_interval_onnx_cache_hits: int = 0
    total_interval_onnx_cache_misses: int = 0
    total_interval_engine_cache_hits: int = 0
    total_interval_engine_cache_misses: int = 0

    # Accumulated wall-clock times across all mask evaluations in this run
    total_export_wall_s: float = 0.0
    total_build_wall_s: float = 0.0
    total_profile_wall_s: float = 0.0
    total_interval_engine_build_wall_s: float = 0.0
    total_estimated_cold_s: float = 0.0

    def record_k_split_candidate_mask_profiles(
        self,
        model_name: str,
        precision: str,
        masks: List[List[int]],
        warmup: int = 0,
        iters: int = 0,
    ) -> None:
        """
        Count e2e mask profiles under a per-run cold interval-cache model.

        One profiled mask fills timing for all intervals/chunks in that mask.
        Later candidate masks are free when every required interval was already
        observed during this taskset-algorithm run.
        """
        for mask in masks:
            interval_keys = [
                (model_name, precision, tuple(group))
                for group in _mask_interval_groups(mask)
            ]
            if interval_keys and all(
                key in self._seen_k_split_candidate_intervals for key in interval_keys
            ):
                continue
            self.k_split_candidate_mask_profiles += 1
            self.k_split_candidate_mask_inference_runs += int(warmup) + int(iters)
            for key in interval_keys:
                self._seen_k_split_candidate_intervals.add(key)

    def update(self, result: MaskApplicationResult) -> None:
        self.masks_evaluated += 1
        if result.is_k1_baseline:
            # Legacy path retained for old callers; normal K=1 is measured.
            self.baseline_k1_hits += 1
            return

        mask_key = tuple(result.mask) if result.mask is not None else None

        if result.dry_run:
            self.dry_run_evaluations += 1
        elif getattr(result, "interval_timing_cache_hit", False):
            self.cache_hits += 1
            self.interval_timing_cache_hits += 1
            if mask_key is not None and mask_key not in self._seen_evaluated:
                self._seen_evaluated.add(mask_key)
                self.unique_masks_evaluated += 1
            if mask_key is not None and mask_key not in self._seen_cache_hits:
                self._seen_cache_hits.add(mask_key)
                self.unique_mask_cache_hits += 1
        elif result.cache_hit:
            self.cache_hits += 1
            if mask_key is not None and mask_key not in self._seen_evaluated:
                self._seen_evaluated.add(mask_key)
                self.unique_masks_evaluated += 1
            if mask_key is not None and mask_key not in self._seen_cache_hits:
                self._seen_cache_hits.add(mask_key)
                self.unique_mask_cache_hits += 1
        elif result.error in (
            "cache_miss_live_disabled",
            "global_budget_exhausted",
            "stop_on_first_build",
            "stopped_on_first_build",
        ):
            self.skipped_cache_misses += 1
            if mask_key is not None and mask_key not in self._seen_skipped:
                self._seen_skipped.add(mask_key)
                self.unique_skipped_masks += 1
                model = getattr(result, "model_name", "") or ""
                if model:
                    if model not in self._skipped_by_model:
                        self._skipped_by_model[model] = []
                    self._skipped_by_model[model].append(list(mask_key))
        else:
            self.real_profiles += 1
            if mask_key is not None and mask_key not in self._seen_evaluated:
                self._seen_evaluated.add(mask_key)
                self.unique_masks_evaluated += 1
        # Only count builds/exports that happened in THIS run (not historical
        # values inherited from a cached eval record's built/exported fields).
        if result.did_build and not result.cache_hit:
            self.builds_triggered += 1
        if result.did_export and not result.cache_hit:
            self.exports_triggered += 1
        self.total_interval_cache_hits += result.interval_cache_hits
        self.total_interval_cache_misses += result.interval_cache_misses
        self.total_interval_onnx_cache_hits += result.interval_onnx_cache_hits
        self.total_interval_onnx_cache_misses += result.interval_onnx_cache_misses
        self.total_interval_engine_cache_hits += result.interval_engine_cache_hits
        self.total_interval_engine_cache_misses += result.interval_engine_cache_misses
        if not result.cache_hit:
            self.total_interval_engine_build_wall_s += result.interval_engine_build_wall_s
            self.total_export_wall_s += result.export_wall_s
            self.total_build_wall_s += result.build_wall_s
        self.total_profile_wall_s += result.profile_wall_s
        if result.estimated_cold_total_s is not None:
            self.total_estimated_cold_s += result.estimated_cold_total_s

    def to_dict(self) -> dict:
        return {
            "masks_evaluated": self.masks_evaluated,
            "baseline_k1_hits": self.baseline_k1_hits,
            "cache_hits": self.cache_hits,
            "real_profiles": self.real_profiles,
            "builds_triggered": self.builds_triggered,
            "exports_triggered": self.exports_triggered,
            "dry_run_evaluations": self.dry_run_evaluations,
            "skipped_cache_misses": self.skipped_cache_misses,
            "unique_masks_evaluated": self.unique_masks_evaluated,
            "unique_mask_cache_hits": self.unique_mask_cache_hits,
            "unique_skipped_masks": self.unique_skipped_masks,
            "interval_timing_cache_hits": self.interval_timing_cache_hits,
            "k_split_calls": self.k_split_calls,
            "k_split_cache_hits": self.k_split_cache_hits,
            "k_split_candidate_masks": self.k_split_candidate_masks,
            "k_split_candidate_mask_profiles": self.k_split_candidate_mask_profiles,
            "k_split_candidate_mask_inference_runs": self.k_split_candidate_mask_inference_runs,
            "k_split_candidate_chunk_profiles": self.k_split_candidate_chunk_profiles,
            "k_split_candidate_inference_runs": self.k_split_candidate_inference_runs,
            "early_stop_optimistic_checks": self.early_stop_optimistic_checks,
            "early_stop_optimistic_deadline_misses": self.early_stop_optimistic_deadline_misses,
            "skipped_masks_detail": self._skipped_by_model,
            "total_interval_cache_hits": self.total_interval_cache_hits,
            "total_interval_cache_misses": self.total_interval_cache_misses,
            "total_interval_onnx_cache_hits": self.total_interval_onnx_cache_hits,
            "total_interval_onnx_cache_misses": self.total_interval_onnx_cache_misses,
            "total_interval_engine_cache_hits": self.total_interval_engine_cache_hits,
            "total_interval_engine_cache_misses": self.total_interval_engine_cache_misses,
            "total_interval_engine_build_wall_s": self.total_interval_engine_build_wall_s,
            "total_export_wall_s": self.total_export_wall_s,
            "total_build_wall_s": self.total_build_wall_s,
            "total_profile_wall_s": self.total_profile_wall_s,
            "total_estimated_cold_s": self.total_estimated_cold_s,
        }


def _mask_interval_groups(mask: List[int]) -> List[List[int]]:
    """Return consecutive base-chunk index groups implied by a boundary mask."""
    if not mask:
        return [[0]]
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


@dataclass
class TaskResult:
    task_name: str
    model_name: str
    cpu_id: int
    period_ms: float
    deadline_ms: float
    C_ms: float
    G_ms: float
    R_ms: float
    slack_ms: float
    schedulable: bool
    m: int
    max_G_block: float
    B_high: float
    B_low: float
    I: float
    final_mask: List[int]
    final_k_chunks: int
    final_chunk_times_ms: List[float]
    variant_name: str
    profile_result_path: str


@dataclass
class DNNAlgorithmResult:
    rta_model: str          # "SS" or "UNI"
    algorithm: str          # "single","max","tol","tol-fb","tol-fb-off","heu","heu-k","opt","opt-k"
    taskset_path: str
    precision: str
    wcet_metric: str
    dry_run: bool
    policy_name: str = "all"

    schedulable: bool = False
    error_type: Optional[str] = None
    error: Optional[str] = None
    unschedulable_reason: Optional[str] = None
    diagnostic_message: Optional[str] = None
    analysis_error: bool = False

    task_results: List[TaskResult] = field(default_factory=list)
    stats: ProfilingStats = field(default_factory=ProfilingStats)

    duration_s: float = 0.0
    algorithm_iterations: int = 0
    single_schedulable: Optional[bool] = None
    early_stopped_no_split: bool = False

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"DNN Algorithm Result: {self.rta_model}-{self.algorithm}",
            f"  taskset  : {self.taskset_path}",
            f"  precision: {self.precision}  wcet_metric: {self.wcet_metric}  "
            f"dry_run: {self.dry_run}  policy: {self.policy_name}",
            f"  result   : {'SCHEDULABLE' if self.schedulable else 'NOT SCHEDULABLE'}",
            f"  duration : {self.duration_s:.2f}s  iterations: {self.algorithm_iterations}",
            f"",
            f"  Profiling stats:",
            f"    masks_evaluated : {self.stats.masks_evaluated}",
            f"    cache_hits      : {self.stats.cache_hits}",
            f"    real_profiles   : {self.stats.real_profiles}",
            f"    builds          : {self.stats.builds_triggered}",
            f"    dry_run_evals   : {self.stats.dry_run_evaluations}",
        ]
        if self.single_schedulable is not None:
            lines.insert(
                7,
                f"  no-split : {'SCHEDULABLE' if self.single_schedulable else 'NOT SCHEDULABLE'}"
                f"  early_stop: {self.early_stopped_no_split}",
            )
        if self.error:
            prefix = f"{self.error_type}: " if self.error_type else ""
            lines.append(f"  ERROR: {prefix}{self.error}")
        if self.unschedulable_reason:
            lines.append(f"  UNSCHEDULABLE_REASON: {self.unschedulable_reason}")
        if self.diagnostic_message:
            lines.append(f"  DIAGNOSTIC: {self.diagnostic_message}")
        lines.append("")
        lines.append(f"  {'Task':25s}  {'R(ms)':9s}  {'D(ms)':9s}  {'Slack':9s}  "
                     f"{'C':7s}  {'G':7s}  {'K':4s}  {'MaxBlk':8s}  Sched?")
        lines.append("  " + "-" * 100)
        for tr in self.task_results:
            flag = "OK" if tr.schedulable else "MISS"
            lines.append(
                f"  {tr.task_name:25s}  {tr.R_ms:9.4f}  {tr.deadline_ms:9.4f}  "
                f"{tr.slack_ms:9.4f}  {tr.C_ms:7.4f}  {tr.G_ms:7.4f}  "
                f"{tr.final_k_chunks:4d}  {tr.max_G_block:8.4f}  {flag}"
            )
        lines.append(f"{'='*70}")
        return "\n".join(lines)


# ── State helpers ─────────────────────────────────────────────────────────────

def _snapshot_segment_state(seg_task, segment_idx: int):
    """
    Capture mutable splitting state for a single inference segment.

    DNN mask application patches SegInfTask in place. Paper-style SS OPT/HEU
    use a full-split feasibility probe before the real candidate search; that
    probe must not leak into the final selected mask.
    """
    seg = seg_task.inference_segment_list[segment_idx]
    return {
        "splitting_config": list(getattr(seg, "splitting_config", [])),
        "g_block_list": list(getattr(seg, "G_block_list", [])),
        "g_segment_list": list(seg_task.G_segment_list[segment_idx]),
        "G": seg_task.G,
        "max_G_block": seg_task.max_G_block,
    }


def _restore_segment_state(seg_task, segment_idx: int, snapshot) -> None:
    """Restore a snapshot captured by _snapshot_segment_state()."""
    seg = seg_task.inference_segment_list[segment_idx]
    seg.splitting_config = list(snapshot["splitting_config"])
    seg.G_block_list = list(snapshot["g_block_list"])
    seg_task.G_segment_list[segment_idx] = list(snapshot["g_segment_list"])
    seg_task.G = snapshot["G"]
    seg_task.max_G_block = snapshot["max_G_block"]


def _policy_max_chunks(dnn_task, seg, policy_name: str = "all") -> int:
    """Maximum chunks expressible by the active split policy for this segment."""
    if not policy_name or policy_name.lower() == "all":
        return int(seg.max_block_count)
    from src.integration.split_point_policy import get_enabled_boundaries
    enabled = get_enabled_boundaries(
        dnn_task.model_name, policy_name, int(seg.max_block_count) - 1
    )
    return min(int(seg.max_block_count), len(enabled) + 1)


# ── Main dispatcher ───────────────────────────────────────────────────────────

def run_dnn_rta_algorithm(
    dnn_taskset_path: str | Path,
    model: str,                     # "ss" or "uni"
    algorithm: str,                 # "single","max","tol","tol-fb","tol-fb-off","heu","heu-k","opt","opt-k"
    precision: str = "fp32",
    wcet_metric: str = "max",
    use_cpp: bool = True,
    force_profile: bool = False,
    dry_run: bool = False,
    max_iterations: int = 1000,
    exact_opt_max_boundaries: int = 0,  # retained for CLI compatibility
    warmup: int = 20,
    iters: int = 200,
    policy_name: str = "all",           # split-point policy: "all","paper_like","stage","five_points","ten_points"
    max_profiles: int = 500,            # budget for paper-style OPT/HEU per task
    max_candidates: int = 10000,        # max BFS candidates for paper-style OPT
    live_budget=None,                   # Optional[LiveProfileBudget] — shared global budget
    allow_proactive_splitting: bool = False,
    allow_equal_wcet_fallback: bool = False,
) -> DNNAlgorithmResult:
    """
    Run a DNN-aware splitting algorithm on a taskset JSON.

    Returns a DNNAlgorithmResult with schedulability verdict, per-task details,
    and profiling statistics.
    """
    t0 = time.time()

    result = DNNAlgorithmResult(
        rta_model=model.upper(),
        algorithm=algorithm,
        taskset_path=str(dnn_taskset_path),
        precision=precision,
        wcet_metric=wcet_metric,
        dry_run=dry_run,
        policy_name=policy_name,
        schedulable=False,
        error=None,
    )

    # Shared kwargs for all evaluate_and_apply_mask calls
    eval_kwargs = dict(
        precision=precision,
        wcet_metric=wcet_metric,
        use_cpp=use_cpp,
        force=force_profile,
        dry_run=dry_run,
        warmup=warmup,
        iters=iters,
        live_budget=live_budget,
    )

    # Load DNN tasks + build initial SegInfTask task_set
    dnn_tasks = generate_dnn_taskset(
        dnn_taskset_path,
        overlay_evaluations=True,
        allow_equal_wcet_fallback=allow_equal_wcet_fallback,
        allow_missing_base_timing_for_live=not dry_run,
    )
    task_set = build_task_set_dict(dnn_tasks)

    # Build task map: task.id (str) → (DNNBackedTask, SegInfTask)
    sorted_list = sort_task_set(task_set)
    task_map: Dict[str, Tuple] = {}
    for dt in dnn_tasks:
        # Find the corresponding SegInfTask in sorted_list by id
        for st in sorted_list:
            if str(st.id) == str(dt.task_name):
                task_map[str(dt.task_name)] = (dt, st)
                break

    try:
        if model.lower() == "ss":
            _dispatch_ss(
                algorithm=algorithm,
                sorted_task_list=sorted_list,
                task_map=task_map,
                result=result,
                eval_kwargs=eval_kwargs,
                max_iterations=max_iterations,
                exact_opt_max_boundaries=exact_opt_max_boundaries,
                policy_name=policy_name,
                max_profiles=max_profiles,
                max_candidates=max_candidates,
                allow_proactive_splitting=allow_proactive_splitting,
            )
        elif model.lower() == "uni":
            _dispatch_uni(
                algorithm=algorithm,
                sorted_task_list=sorted_list,
                task_map=task_map,
                result=result,
                eval_kwargs=eval_kwargs,
                max_iterations=max_iterations,
                policy_name=policy_name,
                max_profiles=max_profiles,
                max_candidates=max_candidates,
                allow_proactive_splitting=allow_proactive_splitting,
            )
        else:
            result.error = f"Unknown model: {model!r}. Use 'ss' or 'uni'."
    except Exception as exc:
        import traceback
        if type(exc).__name__ == "NumeratorExplosionError":
            _mark_clean_unschedulable(
                result,
                "rta_diverged_unschedulable",
                f"{exc}",
                sorted_list,
                task_map,
            )
        else:
            result.error_type = type(exc).__name__
            result.error = f"{exc}\n{traceback.format_exc()}"
            result.analysis_error = True
            result.schedulable = False

    result.duration_s = time.time() - t0
    return result


# ── Early infeasibility guards ───────────────────────────────────────────────

def _detect_rta_overload(sorted_task_list: list, model: str) -> Optional[Tuple[str, str]]:
    """
    Detect simple divergent/overloaded tasksets before entering fixed-point RTA.

    These are sufficient guards, not a replacement for the detailed DNNSplitting
    analysis. They keep obviously divergent tasksets from surfacing as Python
    exceptions during large Fig.4/Fig.5 sweeps.
    """
    eps = 1e-9
    for task in sorted_task_list:
        total = float(task.C + task.G)
        if total > float(task.D) + eps:
            return (
                "task_cost_exceeds_deadline",
                f"{task.id}: C+G={total:.6f}ms > D={float(task.D):.6f}ms",
            )

    if model == "uni":
        total_u = sum(float(t.C + t.G) / float(t.T) for t in sorted_task_list)
        if total_u >= 1.0 - eps:
            return (
                "uni_utilization_overload",
                f"sum((C+G)/T)={total_u:.6f} >= 1.0",
            )
        return None

    if model == "ss":
        gpu_u = sum(float(t.G) / float(t.T) for t in sorted_task_list)
        if gpu_u >= 1.0 - eps:
            return ("ss_gpu_overload", f"sum(G/T)={gpu_u:.6f} >= 1.0")

        per_cpu: Dict[object, float] = {}
        for task in sorted_task_list:
            per_cpu[task.cpu] = per_cpu.get(task.cpu, 0.0) + float(task.C) / float(task.T)
        for cpu, util in per_cpu.items():
            if util >= 1.0 - eps:
                return (
                    "ss_cpu_partition_overload",
                    f"cpu={cpu} sum(C/T)={util:.6f} >= 1.0",
                )

        # DNNSplitting's SS job-level response-time recurrence includes both
        # CPU and GPU demand from higher-priority tasks. If a priority prefix
        # already has combined utilization >= 1, the fixed point can diverge
        # even when separate CPU and GPU totals are each below 1.
        prefix_u = 0.0
        for idx, task in enumerate(sorted_task_list):
            prefix_u += float(task.C + task.G) / float(task.T)
            if idx > 0 and prefix_u >= 1.0 - eps:
                return (
                    "ss_combined_busy_period_overload",
                    f"priority_prefix_0..{idx} sum((C+G)/T)={prefix_u:.6f} >= 1.0",
                )
        return None

    return None


def _mark_clean_unschedulable(
    result: DNNAlgorithmResult,
    reason: str,
    message: str,
    sorted_task_list: list,
    task_map: dict,
) -> None:
    """Record a clean unschedulable result without counting it as analysis error."""
    result.schedulable = False
    result.unschedulable_reason = reason
    result.diagnostic_message = message
    result.error_type = None
    result.error = None
    result.analysis_error = False
    if not result.task_results:
        for st in sorted_task_list:
            nominal_r = float(st.C + st.G)
            result.task_results.append(
                _make_task_result(st, nominal_r, 0.0, 0.0, 0.0, task_map)
            )


# ── SS dispatcher ─────────────────────────────────────────────────────────────

def _dispatch_ss(
    algorithm: str,
    sorted_task_list: list,
    task_map: dict,
    result: DNNAlgorithmResult,
    eval_kwargs: dict,
    max_iterations: int,
    exact_opt_max_boundaries: int,
    policy_name: str = "all",
    max_profiles: int = 500,
    max_candidates: int = 10000,
    allow_proactive_splitting: bool = False,
) -> None:
    alg = algorithm.lower().replace("-", "_").replace(" ", "_")
    if alg == "single":
        _run_ss_single(sorted_task_list, task_map, result, eval_kwargs)
    elif alg == "max":
        _run_ss_max(sorted_task_list, task_map, result, eval_kwargs)
    elif alg == "tol":
        _run_ss_tol(sorted_task_list, task_map, result, eval_kwargs, max_iterations, policy_name)
    elif alg in ("tol_fb", "tolfb"):
        _run_ss_tol_fb(
            sorted_task_list, task_map, result, eval_kwargs, max_iterations,
            policy_name, early_stop=True,
        )
    elif alg in ("tol_fb_off", "tolfb_off", "tol_fb_no_es", "tol_fb_no_early_stop"):
        _run_ss_tol_fb(
            sorted_task_list, task_map, result, eval_kwargs, max_iterations,
            policy_name, early_stop=False,
        )
    elif alg == "heu":
        _run_ss_heu_paper(sorted_task_list, task_map, result, eval_kwargs,
                          policy_name, max_profiles)
    elif alg == "heu_k":
        _run_ss_heu_k(sorted_task_list, task_map, result, eval_kwargs, max_iterations, policy_name)
    elif alg == "opt":
        _run_ss_opt_paper(sorted_task_list, task_map, result, eval_kwargs,
                          policy_name, max_profiles, max_candidates)
    elif alg == "opt_k":
        _run_ss_opt_k(sorted_task_list, task_map, result, eval_kwargs, exact_opt_max_boundaries, policy_name)
    else:
        result.error = (
            f"Unknown SS algorithm: {algorithm!r}. "
            f"Supported: single, max, tol, tol-fb, tol-fb-off, heu, heu-k, opt, opt-k"
        )


# ── UNI dispatcher ────────────────────────────────────────────────────────────

def _dispatch_uni(
    algorithm: str,
    sorted_task_list: list,
    task_map: dict,
    result: DNNAlgorithmResult,
    eval_kwargs: dict,
    max_iterations: int,
    policy_name: str = "all",
    max_profiles: int = 500,
    max_candidates: int = 10000,
    allow_proactive_splitting: bool = False,
) -> None:
    alg = algorithm.lower().replace("-", "_").replace(" ", "_")
    if alg == "single":
        _run_uni_single(sorted_task_list, task_map, result, eval_kwargs)
    elif alg == "max":
        _run_uni_max(sorted_task_list, task_map, result, eval_kwargs)
    elif alg == "tol":
        _run_uni_tol(sorted_task_list, task_map, result, eval_kwargs, max_iterations, policy_name)
    elif alg in ("tol_fb", "tolfb"):
        _run_uni_tol_fb(sorted_task_list, task_map, result, eval_kwargs, max_iterations, policy_name)
    elif alg == "heu":
        _run_uni_heu_paper(sorted_task_list, task_map, result, eval_kwargs,
                           policy_name, max_profiles)
    elif alg == "opt":
        _run_uni_opt_paper(sorted_task_list, task_map, result, eval_kwargs,
                           policy_name, max_profiles, max_candidates)
    else:
        result.error = (
            f"Unknown UNI algorithm: {algorithm!r}. "
            f"Supported: single, max, tol, tol-fb, heu, opt"
        )


def _paper_no_split_gate_ss(sorted_task_list, task_map, result, eval_kwargs) -> bool:
    """
    Baseline gate for paper-style SS OPT/HEU.

    The TensorRT paper checks the non-split taskset first and only enters split
    search when that baseline misses deadlines. If the baseline passes, keep
    the no-split task rows and stop. If it fails, retain any no-split
    profiling/cache accounting, clear interim task rows, and let paper search
    continue from the no-split state.
    """
    _run_ss_single(sorted_task_list, task_map, result, eval_kwargs)
    result.single_schedulable = bool(result.schedulable)
    if result.schedulable:
        result.early_stopped_no_split = True
        return True

    result.early_stopped_no_split = False
    result.task_results.clear()
    result.schedulable = False
    result.algorithm_iterations = 0
    return False


def _apply_no_split_measured_to_all(sorted_task_list, task_map, result, eval_kwargs) -> bool:
    """Apply measured K=1 timing to every task before RTA consumes G values.

    K=1 is the all-zero mask: one merged interval int_0_(N-1).  It must use the
    measured per_chunk_gpu_max_ms from that mask, not the dag_aligned_full base
    chunk sum.  When live loading inserted placeholder base timing, failure here
    is fatal because RTA must never consume placeholders.
    """
    if getattr(result, "_measured_no_split_initialized", False):
        return True
    for st in sorted_task_list:
        dt, _ = task_map[str(st.id)]
        r = apply_no_split_mask(dt, st, 0, **eval_kwargs)
        result.stats.update(r)
        if not r.success:
            result.error = r.error or f"K=1 measured timing failed for task {st.id}"
            result.schedulable = False
            return False
    if not _assert_no_placeholder_timing_reaches_rta(sorted_task_list, task_map, result, "K=1 initialization"):
        return False
    result._measured_no_split_initialized = True
    return True


def _assert_no_placeholder_timing_reaches_rta(sorted_task_list, task_map, result, context: str) -> bool:
    """Reject analysis if metadata-only placeholder timing would feed RTA."""
    for st in sorted_task_list:
        dt, _ = task_map[str(st.id)]
        if bool(getattr(dt, "base_timing_placeholder", False)) and not bool(
            getattr(dt, "current_timing_measured", False)
        ):
            result.error = (
                f"{context}: placeholder dag_aligned_full timing for task {dt.task_name} "
                f"({dt.model_name}) was not replaced by measured mask timing"
            )
            result.schedulable = False
            return False
        for seg in getattr(st, "inference_segment_list", []):
            if bool(getattr(seg, "_base_timing_placeholder", False)) and not bool(
                getattr(seg, "_current_timing_measured", False)
            ):
                result.error = (
                    f"{context}: placeholder segment timing for task {dt.task_name} "
                    f"({dt.model_name}) was not replaced by measured mask timing"
                )
                result.schedulable = False
                return False
    return True


def _paper_no_split_gate_uni(sorted_task_list, task_map, result, eval_kwargs) -> bool:
    """
    Baseline gate for paper-style UNI OPT/HEU.

    The current UNI bridge performs this no-split test analytically, matching
    _run_uni_single. Candidate mask profiling is skipped when no split is
    required.
    """
    _run_uni_single(sorted_task_list, task_map, result, eval_kwargs)
    result.single_schedulable = bool(result.schedulable)
    if result.schedulable:
        result.early_stopped_no_split = True
        return True

    result.early_stopped_no_split = False
    result.task_results.clear()
    result.schedulable = False
    result.algorithm_iterations = 0
    return False


# ── SS: single (no-split baseline) ───────────────────────────────────────────

def _run_ss_single(sorted_task_list, task_map, result, eval_kwargs):
    """Apply K=1 mask to all tasks, run SS RTA."""
    if not _apply_no_split_measured_to_all(sorted_task_list, task_map, result, eval_kwargs):
        return

    R_list = []

    # Analysis
    for i in range(len(sorted_task_list)):
        R_i, B_i_high, B_i_low, I_i = get_SS_R(sorted_task_list, i, R_list)
        result.task_results.append(_make_task_result(
            sorted_task_list[i], R_i, B_i_high, B_i_low, I_i, task_map
        ))
        if R_i > sorted_task_list[i].D:
            result.schedulable = False
            result.algorithm_iterations = 1
            return
        R_list.append(R_i)

    result.schedulable = True
    result.algorithm_iterations = 1


# ── SS: max (full-split baseline) ─────────────────────────────────────────────

def _run_ss_max(sorted_task_list, task_map, result, eval_kwargs):
    """Apply K=N mask to all tasks, run SS RTA."""
    # full splitting
    for task in sorted_task_list:
        for segment_idx in range(task.m - 1):
            dt, _ = task_map[str(task.id)]
            r = apply_full_split_mask(dt, task, segment_idx, **eval_kwargs)
            result.stats.update(r)
            if not r.success:
                result.error = r.error or f"full-split measured timing failed for task {task.id}"
                result.schedulable = False
                return
    if not _assert_no_placeholder_timing_reaches_rta(sorted_task_list, task_map, result, "SS max initialization"):
        return

    # Analysis
    R_list = []
    for i in range(len(sorted_task_list)):
        R_i, B_i_high, B_i_low, I_i = get_SS_R(sorted_task_list, i, R_list)
        result.task_results.append(_make_task_result(
            sorted_task_list[i], R_i, B_i_high, B_i_low, I_i, task_map
        ))
        if R_i > sorted_task_list[i].D:
            result.schedulable = False
            result.algorithm_iterations = 1
            return
        R_list.append(R_i)

    result.schedulable = True
    result.algorithm_iterations = 1


# ── SS: tol ───────────────────────────────────────────────────────────────────

def _run_ss_tol(sorted_task_list, task_map, result, eval_kwargs, max_iterations,
                policy_name="all"):
    """
    DNN-aware SS-tol.

    Uses measured K-search. It still operates in K-space, but disabled policy
    boundaries are forced to 0.

    Follows DNNSplitting RTA_SS_tol logic:
    For each task in priority order:
      1. Compute SS RTA + tolerance
      2. If schedulable, continue
      3. If not, find lower-priority task whose max_G_block exceeds tolerance
         and split it to K+1 using measured-best K mask
      4. Re-evaluate RTA
      5. If all lower tasks meet tolerance → done; else → not schedulable
    """
    if not _apply_no_split_measured_to_all(sorted_task_list, task_map, result, eval_kwargs):
        return

    is_schedulable = True
    profiling_count = 0
    R_list = []
    tolerance_list = [math.inf for _ in range(len(sorted_task_list))]

    for i in range(len(sorted_task_list)):
        # Step 1: SS RTA
        profiling_count += 1
        task_i = sorted_task_list[i]
        C_i = task_i.C
        G_i = task_i.G
        D_i = task_i.D

        R_i, B_i_high, B_i_low, I_i = get_SS_R(sorted_task_list, i, R_list)

        is_last_task = (i == len(sorted_task_list) - 1)

        # Update tolerance
        if not is_last_task:
            tolerance_i = get_SS_tolerance(task_i, D_i, C_i, G_i, I_i, B_i_high)
        else:
            tolerance_i = math.inf
        tolerance_list[i] = tolerance_i

        if R_i <= D_i: # Current task meets deadline
            R_list.append(R_i)
            continue

        # Step 2: tolerance-fit splitting
        target_tolerance = min(tolerance_list[: i+1])
        if is_last_task or tolerance_i <= 0:
            is_schedulable = False
            result.schedulable = is_schedulable
            result.algorithm_iterations = profiling_count
            return

        meet_tolerance = False
        while profiling_count < max_iterations:
            target_tolerance = min(tolerance_list[: i+1])

            # All lower tasks meet tolerance
            if does_all_lower_meet_tolerance(sorted_task_list,i,target_tolerance):
                meet_tolerance = True
                break

            # Find splitting target
            split_target = find_splitting_target(sorted_task_list, i, target_tolerance)

            # No more splitting target
            if split_target is None:
                meet_tolerance = False
                break

            # Apply splitting
            target_task_index, target_segment_index, current_n = split_target
            dt, st = task_map[str(sorted_task_list[target_task_index].id)]
            if current_n >= _policy_max_chunks(dt, st.inference_segment_list[target_segment_index], policy_name):
                is_split = False
            else:
                app_r = apply_k_chunks(
                    dt, st, target_segment_index, current_n + 1, policy_name=policy_name,
                    search_stats=result.stats, **eval_kwargs
                )
                result.stats.update(app_r)
                is_split = app_r.success

            # Not splitable
            if not is_split:
                meet_tolerance = False
                break

            profiling_count += 1

            # Update R_list and tolerance_list
            R_list, new_tolerance_list = update_SS_R_list_and_tolerance_list(sorted_task_list, last_task_idx=i)
            tolerance_list[: i + 1] = new_tolerance_list

        if not meet_tolerance:
            is_schedulable = False
        else:
            if len(R_list) <= i:
                R_list.append(R_i)

        # Detect not schedulable
        if not is_schedulable:
            break

    result.schedulable = is_schedulable
    result.algorithm_iterations = profiling_count


# ── SS: tol-fb ────────────────────────────────────────────────────────────────

def _run_ss_tol_fb(sorted_task_list, task_map, result, eval_kwargs, max_iterations,
                   policy_name="all", early_stop=True):
    """
    DNN-aware SS-tol-fb.

    Uses measured K-search. It still operates in K-space, but disabled policy
    boundaries are forced to 0.

    Follows DNNSplitting _RTA_SS_tol_fb_impl logic with DNN-aware split hook.
    The `early_stop` flag toggles the optimistic-R stop check after fallback.
    Fallback: split the task with the largest max_G_block (excluding highest-priority).
    """
    if not _apply_no_split_measured_to_all(sorted_task_list, task_map, result, eval_kwargs):
        return

    is_schedulable = True
    profiling_count = 0
    R_list = []
    tolerance_list = [math.inf for _ in range(len(sorted_task_list))]

    i = 0
    while i < len(sorted_task_list) and profiling_count < max_iterations:
        '''
        # Step 1: SS RTA #####
        '''
        profiling_count += 1
        task_i = sorted_task_list[i]
        C_i = task_i.C
        G_i = task_i.G
        D_i = task_i.D

        R_i, B_i_high, B_i_low, I_i = get_SS_R(sorted_task_list, i, R_list)

        is_last_task = (i == len(sorted_task_list) - 1)

        # Update tolerance
        if not is_last_task:
            tolerance_i = get_SS_tolerance(task_i, D_i, C_i, G_i, I_i, B_i_high)
        else:
            tolerance_i = math.inf
        tolerance_list[i] = tolerance_i

        if R_i <= D_i: # Current task meets deadline
            if len(R_list) <= i:
                R_list.append(R_i)
            else:
                R_list[i] = R_i
            i += 1
            continue

        '''
        # Step 2: tol-based splitting
        '''

        meet_tolerance = False
        while profiling_count < max_iterations:
            # If invalid, go to fallback
            target_tolerance = min(tolerance_list[: i+1])
            if is_last_task or tolerance_i <= 0:
                break

            # All lower tasks meet tolerance
            if does_all_lower_meet_tolerance(sorted_task_list,i,target_tolerance):
                meet_tolerance = True
                break

            # Find splitting target
            split_target = find_splitting_target(sorted_task_list, i, target_tolerance)

            # No more splitting target
            if split_target is None:
                meet_tolerance = False
                break

            # Apply splitting
            target_task_index, target_segment_index, current_n = split_target
            dt, st_target = task_map[str(sorted_task_list[target_task_index].id)]
            if current_n >= _policy_max_chunks(
                dt, st_target.inference_segment_list[target_segment_index], policy_name
            ):
                is_split = False
            else:
                app_r = apply_k_chunks(
                    dt, st_target, target_segment_index, current_n + 1,
                    policy_name=policy_name, search_stats=result.stats, **eval_kwargs
                )
                result.stats.update(app_r)
                is_split = app_r.success

            # Not splitable
            if not is_split:
                meet_tolerance = False
                break

            profiling_count += 1

            # Update R_list and tolerance_list
            R_list, new_tolerance_list = update_SS_R_list_and_tolerance_list(sorted_task_list, last_task_idx=i)
            tolerance_list[: i + 1] = new_tolerance_list

        if meet_tolerance:
            if len(R_list) <= i:
                R_list.append(R_i)
            i += 1
            continue

        '''
        # Step 3: Fallback
        '''
        is_splitted, splitted_task_idx, app_r = _dnn_split_largest_excluding_highest(
            sorted_task_list, task_map, eval_kwargs, policy_name=policy_name,
            search_stats=result.stats,
        )
        if app_r is not None:
            result.stats.update(app_r)
        if not is_splitted or splitted_task_idx is None:
            is_schedulable = False
            break
        profiling_count += 1

        # Update task idx if splited task idx is higher
        restart_idx = i if i <= splitted_task_idx else splitted_task_idx

        # Update R_list and tolerance_list
        if R_list:
            R_list, new_tolerance_list = update_SS_R_list_and_tolerance_list(
                sorted_task_list, last_task_idx=(len(R_list)-1)
            )
            tolerance_list[: len(new_tolerance_list)] = new_tolerance_list
        else:
            R_list = []

        '''
        # Step 4: Early stop
        '''
        if early_stop:
            optimistic_R_list = get_optimistic_SS_R(sorted_task_list)
            result.stats.early_stop_optimistic_checks += len(optimistic_R_list)
            early_stop_hit = False
            for k, R_k in enumerate(optimistic_R_list):
                task_k = sorted_task_list[k]
                if task_k.D < R_k:
                    result.stats.early_stop_optimistic_deadline_misses += 1
                    is_schedulable = False
                    early_stop_hit = True
                    break
            if early_stop_hit:
                break

        # Detect not schedulable
        if not is_schedulable:
            break

        # Reset target index
        i = restart_idx

    if profiling_count >= max_iterations:
        result.error = f"Max iterations ({max_iterations}) reached"

    result.algorithm_iterations = profiling_count

    final_R_list = []
    for i, st in enumerate(sorted_task_list):
        R_i, B_i_high, B_i_low, I_i = get_SS_R(sorted_task_list, i, final_R_list)
        final_R_list.append(R_i)
        if R_i > st.D:
            is_schedulable = False
        result.task_results.append(_make_task_result(
            st, R_i, B_i_high, B_i_low, I_i, task_map
        ))

    result.schedulable = is_schedulable


# ── SS: heu-k (measured K-search greedy) ─────────────────────────────────────

def _run_ss_heu_k(sorted_task_list, task_map, result, eval_kwargs, max_iterations,
                  policy_name="all"):
    """
    DNN-aware SS-heu-k (measured K-search greedy).

    Uses measured K-search. It still operates in K-space, but disabled policy
    boundaries are forced to 0.

    For each task in priority order:
      Compute SS RTA + tolerance.
      While lower-priority task has max_G_block > tolerance:
        Increment its K by 1 using measured-best K split.
        Re-evaluate tolerance.
    This is a greedy per-task heuristic without full re-analysis from scratch.
    Use 'heu' for paper-style boundary-subset greedy.
    """
    if not _apply_no_split_measured_to_all(sorted_task_list, task_map, result, eval_kwargs):
        return

    n = len(sorted_task_list)
    R_list = []
    tolerance_list = [math.inf] * n
    schedulable = True
    iterations = 0

    for i in range(n):
        task_i = sorted_task_list[i]
        C_i, G_i, D_i = task_i.C, task_i.G, task_i.D
        is_last = (i == n - 1)

        R_i, B_hi, B_lo, I_i = get_SS_R(sorted_task_list, i, R_list)
        iterations += 1

        tolerance_i = get_SS_tolerance(task_i, D_i, C_i, G_i, I_i, B_hi) if not is_last else math.inf
        tolerance_list[i] = tolerance_i

        if R_i > D_i:
            schedulable = False
            R_list.append(R_i)
            continue

        R_list.append(R_i)

        if is_last:
            continue

        cur_tol = min(tolerance_list[:i + 1])

        # Greedily increase K for lower-priority tasks until tolerance met
        for j in range(i + 1, n):
            task_j = sorted_task_list[j]
            for s_idx, seg in enumerate(task_j.inference_segment_list):
                while (
                    seg.G_block_list and
                    max(seg.G_block_list) > cur_tol and
                    seg.size < _policy_max_chunks(
                        task_map[str(task_j.id)][0], seg, policy_name
                    ) and
                    iterations < max_iterations
                ):
                    dt_j, st_j = task_map[str(task_j.id)]
                    app_r = apply_k_chunks(
                        dt_j, st_j, s_idx, seg.size + 1,
                        policy_name=policy_name, search_stats=result.stats, **eval_kwargs
                    )
                    iterations += 1
                    result.stats.update(app_r)
                    if not app_r.success:
                        break

    # Final RTA pass
    final_R_list = []
    for i, st in enumerate(sorted_task_list):
        R, B_hi, B_lo, I = get_SS_R(sorted_task_list, i, final_R_list)
        final_R_list.append(R)
        if R > st.D:
            schedulable = False
        result.task_results.append(_make_task_result(st, R, B_hi, B_lo, I, task_map))

    result.schedulable = schedulable
    result.algorithm_iterations = iterations


# ── SS: opt-k (measured K-search OPT) ────────────────────────────────────────

def _run_ss_opt_k(sorted_task_list, task_map, result, eval_kwargs,
                  exact_opt_max_boundaries, policy_name="all"):
    """
    DNN-aware SS-opt-k (measured K-search OPT).

    Uses measured K-search for the K-sweep. It still operates in K-space, but
    disabled policy boundaries are forced to 0.

    Enumerate K = 1, 2, ..., N for each lower-priority task using measured-best
    K masks.
    For each task, select the minimum K (minimum max_G_block) that satisfies the
    current tolerance.

    This is NOT the exhaustive OPT over all 2^(N-1) masks (infeasible for VGG19).
    It is O(N_j) evaluations per task j, which is at most 22/46/14 per model.

    If exact_opt_max_boundaries > 0 and boundary_count <= exact_opt_max_boundaries,
    enumerate all 2^(N-1) masks instead.
    Use 'opt' for paper-style boundary-subset BFS-OPT.
    """
    if not _apply_no_split_measured_to_all(sorted_task_list, task_map, result, eval_kwargs):
        return

    n = len(sorted_task_list)
    R_list = []
    tolerance_list = []
    schedulable = True
    iterations = 0

    for i in range(n):
        task_i = sorted_task_list[i]
        C_i, G_i, D_i = task_i.C, task_i.G, task_i.D
        is_last = (i == n - 1)

        R_i, B_hi, B_lo, I_i = get_SS_R(sorted_task_list, i, R_list)
        iterations += 1

        tolerance_i = get_SS_tolerance(task_i, D_i, C_i, G_i, I_i, B_hi) if not is_last else math.inf
        tolerance_list.append(tolerance_i)
        R_list.append(R_i)

        if i == 0:
            continue

        cur_tol = min(tolerance_list)

        # Full-split feasibility check
        for j in range(1, i + 1):
            task_j = sorted_task_list[j]
            for s_idx, seg in enumerate(task_j.inference_segment_list):
                if seg.max_block_count <= 1:
                    continue
                # Probe: max split
                dt_j, st_j = task_map[str(task_j.id)]
                saved_state = _snapshot_segment_state(st_j, s_idx)
                full_r = apply_k_chunks(
                    dt_j, st_j, s_idx, seg.max_block_count,
                    policy_name=policy_name, search_stats=result.stats, **eval_kwargs
                )
                iterations += 1
                result.stats.update(full_r)
                if not full_r.success:
                    schedulable = False
                    result.error = full_r.error or "full-split measured timing failed"
                    _restore_segment_state(st_j, s_idx, saved_state)
                    break
                if seg.G_block_list and max(seg.G_block_list) > cur_tol:
                    # Even full split can't meet tolerance for this task
                    schedulable = False
                # Restore: we just probed, now select optimal K
                _restore_segment_state(st_j, s_idx, saved_state)

        if not schedulable:
            break

        # For each lower-priority task, find minimum K satisfying tolerance
        for j in range(1, i + 1):
            task_j = sorted_task_list[j]
            for s_idx, seg in enumerate(task_j.inference_segment_list):
                N_j = seg.max_block_count
                dt_j, st_j = task_map[str(task_j.id)]
                max_policy_k = _policy_max_chunks(dt_j, seg, policy_name)
                best_k = None
                best_max_block = math.inf

                if (
                    exact_opt_max_boundaries > 0
                    and N_j - 1 <= exact_opt_max_boundaries
                ):
                    # Exhaustive: enumerate all 2^(N-1) masks
                    from src.integration.split_point_policy import (
                        apply_policy_to_mask, get_enabled_boundaries,
                    )
                    enabled = get_enabled_boundaries(
                        dt_j.model_name, policy_name, N_j - 1
                    )
                    seen_masks = set()
                    for bits in range(1 << (N_j - 1)):
                        mask = [(bits >> k) & 1 for k in range(N_j - 1)]
                        mask = apply_policy_to_mask(mask, enabled)
                        key = tuple(mask)
                        if key in seen_masks:
                            continue
                        seen_masks.add(key)
                        app_r = evaluate_and_apply_mask(dt_j, st_j, mask, s_idx, **eval_kwargs)
                        iterations += 1
                        result.stats.update(app_r)
                        if app_r.success and app_r.max_block <= cur_tol:
                            if app_r.max_block < best_max_block:
                                best_max_block = app_r.max_block
                                best_k = sum(mask) + 1
                                # Keep this split applied
                                # (apply_mask already patched seg_task in place)
                    if best_k is None:
                        schedulable = False
                else:
                    # Sweep K = 1 .. N_j using measured-best K masks
                    for k in range(1, max_policy_k + 1):
                        app_r = apply_k_chunks(
                            dt_j, st_j, s_idx, k,
                            policy_name=policy_name, search_stats=result.stats, **eval_kwargs
                        )
                        iterations += 1
                        result.stats.update(app_r)
                        if app_r.success and app_r.max_block <= cur_tol:
                            if best_k is None:
                                best_k = k  # smallest K that satisfies tolerance
                                best_max_block = app_r.max_block
                                break  # stop at first feasible K

                if best_k is None:
                    schedulable = False

        if not schedulable:
            break

        # Recompute R and tolerance after all lower tasks optimized
        R_list, tolerance_list = update_SS_R_list_and_tolerance_list(sorted_task_list, i)

    # Final RTA pass
    final_R_list = []
    for i, st in enumerate(sorted_task_list):
        R, B_hi, B_lo, I = get_SS_R(sorted_task_list, i, final_R_list)
        final_R_list.append(R)
        if R > st.D:
            schedulable = False
        result.task_results.append(_make_task_result(st, R, B_hi, B_lo, I, task_map))

    result.schedulable = schedulable
    result.algorithm_iterations = iterations


# ── SS: heu (paper-style greedy) ─────────────────────────────────────────────

def _run_ss_heu_paper(sorted_task_list, task_map, result, eval_kwargs,
                      policy_name, max_profiles):
    """
    Paper-style SS-HEU (follows RTA_SS_heu from analysis.py).

    For each lower-priority task i=1..n-1:
      1. Probe full-split; if max_G_block > tolerance → infeasible.
      2. Greedy search: add one boundary at a time, picking the extension with
         minimum max_G_block each round, until tolerance met.
      3. Apply best config found, update R_list and tolerance_list.

    Split-point policy restricts which boundaries may be active.
    """
    from src.integration.split_point_policy import get_enabled_boundaries, apply_policy_to_mask
    from src.integration.paper_style_search import search_heuristic_ss_mask

    if not _apply_no_split_measured_to_all(sorted_task_list, task_map, result, eval_kwargs):
        return

    n = len(sorted_task_list)
    R_list = []
    tolerance_list = []
    schedulable = True
    iterations = 0

    # Task 0: compute R and tolerance, no splitting
    task_0 = sorted_task_list[0]
    R_0, B_hi, B_lo, I_0 = get_SS_R(sorted_task_list, 0, R_list)
    iterations += 1
    tol_0 = get_SS_tolerance(task_0, task_0.D, task_0.C, task_0.G, I_0, B_hi)
    tolerance_list.append(tol_0)
    R_list.append(R_0)

    for i in range(1, n):
        task_i = sorted_task_list[i]
        dt_i, st_i = task_map[str(task_i.id)]
        cur_tol = min(tolerance_list)
        seg_i = st_i.inference_segment_list[0]

        # Policy-limited full-split feasibility probe (Fix 3: only probe boundaries
        # reachable by the policy, not the unrestricted K=N mask).
        enabled = get_enabled_boundaries(
            dt_i.model_name, policy_name, seg_i.max_block_count - 1
        )
        policy_full_mask = apply_policy_to_mask(
            [1] * (seg_i.max_block_count - 1), enabled
        )
        probe_snapshot = _snapshot_segment_state(st_i, 0)
        dnn_probe_snapshot = (
            list(dt_i.current_chunk_times_ms),
            dt_i.selected_variant_name,
            dt_i.selected_config_path,
            dt_i.profile_result_path,
        )
        full_r = evaluate_and_apply_mask(dt_i, st_i, policy_full_mask, 0, **eval_kwargs)
        result.stats.update(full_r)
        iterations += 1
        if not full_r.success:
            schedulable = False
            result.error = full_r.error or "full-split measured timing failed"
            _restore_segment_state(st_i, 0, probe_snapshot)
            (
                dt_i.current_chunk_times_ms,
                dt_i.selected_variant_name,
                dt_i.selected_config_path,
                dt_i.profile_result_path,
            ) = dnn_probe_snapshot
            break
        full_split_max_block = st_i.max_G_block
        _restore_segment_state(st_i, 0, probe_snapshot)
        (
            dt_i.current_chunk_times_ms,
            dt_i.selected_variant_name,
            dt_i.selected_config_path,
            dt_i.profile_result_path,
        ) = dnn_probe_snapshot

        if full_split_max_block > cur_tol:
            schedulable = False
            break

        # Paper-style HEU search (enabled already computed above)
        search_result = search_heuristic_ss_mask(
            dt_i, st_i, 0, cur_tol, enabled, eval_kwargs, result.stats,
            max_profiles=max_profiles,
        )
        iterations += search_result.profiles_used
        if not search_result.found:
            schedulable = False
            break

        # Update R and tolerance
        R_list, tolerance_list = update_SS_R_list_and_tolerance_list(
            sorted_task_list, last_task_idx=i
        )

    # Final RTA pass
    final_R_list = []
    for i, st in enumerate(sorted_task_list):
        R, B_hi, B_lo, I = get_SS_R(sorted_task_list, i, final_R_list)
        final_R_list.append(R)
        if R > st.D:
            schedulable = False
            result.task_results.append(_make_task_result(st, R, B_hi, B_lo, I, task_map))
            result.schedulable = schedulable
            result.algorithm_iterations = iterations
            return
        result.task_results.append(_make_task_result(st, R, B_hi, B_lo, I, task_map))

    result.schedulable = schedulable
    result.algorithm_iterations = iterations


# ── SS: opt (paper-style BFS-OPT) ────────────────────────────────────────────

def _run_ss_opt_paper(sorted_task_list, task_map, result, eval_kwargs,
                      policy_name, max_profiles, max_candidates):
    """
    Paper-style SS-OPT (follows RTA_SS_opt from analysis.py).

    For each lower-priority task i=1..n-1:
      1. Probe full-split; if max_G_block > tolerance → infeasible.
      2. BFS over enabled-boundary subsets (starting from no-split).
         Expand non-feasible configs; record feasible ones.
         Pick the feasible config with minimum max_G_block (proxy for min WCET).
      3. Apply best config found, update R_list and tolerance_list.

    Split-point policy restricts which boundaries may be active.
    """
    from src.integration.split_point_policy import get_enabled_boundaries, apply_policy_to_mask
    from src.integration.paper_style_search import search_optimal_ss_mask

    if not _apply_no_split_measured_to_all(sorted_task_list, task_map, result, eval_kwargs):
        return

    n = len(sorted_task_list)
    R_list = []
    tolerance_list = []
    schedulable = True
    iterations = 0

    # Task 0: compute R and tolerance, no splitting
    task_0 = sorted_task_list[0]
    R_0, B_hi, B_lo, I_0 = get_SS_R(sorted_task_list, 0, R_list)
    iterations += 1
    tol_0 = get_SS_tolerance(task_0, task_0.D, task_0.C, task_0.G, I_0, B_hi)
    tolerance_list.append(tol_0)
    R_list.append(R_0)

    for i in range(1, n):
        task_i = sorted_task_list[i]
        dt_i, st_i = task_map[str(task_i.id)]
        cur_tol = min(tolerance_list)
        seg_i = st_i.inference_segment_list[0]

        # Policy-limited full-split feasibility probe (Fix 3: only probe boundaries
        # reachable by the policy, not the unrestricted K=N mask).
        enabled = get_enabled_boundaries(
            dt_i.model_name, policy_name, seg_i.max_block_count - 1
        )
        policy_full_mask = apply_policy_to_mask(
            [1] * (seg_i.max_block_count - 1), enabled
        )
        probe_snapshot = _snapshot_segment_state(st_i, 0)
        dnn_probe_snapshot = (
            list(dt_i.current_chunk_times_ms),
            dt_i.selected_variant_name,
            dt_i.selected_config_path,
            dt_i.profile_result_path,
        )
        full_r = evaluate_and_apply_mask(dt_i, st_i, policy_full_mask, 0, **eval_kwargs)
        result.stats.update(full_r)
        iterations += 1
        if not full_r.success:
            schedulable = False
            result.error = full_r.error or "full-split measured timing failed"
            _restore_segment_state(st_i, 0, probe_snapshot)
            (
                dt_i.current_chunk_times_ms,
                dt_i.selected_variant_name,
                dt_i.selected_config_path,
                dt_i.profile_result_path,
            ) = dnn_probe_snapshot
            break
        full_split_max_block = st_i.max_G_block
        _restore_segment_state(st_i, 0, probe_snapshot)
        (
            dt_i.current_chunk_times_ms,
            dt_i.selected_variant_name,
            dt_i.selected_config_path,
            dt_i.profile_result_path,
        ) = dnn_probe_snapshot

        if full_split_max_block > cur_tol:
            schedulable = False
            break

        # Paper-style BFS-OPT search (enabled already computed above)
        search_result = search_optimal_ss_mask(
            dt_i, st_i, 0, cur_tol, enabled, eval_kwargs, result.stats,
            max_profiles=max_profiles, max_candidates=max_candidates,
        )
        iterations += search_result.profiles_used
        if not search_result.found:
            schedulable = False
            break

        # Update R and tolerance
        R_list, tolerance_list = update_SS_R_list_and_tolerance_list(
            sorted_task_list, last_task_idx=i
        )

    # Final RTA pass
    final_R_list = []
    for i, st in enumerate(sorted_task_list):
        R, B_hi, B_lo, I = get_SS_R(sorted_task_list, i, final_R_list)
        final_R_list.append(R)
        if R > st.D:
            schedulable = False
            result.task_results.append(_make_task_result(st, R, B_hi, B_lo, I, task_map))
            result.schedulable = schedulable
            result.algorithm_iterations = iterations
            return
        result.task_results.append(_make_task_result(st, R, B_hi, B_lo, I, task_map))

    result.schedulable = schedulable
    result.algorithm_iterations = iterations


# ── UNI: single ──────────────────────────────────────────────────────────────

def _run_uni_single(sorted_task_list, task_map, result, eval_kwargs):
    """
    Convert SS→UNI, apply no-split in UNI space, run UNI RTA.

    No-split in UNI space: only the fixed CPU-GPU boundaries are active.
    K=1 uses the same measured evaluator/cache path as _run_ss_single.
    """
    if not _apply_no_split_measured_to_all(sorted_task_list, task_map, result, eval_kwargs):
        return

    uni_tasks = [convert_task_SS_to_UNI(deepcopy(st)) for st in sorted_task_list]

    schedulable = True
    for i, ut in enumerate(uni_tasks):
        R_i, K_i = get_UNI_R_and_K(uni_tasks, i)
        sched = R_i <= ut.D
        if not sched:
            schedulable = False
        # Find original SS task for reporting
        st_orig = sorted_task_list[i]
        R_list_stub = []
        for k in range(i + 1):
            R_k, B_hi_k, B_lo_k, I_k = get_SS_R(sorted_task_list, k, R_list_stub)
            R_list_stub.append(R_k)
        result.task_results.append(_make_task_result_from_uni(
            ut, sorted_task_list[i], R_i, task_map
        ))

    result.schedulable = schedulable
    result.algorithm_iterations = 1


# ── UNI: max ─────────────────────────────────────────────────────────────────

def _run_uni_max(sorted_task_list, task_map, result, eval_kwargs):
    """
    Apply measured K=N timing, convert SS→UNI, run UNI RTA.
    """
    for st in sorted_task_list:
        dt, _ = task_map[str(st.id)]
        r = apply_full_split_mask(dt, st, 0, **eval_kwargs)
        result.stats.update(r)
        if not r.success:
            result.error = r.error or f"full-split measured timing failed for task {st.id}"
            result.schedulable = False
            return
    if not _assert_no_placeholder_timing_reaches_rta(sorted_task_list, task_map, result, "UNI max initialization"):
        return

    uni_tasks = [convert_task_SS_to_UNI(deepcopy(st)) for st in sorted_task_list]

    schedulable = True
    for i, ut in enumerate(uni_tasks):
        R_i, K_i = get_UNI_R_and_K(uni_tasks, i)
        sched = R_i <= ut.D
        if not sched:
            schedulable = False
        result.task_results.append(_make_task_result_from_uni(
            ut, sorted_task_list[i], R_i, task_map
        ))

    result.schedulable = schedulable
    result.algorithm_iterations = 1


# ── UNI: tol-fb ──────────────────────────────────────────────────────────────

def _run_uni_tol_fb(sorted_task_list, task_map, result, eval_kwargs, max_iterations,
                    policy_name="all"):
    """
    DNN-aware UNI-tol-fb.

    Uses measured K-search. It still operates in K-space, but disabled policy
    boundaries are forced to 0.

    Convert SS→UNI. When the algorithm needs to split in UNI space at GPU
    boundaries: extract TRT mask, call evaluate_mask, reconstruct UNI G_block_list.

    UNI-to-TRT mapping:
      UNI base_block_list = [cpu_pre?, GPU_0..N-1, cpu_post?]
      Fixed boundaries (from fixed_one_indices) surround the GPU block.
      UNI GPU boundary k (1-indexed within GPU block) = TRT boundary k-1.

    After TRT evaluation with K-chunk mask, keep the simulation UNI structure:
      UNI G_block_list = [cpu_pre?, measured_gpu_chunk[0..K-1], cpu_post?]
    """
    if not _apply_no_split_measured_to_all(sorted_task_list, task_map, result, eval_kwargs):
        return

    # Convert to unified resource model
    uni_tasks = []
    for task in sorted_task_list:
        uni_tasks.append(convert_task_SS_to_UNI(deepcopy(task)))

    is_schedulable = True
    profiling_count = 0

    R_list = []
    tolerance_list = [math.inf for _ in range(len(uni_tasks))]

    i = 0
    while i < len(uni_tasks) and profiling_count < max_iterations:
        '''
        # Step 1: UNI RTA
        '''
        profiling_count += 1

        task_i = uni_tasks[i]
        D_i = task_i.D
        R_i, K_i = get_UNI_R_and_K(uni_tasks, i)

        is_last_task = (i == len(uni_tasks) - 1)

        # Update tolerance
        if not is_last_task:
            tolerance_i = get_UNI_tolerance(uni_tasks, i, K_i)
        else:
            tolerance_i = math.inf
        tolerance_list[i] = tolerance_i

        if R_i <= D_i: # Current task meets deadline
            if len(R_list) <= i:
                R_list.append(R_i)
            else:
                R_list[i] = R_i
            i += 1
            continue

        '''
        # Step 2: tolerance-fit splitting
        '''

        meet_tolerance = False
        while profiling_count < max_iterations:
            # If invalid, go to fallback
            target_tolerance = min(tolerance_list[: i+1])
            if is_last_task or tolerance_i <= 0:
                break

            # All lower tasks meet tolerance
            ss_task_view = convert_task_list_to_SS([deepcopy(t) for t in uni_tasks])
            if does_all_lower_meet_tolerance(ss_task_view,i,target_tolerance):
                meet_tolerance = True
                break

            # Find splitting target
            split_target = find_splitting_target(ss_task_view, i, target_tolerance)

            # No more splitting target
            if split_target is None:
                meet_tolerance = False
                break

            # Apply splitting
            target_task_index, target_segment_index, current_n = split_target
            dt, st_orig = task_map[str(sorted_task_list[target_task_index].id)]
            if current_n >= _policy_max_chunks(
                dt, st_orig.inference_segment_list[target_segment_index], policy_name
            ):
                is_split = False
            else:
                app_r = _uni_apply_k_chunks(
                    dt, st_orig, uni_tasks[target_task_index],
                    target_segment_index, current_n + 1, eval_kwargs,
                    policy_name=policy_name, search_stats=result.stats,
                )
                result.stats.update(app_r)
                is_split = app_r.success

            # Not splitable
            if not is_split:
                meet_tolerance = False
                break

            profiling_count += 1

            # Update R_list and tolerance_list
            R_list, new_tolerance_list = update_UNI_R_list_and_tolerance_list(uni_tasks, last_task_idx=i)
            tolerance_list[: i + 1] = new_tolerance_list

        if meet_tolerance:
            if len(R_list) <= i:
                R_list.append(R_i)
            i += 1
            continue

        '''
        # Step 3: Fallback
        '''
        ss_task_view = convert_task_list_to_SS([deepcopy(t) for t in uni_tasks])
        is_splitted, splitted_task_idx, app_r = _dnn_uni_split_largest_excluding_highest(
            uni_tasks, ss_task_view, sorted_task_list, task_map, eval_kwargs,
            policy_name=policy_name, search_stats=result.stats,
        )
        if app_r is not None:
            result.stats.update(app_r)
        if not is_splitted or splitted_task_idx is None:
            is_schedulable = False
            break
        profiling_count += 1

        # Update task idx if splited task idx is higher
        restart_idx = i if i <= splitted_task_idx else splitted_task_idx

        # Update R_list and tolerance_list
        if R_list:
            R_list, new_tolerance_list = update_UNI_R_list_and_tolerance_list(uni_tasks, last_task_idx=(len(R_list)-1))
            tolerance_list[: len(new_tolerance_list)] = new_tolerance_list
        else:
            R_list = []

        '''
        # Step 4: Early stop
        '''
        optimistic_R_list = get_optimistic_UNI_R(uni_tasks)
        for k in range(len(optimistic_R_list)):
            task_k = uni_tasks[k]
            D_k = task_k.D
            R_k = optimistic_R_list[k]
            if D_k < R_k:
                is_schedulable = False
                result.schedulable = is_schedulable
                result.algorithm_iterations = profiling_count
                return

        # Detect not schedulable
        if not is_schedulable:
            break

        # Reset target index
        i = restart_idx

    if profiling_count >= max_iterations:
        result.error = f"Max iterations ({max_iterations}) reached"

    # Final validation is required for the TensorRT-backed UNI bridge.  The
    # tolerance loop can certify lower-priority blocking bounds while an already
    # visited lower task still misses its own response-time deadline under the
    # measured split timings.
    for i, ut in enumerate(uni_tasks):
        R_i, K_i = get_UNI_R_and_K(uni_tasks, i)
        if R_i > ut.D:
            is_schedulable = False
        result.task_results.append(_make_task_result_from_uni(
            ut, sorted_task_list[i], R_i, task_map
        ))

    result.schedulable = is_schedulable
    result.algorithm_iterations = profiling_count


# ── UNI: tol ─────────────────────────────────────────────────────────────────

def _run_uni_tol(sorted_task_list, task_map, result, eval_kwargs, max_iterations,
                 policy_name="all"):
    """
    DNN-aware UNI-tol (no fallback).

    Uses measured K-search. It still operates in K-space, but disabled policy
    boundaries are forced to 0.

    Converts SS→UNI at start. When splitting is needed:
      finds lower task via SS-compatible split_target, applies UNI split.

    Follows SS-tol logic but using UNI RTA functions (get_UNI_R_and_K,
    get_UNI_tolerance, update_UNI_R_list_and_tolerance_list).
    """
    if not _apply_no_split_measured_to_all(sorted_task_list, task_map, result, eval_kwargs):
        return

    from copy import deepcopy
    uni_tasks = [convert_task_SS_to_UNI(deepcopy(st)) for st in sorted_task_list]

    n = len(uni_tasks)
    R_list = []
    tolerance_list = [math.inf] * n
    schedulable = True
    iterations = 0

    for i in range(n):
        ut = uni_tasks[i]
        D_i = ut.D
        is_last = (i == n - 1)

        R_i, K_i = get_UNI_R_and_K(uni_tasks, i)
        iterations += 1

        tolerance_i = get_UNI_tolerance(uni_tasks, i, K_i) if not is_last else math.inf
        tolerance_list[i] = tolerance_i

        if R_i <= D_i:
            R_list.append(R_i)
            continue

        if is_last or tolerance_i <= 0:
            schedulable = False
            break

        meet_tolerance = False
        while iterations < max_iterations:
            target_tol = min(tolerance_list[:i + 1])

            ss_tasks_tmp = convert_task_list_to_SS([deepcopy(t) for t in uni_tasks])
            if does_all_lower_meet_tolerance(ss_tasks_tmp, i, target_tol):
                meet_tolerance = True
                break

            split_target = find_splitting_target(ss_tasks_tmp, i, target_tol)
            if split_target is None:
                break

            t_idx, s_idx, cur_n = split_target
            dt, st_orig = task_map[str(sorted_task_list[t_idx].id)]
            ut_target = uni_tasks[t_idx]
            if cur_n >= _policy_max_chunks(
                dt, st_orig.inference_segment_list[s_idx], policy_name
            ):
                break

            new_k = cur_n + 1
            app_r = _uni_apply_k_chunks(
                dt, st_orig, ut_target, s_idx, new_k, eval_kwargs,
                policy_name=policy_name, search_stats=result.stats,
            )
            iterations += 1
            result.stats.update(app_r)
            if not app_r.success:
                break

            R_list, new_tol = update_UNI_R_list_and_tolerance_list(uni_tasks, i)
            tolerance_list[:i + 1] = new_tol

        if not meet_tolerance:
            schedulable = False
        elif len(R_list) <= i:
            R_list.append(R_i)

        if not schedulable:
            break

    # Final RTA pass
    for i, ut in enumerate(uni_tasks):
        R_i, K_i = get_UNI_R_and_K(uni_tasks, i)
        if R_i > ut.D:
            schedulable = False
        result.task_results.append(_make_task_result_from_uni(
            ut, sorted_task_list[i], R_i, task_map
        ))

    result.schedulable = schedulable
    result.algorithm_iterations = iterations


# ── UNI: heu (paper-style greedy) ────────────────────────────────────────────

def _run_uni_heu_paper(sorted_task_list, task_map, result, eval_kwargs,
                       policy_name, max_profiles):
    """
    Paper-style UNI-HEU (follows RTA_UNI_heu from analysis.py).

    For each lower-priority task i=1..n-1:
      1. Probe full-split in UNI space; if max_G_block > tolerance → infeasible.
      2. Greedy TRT boundary search: add one GPU boundary at a time,
         picking the extension with minimum max_G_block each round.
      3. Apply best config, update UNI R_list and tolerance_list.
    """
    from copy import deepcopy
    from src.integration.split_point_policy import get_enabled_boundaries
    from src.integration.paper_style_search import search_heuristic_uni_mask

    if not _apply_no_split_measured_to_all(sorted_task_list, task_map, result, eval_kwargs):
        return

    uni_tasks = [convert_task_SS_to_UNI(deepcopy(st)) for st in sorted_task_list]

    n = len(uni_tasks)
    R_list = []
    tolerance_list = []
    schedulable = True
    iterations = 0

    # Task 0: compute R and tolerance, no splitting
    R_0, K_0 = get_UNI_R_and_K(uni_tasks, 0)
    iterations += 1
    tol_0 = get_UNI_tolerance(uni_tasks, 0, K_0)
    tolerance_list.append(tol_0)

    for i in range(1, n):
        ut = uni_tasks[i]
        dt_i, st_orig_i = task_map[str(sorted_task_list[i].id)]
        cur_tol = min(tolerance_list)
        seg_orig = st_orig_i.inference_segment_list[0]
        N_gpu = len(seg_orig.base_block_list)
        enabled = get_enabled_boundaries(dt_i.model_name, policy_name, N_gpu - 1)

        # Full-split probe with measured TRT timing.
        ok_full, ut_full, full_r = _measured_full_split_uni_probe(
            dt_i, st_orig_i, eval_kwargs, enabled_boundaries=enabled
        )
        result.stats.update(full_r)
        iterations += 1
        if not ok_full:
            schedulable = False
            break

        if ut_full.max_G_block > cur_tol:
            schedulable = False
            break

        search_result = search_heuristic_uni_mask(
            dt_i, st_orig_i, ut, 0, cur_tol, enabled, eval_kwargs, result.stats,
            max_profiles=max_profiles,
        )
        iterations += search_result.profiles_used
        if not search_result.found:
            schedulable = False
            break

        # Sync: convert_task_list_to_UNI reads from SS tasks
        # (st_orig_i is already patched by _uni_apply_raw_mask)
        # Rebuild uni_tasks[i] from updated st_orig_i
        uni_tasks[i] = convert_task_SS_to_UNI(deepcopy(st_orig_i))
        _sync_uni_from_search(uni_tasks[i], ut)

        R_i, K_i = get_UNI_R_and_K(uni_tasks, i)
        tol_i = get_UNI_tolerance(uni_tasks, i, K_i)
        tolerance_list.append(tol_i)

    # Final RTA pass
    for i, ut in enumerate(uni_tasks):
        R_i, K_i = get_UNI_R_and_K(uni_tasks, i)
        if R_i > ut.D:
            schedulable = False
            result.task_results.append(_make_task_result_from_uni(
                ut, sorted_task_list[i], R_i, task_map
            ))
            result.schedulable = schedulable
            result.algorithm_iterations = iterations
            return
        result.task_results.append(_make_task_result_from_uni(
            ut, sorted_task_list[i], R_i, task_map
        ))

    result.schedulable = schedulable
    result.algorithm_iterations = iterations


# ── UNI: opt (paper-style BFS-OPT) ───────────────────────────────────────────

def _run_uni_opt_paper(sorted_task_list, task_map, result, eval_kwargs,
                       policy_name, max_profiles, max_candidates):
    """
    Paper-style UNI-OPT (follows RTA_UNI_opt from analysis.py).

    For each lower-priority task i=1..n-1:
      1. Probe full-split; if max_G_block > tolerance → infeasible.
      2. BFS over enabled TRT GPU boundary subsets.
         Pick the feasible config with minimum max_G_block.
      3. Apply best config, update UNI R_list and tolerance_list.
    """
    from copy import deepcopy
    from src.integration.split_point_policy import get_enabled_boundaries
    from src.integration.paper_style_search import search_optimal_uni_mask

    if not _apply_no_split_measured_to_all(sorted_task_list, task_map, result, eval_kwargs):
        return

    uni_tasks = [convert_task_SS_to_UNI(deepcopy(st)) for st in sorted_task_list]

    n = len(uni_tasks)
    R_list = []
    tolerance_list = []
    schedulable = True
    iterations = 0

    # Task 0: no splitting
    R_0, K_0 = get_UNI_R_and_K(uni_tasks, 0)
    iterations += 1
    tol_0 = get_UNI_tolerance(uni_tasks, 0, K_0)
    tolerance_list.append(tol_0)

    for i in range(1, n):
        ut = uni_tasks[i]
        dt_i, st_orig_i = task_map[str(sorted_task_list[i].id)]
        cur_tol = min(tolerance_list)
        seg_orig = st_orig_i.inference_segment_list[0]
        N_gpu = len(seg_orig.base_block_list)
        enabled = get_enabled_boundaries(dt_i.model_name, policy_name, N_gpu - 1)

        # Full-split probe with measured TRT timing.
        ok_full, ut_full, full_r = _measured_full_split_uni_probe(
            dt_i, st_orig_i, eval_kwargs, enabled_boundaries=enabled
        )
        result.stats.update(full_r)
        iterations += 1
        if not ok_full:
            schedulable = False
            break

        if ut_full.max_G_block > cur_tol:
            schedulable = False
            break

        search_result = search_optimal_uni_mask(
            dt_i, st_orig_i, ut, 0, cur_tol, enabled, eval_kwargs, result.stats,
            max_profiles=max_profiles, max_candidates=max_candidates,
        )
        iterations += search_result.profiles_used
        if not search_result.found:
            schedulable = False
            break

        # Rebuild uni_tasks[i] from patched st_orig_i
        uni_tasks[i] = convert_task_SS_to_UNI(deepcopy(st_orig_i))
        _sync_uni_from_search(uni_tasks[i], ut)

        R_i, K_i = get_UNI_R_and_K(uni_tasks, i)
        tol_i = get_UNI_tolerance(uni_tasks, i, K_i)
        tolerance_list.append(tol_i)

    # Final RTA pass
    for i, ut in enumerate(uni_tasks):
        R_i, K_i = get_UNI_R_and_K(uni_tasks, i)
        if R_i > ut.D:
            schedulable = False
            result.task_results.append(_make_task_result_from_uni(
                ut, sorted_task_list[i], R_i, task_map
            ))
            result.schedulable = schedulable
            result.algorithm_iterations = iterations
            return
        result.task_results.append(_make_task_result_from_uni(
            ut, sorted_task_list[i], R_i, task_map
        ))

    result.schedulable = schedulable
    result.algorithm_iterations = iterations


def _sync_uni_from_search(uni_task_new, ut_with_block_list) -> None:
    """
    After a UNI search, update uni_task_new's G_block_list from ut_with_block_list.

    This syncs the measured G_block_list (set by _uni_apply_raw_mask) into the
    freshly-converted UNI task.
    """
    for s_idx in range(len(uni_task_new.inference_segment_list)):
        if s_idx < len(ut_with_block_list.G_segment_list):
            uni_task_new.inference_segment_list[s_idx].G_block_list = list(
                ut_with_block_list.G_segment_list[s_idx]
            )
    uni_task_new.G = ut_with_block_list.G
    uni_task_new.max_G_block = ut_with_block_list.max_G_block
    uni_task_new.G_segment_list = [list(b) for b in ut_with_block_list.G_segment_list]


# ── DNN-aware split helpers ───────────────────────────────────────────────────

def _measured_full_split_uni_probe(
    dnn_task, st_orig, eval_kwargs, enabled_boundaries=None
):
    """
    Return a UNI full-split probe using measured TRT full-split timing.

    The probe operates on copies so paper-style UNI HEU/OPT feasibility checks do
    not mutate the canonical SS/UNI task state.
    """
    dt_probe = deepcopy(dnn_task)
    st_probe = deepcopy(st_orig)
    if enabled_boundaries is None:
        app_r = apply_full_split_mask(dt_probe, st_probe, 0, **eval_kwargs)
    else:
        boundary_count = len(st_probe.inference_segment_list[0].base_block_list) - 1
        full_mask = [1 if idx in set(enabled_boundaries) else 0 for idx in range(boundary_count)]
        app_r = evaluate_and_apply_mask(dt_probe, st_probe, full_mask, 0, **eval_kwargs)
    if not app_r.success:
        return False, None, app_r
    return True, convert_task_SS_to_UNI(st_probe), app_r

def _dnn_apply_k_split(
    dnn_task, seg_task, segment_idx, k, eval_kwargs, policy_name="all"
) -> MaskApplicationResult:
    """
    DNN-aware replacement for SegInfTask.split_segment(segment_idx, k).

    Computes the measured-best K-chunk mask via TRT/cache and patches the segment.
    """
    return apply_k_chunks(
        dnn_task, seg_task, segment_idx, k,
        policy_name=policy_name, **eval_kwargs
    )


def _dnn_split_largest_excluding_highest(
    sorted_task_list,
    task_map: dict,
    eval_kwargs: dict,
    policy_name: str = "all",
    search_stats=None,
) -> Tuple[bool, Optional[int], Optional[MaskApplicationResult]]:
    """
    DNN-aware replacement for split_largest_block_excluding_highest().

    Finds the task (excluding index 0 = highest priority) with the largest
    max_G_block that can still be split, then increases its K by 1.
    """
    best = None
    for task_idx in range(1, len(sorted_task_list)):
        st = sorted_task_list[task_idx]
        for s_idx, seg in enumerate(st.inference_segment_list):
            dt, _ = task_map[str(st.id)]
            if seg.size >= _policy_max_chunks(dt, seg, policy_name):
                continue
            cur_max = max(seg.G_block_list) if seg.G_block_list else 0.0
            if best is None or cur_max > best[0]:
                best = (cur_max, task_idx, s_idx, seg.size)

    if best is None:
        return False, None, None

    _, task_idx, s_idx, cur_n = best
    target_st = sorted_task_list[task_idx]
    dt, _ = task_map[str(target_st.id)]
    app_r = apply_k_chunks(
        dt, target_st, s_idx, cur_n + 1,
        policy_name=policy_name, search_stats=search_stats, **eval_kwargs
    )
    return app_r.success, task_idx, app_r


def _dnn_uni_split_largest_excluding_highest(
    uni_tasks,
    ss_view_task_list,
    original_sorted_task_list,
    task_map: dict,
    eval_kwargs: dict,
    policy_name: str = "all",
    search_stats=None,
) -> Tuple[bool, Optional[int], Optional[MaskApplicationResult]]:
    """
    UNI fallback equivalent of _dnn_split_largest_excluding_highest().

    The largest-block decision is made on a disposable SS view, but the split is
    applied directly to the canonical UNI task.  This preserves measured TRT
    chunk timings when choosing the next split.
    """
    best = None
    for task_idx in range(1, len(ss_view_task_list)):
        st_view = ss_view_task_list[task_idx]
        original_st = original_sorted_task_list[task_idx]
        dt, st_orig = task_map[str(original_st.id)]
        for s_idx, seg_view in enumerate(st_view.inference_segment_list):
            seg_orig = st_orig.inference_segment_list[s_idx]
            if seg_view.size >= _policy_max_chunks(dt, seg_orig, policy_name):
                continue
            cur_max = max(seg_view.G_block_list) if seg_view.G_block_list else 0.0
            if best is None or cur_max > best[0]:
                best = (cur_max, task_idx, s_idx, seg_view.size)

    if best is None:
        return False, None, None

    _, task_idx, s_idx, cur_n = best
    original_st = original_sorted_task_list[task_idx]
    dt, st_orig = task_map[str(original_st.id)]
    app_r = _uni_apply_k_chunks(
        dt, st_orig, uni_tasks[task_idx], s_idx, cur_n + 1, eval_kwargs,
        policy_name=policy_name, search_stats=search_stats,
    )
    return app_r.success, task_idx, app_r


def _uni_apply_k_chunks(
    dnn_task,
    st_orig,        # original SS SegInfTask (has base_block_list with N GPU chunks)
    ut_target,      # canonical UNI task to patch with reconstructed G blocks
    s_idx: int,
    new_k: int,
    eval_kwargs: dict,
    policy_name: str = "all",
    search_stats=None,
) -> MaskApplicationResult:
    """
    Apply a K-chunk split in UNI space with TRT evaluation.

    Extracts the TRT GPU mask, evaluates it, and reconstructs the UNI
    G_block_list with CPU pre/post as separate fixed UNI blocks.
    """
    app_r = apply_k_chunks(
        dnn_task, st_orig, s_idx, new_k, policy_name=policy_name,
        search_stats=search_stats, **eval_kwargs
    )

    if not app_r.success:
        return app_r

    _patch_uni_measured_blocks(
        dnn_task, ut_target, s_idx, app_r.mask, app_r.selected_chunk_times
    )

    return app_r


def _uni_apply_raw_mask(
    dnn_task,
    st_orig,        # original SS SegInfTask (has base_block_list with N GPU chunks)
    ut_target,      # UNI task to be patched with reconstructed G_block_list
    s_idx: int,
    trt_mask: List[int],   # raw TRT GPU boundary mask (N_gpu-1 bits)
    eval_kwargs: dict,
) -> MaskApplicationResult:
    """
    Apply a raw TRT boundary mask in UNI space with TRT evaluation.

    Evaluates the given TRT mask on st_orig, then reconstructs ut_target's
    G_block_list with CPU pre/post as separate fixed UNI blocks.

    Used by paper-style UNI search functions (search_optimal_uni_mask,
    search_heuristic_uni_mask) which generate masks directly rather than
    going through measured K-search.
    """
    # Apply mask to SS task (st_orig has the GPU base blocks)
    app_r = evaluate_and_apply_mask(
        dnn_task, st_orig, trt_mask, s_idx, **eval_kwargs
    )

    if not app_r.success:
        return app_r

    _patch_uni_measured_blocks(
        dnn_task, ut_target, s_idx, trt_mask, app_r.selected_chunk_times
    )

    return app_r


def _uni_g_blocks_from_measured_chunks(
    dnn_task, measured_chunks: List[float]
) -> List[float]:
    """
    Build a measured UNI block list using the same C/G separation as
    SegInfTask.convert_SS_to_UNI().

    Positive CPU pre/post blocks remain separate fixed UNI blocks.  Measured TRT
    GPU chunks replace the current GPU groups, but are not merged with CPU.
    """
    blocks: List[float] = []
    cpu_pre = float(getattr(dnn_task, "cpu_pre_ms", 0.0))
    cpu_post = float(getattr(dnn_task, "cpu_post_ms", 0.0))
    if cpu_pre > 0.0:
        blocks.append(cpu_pre)
    blocks.extend(float(t) for t in measured_chunks)
    if cpu_post > 0.0:
        blocks.append(cpu_post)
    if not blocks:
        blocks.append(0.0)
    return blocks


def _patch_uni_measured_blocks(
    dnn_task,
    ut_target,
    s_idx: int,
    trt_mask: List[int],
    measured_chunks: List[float],
) -> None:
    """
    Patch a UNI task with measured TRT GPU timings without changing the
    simulation UNI/SS block semantics.
    """
    uni_g_list = _uni_g_blocks_from_measured_chunks(dnn_task, measured_chunks)
    seg_ut = ut_target.inference_segment_list[s_idx]
    seg_ut.splitting_config = _uni_config_from_trt_mask(ut_target, trt_mask)
    seg_ut.G_block_list = list(uni_g_list)
    ut_target.G_segment_list[s_idx] = list(uni_g_list)
    ut_target.G = sum(sum(b) for b in ut_target.G_segment_list)
    ut_target.max_G_block = max(
        (max(b) for b in ut_target.G_segment_list if b), default=0.0
    )


def _uni_config_from_trt_mask(uni_task, trt_mask: List[int]) -> List[int]:
    """
    Expand a TRT GPU-boundary mask into UNI segment boundary space.

    UNI base blocks may include positive CPU pre/post blocks.  Boundaries touching
    CPU blocks are fixed split points and have no TRT counterpart, so storing the
    raw TRT mask directly in a UNI InferenceSegment can make later UNI->SS
    conversion index past the end of splitting_config.
    """
    seg = uni_task.inference_segment_list[0]
    block_sources = getattr(uni_task, "_UNI_block_sources", None)
    if not block_sources:
        if len(trt_mask) == max(seg.max_block_count - 1, 0):
            return list(trt_mask)
        return list(getattr(uni_task, "non_splitting_config", []))

    config: List[int] = []
    for boundary_idx in range(max(len(block_sources) - 1, 0)):
        left = block_sources[boundary_idx]
        right = block_sources[boundary_idx + 1]
        if left[0] == "C" or right[0] == "C":
            config.append(1)
        elif left[0] == "G" and right[0] == "G" and left[1] == right[1]:
            trt_idx = left[2]
            config.append(int(trt_mask[trt_idx]) if trt_idx < len(trt_mask) else 0)
        else:
            config.append(1)
    return config


# ── Reporting helpers ─────────────────────────────────────────────────────────

def _make_task_result(st, R, B_hi, B_lo, I, task_map) -> TaskResult:
    dt, _ = task_map[str(st.id)]
    seg = st.inference_segment_list[0] if st.inference_segment_list else None
    mask = list(seg.splitting_config) if seg else []
    k = len(seg.G_block_list) if seg else 0
    chunk_times = list(seg.G_block_list) if seg else []
    return TaskResult(
        task_name=str(st.id),
        model_name=dt.model_name,
        cpu_id=dt.cpu_id,
        period_ms=float(st.T),
        deadline_ms=float(st.D),
        C_ms=float(st.C),
        G_ms=float(st.G),
        R_ms=float(R),
        slack_ms=float(st.D - R),
        schedulable=(R <= st.D),
        m=int(st.m),
        max_G_block=float(st.max_G_block),
        B_high=float(B_hi),
        B_low=float(B_lo),
        I=float(I),
        final_mask=mask,
        final_k_chunks=k,
        final_chunk_times_ms=chunk_times,
        variant_name=dt.selected_variant_name or "",
        profile_result_path=dt.profile_result_path or "",
    )


def _make_task_result_from_uni(ut, st_orig, R_uni, task_map) -> TaskResult:
    """Build a TaskResult for a UNI task (use original SS task for metadata)."""
    dt, _ = task_map[str(st_orig.id)]
    seg = ut.inference_segment_list[0] if ut.inference_segment_list else None
    mask = _extract_trt_mask_from_uni_task(ut)
    k = sum(mask) + 1 if mask else 1
    chunk_times = list(seg.G_block_list) if seg else []
    return TaskResult(
        task_name=str(st_orig.id),
        model_name=dt.model_name,
        cpu_id=dt.cpu_id,
        period_ms=float(st_orig.T),
        deadline_ms=float(st_orig.D),
        C_ms=float(ut.C),
        G_ms=float(ut.G),
        R_ms=float(R_uni),
        slack_ms=float(st_orig.D - R_uni),
        schedulable=(R_uni <= st_orig.D),
        m=int(ut.m),
        max_G_block=float(ut.max_G_block),
        B_high=0.0,
        B_low=0.0,
        I=0.0,
        final_mask=mask,
        final_k_chunks=k,
        final_chunk_times_ms=chunk_times,
        variant_name=dt.selected_variant_name or "",
        profile_result_path=dt.profile_result_path or "",
    )


def _extract_trt_mask_from_uni_task(ut) -> List[int]:
    """
    Extract the original TRT GPU-boundary mask from a UNI-converted task.

    UNI conversion inserts fixed CPU/GPU boundaries around CPU pre/post blocks.
    Those fixed boundaries are not TensorRT split points and should not be
    counted as DNN split activity.
    """
    if not ut.inference_segment_list:
        return []
    seg = ut.inference_segment_list[0]
    sources = getattr(ut, "_UNI_block_sources", None)
    if not sources:
        return list(getattr(seg, "splitting_config", []))

    gpu_positions: Dict[Tuple[int, int], int] = {}
    max_block_idx_by_segment: Dict[int, int] = {}
    for pos, source in enumerate(sources):
        if source[0] != "G":
            continue
        seg_idx = int(source[1])
        block_idx = int(source[2])
        gpu_positions[(seg_idx, block_idx)] = pos
        max_block_idx_by_segment[seg_idx] = max(
            max_block_idx_by_segment.get(seg_idx, -1), block_idx
        )

    mask: List[int] = []
    split_config = list(getattr(seg, "splitting_config", []))
    for seg_idx in sorted(max_block_idx_by_segment):
        for block_idx in range(max_block_idx_by_segment[seg_idx]):
            left_pos = gpu_positions.get((seg_idx, block_idx))
            right_pos = gpu_positions.get((seg_idx, block_idx + 1))
            if left_pos is None or right_pos is None:
                continue
            if right_pos == left_pos + 1 and left_pos < len(split_config):
                mask.append(int(split_config[left_pos]))
            else:
                mask.append(1)
    return mask
