"""
src/integration — Bridge between TensorRT profiling and RTA scheduling analysis.

Converts TensorRT profiling results (CandidateSpace, EvaluationResult) into
src.rta SegInfTask objects for SS/UNI schedulability analysis.

RTA is self-contained in src/rta/; no external DNNSplitting repo is required.
"""
from src.integration.dnn_task import DNNBackedTask
from src.integration.dnnsplitting_adapter import dnn_task_to_seginftask, get_dnnsplitting_dir, build_task_set_dict
from src.integration.dnn_taskset_loader import load_dnn_taskset, validate_dnn_taskset_json
from src.integration.dnn_taskset_generator import generate_dnn_taskset
from src.integration.mask_applicator import (
    MaskApplicationResult, evaluate_and_apply_mask,
    apply_no_split_mask, apply_full_split_mask, apply_k_chunks,
)
from src.integration.dnn_algorithm_runner import (
    DNNAlgorithmResult, ProfilingStats, run_dnn_rta_algorithm,
)
from src.integration.split_point_policy import (
    get_enabled_boundaries, apply_policy_to_mask, policy_summary,
)
from src.integration.paper_style_search import (
    CandidateSearchResult, search_optimal_ss_mask, search_heuristic_ss_mask,
    search_optimal_uni_mask, search_heuristic_uni_mask,
)
from src.integration.dnn_workload_generator import (
    WorkloadConfig, generate_tasksets, uunifast,
)

__all__ = [
    "DNNBackedTask",
    "dnn_task_to_seginftask",
    "get_dnnsplitting_dir",
    "build_task_set_dict",
    "load_dnn_taskset",
    "validate_dnn_taskset_json",
    "generate_dnn_taskset",
    "MaskApplicationResult",
    "evaluate_and_apply_mask",
    "apply_no_split_mask",
    "apply_full_split_mask",
    "apply_k_chunks",
    "DNNAlgorithmResult",
    "ProfilingStats",
    "run_dnn_rta_algorithm",
    "get_enabled_boundaries",
    "apply_policy_to_mask",
    "policy_summary",
    "CandidateSearchResult",
    "search_optimal_ss_mask",
    "search_heuristic_ss_mask",
    "search_optimal_uni_mask",
    "search_heuristic_uni_mask",
    "WorkloadConfig",
    "generate_tasksets",
    "uunifast",
]
