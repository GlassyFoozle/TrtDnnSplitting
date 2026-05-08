"""
src/rta — Real-time analysis (RTA) functions for SS and UNI scheduling models.

Adapted from the DNNSplitting research codebase. Formulas are preserved exactly.
Only the import path for InferenceSegment was updated from bare `task` to
`src.rta.task` so the module works as a standalone package.

Reference: DNNSplitting paper (see README.md).
"""
from src.rta.task import InferenceSegment, SegInfTask
from src.rta.analysis import (
    NumeratorExplosionError,
    sort_task_set,
    get_SS_R,
    get_SS_tolerance,
    get_UNI_R_and_K,
    get_UNI_tolerance,
    get_max_lower_blocking,
    does_all_lower_meet_tolerance,
    find_splitting_target,
    update_SS_R_list_and_tolerance_list,
    update_UNI_R_list_and_tolerance_list,
    convert_task_SS_to_UNI,
    convert_task_UNI_to_SS,
    convert_task_list_to_SS,
    convert_task_list_to_UNI,
    split_largest_block_excluding_highest,
    split_by_config,
    get_optimistic_SS_R,
    get_optimistic_UNI_R,
)

__all__ = [
    "InferenceSegment", "SegInfTask",
    "NumeratorExplosionError",
    "sort_task_set",
    "get_SS_R", "get_SS_tolerance",
    "get_UNI_R_and_K", "get_UNI_tolerance",
    "get_max_lower_blocking",
    "does_all_lower_meet_tolerance",
    "find_splitting_target",
    "update_SS_R_list_and_tolerance_list",
    "update_UNI_R_list_and_tolerance_list",
    "convert_task_SS_to_UNI", "convert_task_UNI_to_SS",
    "convert_task_list_to_SS", "convert_task_list_to_UNI",
    "split_largest_block_excluding_highest",
    "split_by_config",
    "get_optimistic_SS_R", "get_optimistic_UNI_R",
]
