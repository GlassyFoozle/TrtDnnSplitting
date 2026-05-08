"""
test_rta_import.py — Verify src/rta is self-contained (no external DNNSplitting dep).
"""
import sys
import types


def test_no_sys_path_insertion(monkeypatch):
    """Importing src.rta must not insert any path containing 'DNNSplitting'."""
    before = list(sys.path)
    import src.rta  # noqa: F401
    added = [p for p in sys.path if p not in before]
    dnnsplit_paths = [p for p in added if "DNNSplitting" in p]
    assert not dnnsplit_paths, (
        f"src.rta import inserted DNNSplitting path(s): {dnnsplit_paths}"
    )


def test_rta_core_imports():
    """All public RTA symbols must be importable from src.rta."""
    from src.rta import (
        InferenceSegment, SegInfTask,
        get_SS_R, get_SS_tolerance,
        get_UNI_R_and_K, get_UNI_tolerance,
        sort_task_set,
        does_all_lower_meet_tolerance,
        find_splitting_target,
        update_SS_R_list_and_tolerance_list,
        update_UNI_R_list_and_tolerance_list,
        NumeratorExplosionError,
    )
    for sym in [
        InferenceSegment, SegInfTask,
        get_SS_R, get_SS_tolerance,
        get_UNI_R_and_K, get_UNI_tolerance,
        sort_task_set,
    ]:
        assert callable(sym) or isinstance(sym, type), f"{sym} not callable/type"


def test_inference_segment_basic():
    """InferenceSegment can be constructed and block list computed."""
    from src.rta.task import InferenceSegment
    seg = InferenceSegment(G_segment=10, max_block_count=4, per_splitting_overhead=0.0)
    assert seg.G_segment == 10
    assert seg.max_block_count == 4
    assert len(seg.base_block_list) == 4
    assert abs(sum(seg.base_block_list) - 10) < 1e-9


def test_seg_inf_task_basic():
    """SegInfTask can be built and reports is_valid()."""
    from src.rta.task import SegInfTask
    segment_list = [
        {"C": 1.0, "G_segment": 4, "max_block_count": 4, "per_splitting_overhead": 0.0},
        {"C": 0.5, "G_segment": 0, "max_block_count": 1, "per_splitting_overhead": 0.0},
    ]
    task = SegInfTask(
        id="t1",
        segment_list=segment_list,
        period=20.0,
        deadline=20.0,
        priority=1.0,
        cpu=0,
    )
    assert task.is_valid()
    assert task.G > 0
