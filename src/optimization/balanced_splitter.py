"""
balanced_splitter.py — Baseline balanced K-chunk splitter.

Given N base chunk timings t[0..N-1] and a desired K chunks, find K-1 active
boundaries that partition [0..N-1] into K contiguous groups minimizing the
maximum group execution time.

Algorithm: DP linear partitioning (classic "painters partition" problem).
  dp[i][k] = minimum possible "max group sum" when partitioning t[0:i] into k groups.
  dp[i][1] = prefix_sum[i]
  dp[i][k] = min over j in [k-1, i): max(dp[j][k-1], sum(t[j:i]))
  Answer: dp[N][K]
  Time: O(N^2 * K) — fine for N ≤ 46.

Edge cases:
  K=1  → all boundaries OFF (single chunk)
  K>=N → all boundaries ON (max granularity)

Result:
  mask:            binary list of length N-1
  groups:          list of lists of base chunk indices
  group_times_ms:  estimated execution time per group
  max_group_ms:    max group time (objective value)
  total_ms:        sum of all chunk times
  imbalance_ratio: max_group / (total / K) — 1.0 = perfectly balanced
  objective_score: max_group_ms (lower is better)

IMPORTANT: These times are ESTIMATES based on profiled base chunk latencies.
           Actual merged-chunk latency may differ due to kernel fusion.
           Use profiling after export/build for accurate final timing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SplitPlan:
    model_name: str
    target_k: int
    actual_k: int
    mask: List[int]
    groups: List[List[int]]
    group_times_ms: List[float]
    max_group_ms: float
    total_ms: float
    imbalance_ratio: float      # max_group / (total/K); 1.0 = perfect
    objective_score: float      # = max_group_ms
    has_timing: bool
    fallback_equal: bool = False  # True if equal-size fallback was used

    def active_boundaries(self) -> List[int]:
        return [i for i, b in enumerate(self.mask) if b == 1]

    def mask_str(self) -> str:
        return "".join(str(b) for b in self.mask)

    def summary(self) -> str:
        lines = [
            f"SplitPlan: {self.model_name}  K={self.actual_k}  "
            f"max_group={self.max_group_ms:.4f}ms  imbalance={self.imbalance_ratio:.3f}",
            f"  mask: {self.mask_str()}",
            f"  {'(equal-size fallback — no timing data)' if self.fallback_equal else ''}",
        ]
        avg = self.total_ms / self.actual_k if self.actual_k > 0 else 0.0
        for gi, (grp, t) in enumerate(zip(self.groups, self.group_times_ms)):
            ids_str = ",".join(str(i) for i in grp)
            lines.append(f"  group{gi}: [{ids_str}]  est={t:.4f}ms  (avg={avg:.4f}ms)")
        return "\n".join(lines)


def _prefix_sums(t: List[float]) -> List[float]:
    ps = [0.0] * (len(t) + 1)
    for i, v in enumerate(t):
        ps[i + 1] = ps[i] + v
    return ps


def balanced_split(
    chunk_times: List[float],
    target_k: int,
    model_name: str = "",
) -> SplitPlan:
    """
    Compute optimal K-chunk balanced partition.

    Parameters
    ----------
    chunk_times : per-base-chunk execution time estimates (may be all zeros)
    target_k    : desired number of output chunks
    model_name  : for reporting only

    Returns SplitPlan with mask, groups, and estimated group times.
    """
    n = len(chunk_times)
    has_timing = any(t > 0.0 for t in chunk_times)

    # Edge cases
    k = max(1, min(target_k, n))
    if k == 1:
        return _make_plan(model_name, k, [0] * (n - 1), [list(range(n))], chunk_times, has_timing)
    if k >= n:
        mask = [1] * (n - 1)
        groups = [[i] for i in range(n)]
        return _make_plan(model_name, k, mask, groups, chunk_times, has_timing)

    if not has_timing:
        return _equal_size_plan(chunk_times, k, model_name)

    # DP
    ps = _prefix_sums(chunk_times)

    def range_sum(lo: int, hi: int) -> float:
        return ps[hi] - ps[lo]

    INF = float("inf")
    # dp[i][j] = min max-group-sum to partition t[0:i] into j groups
    # split[i][j] = optimal split point (start of last group)
    dp = [[INF] * (k + 1) for _ in range(n + 1)]
    split = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for j in range(1, k + 1):
        for i in range(j, n + 1):
            best = INF
            best_s = j - 1
            for s in range(j - 1, i):
                val = max(dp[s][j - 1], range_sum(s, i))
                if val < best:
                    best = val
                    best_s = s
            dp[i][j] = best
            split[i][j] = best_s

    # Recover groups
    groups: List[List[int]] = []
    pos = n
    for j in range(k, 0, -1):
        s = split[pos][j]
        groups.append(list(range(s, pos)))
        pos = s
    groups.reverse()

    # Build mask
    mask = [0] * (n - 1)
    for g in groups[:-1]:
        boundary = g[-1]  # boundary after last chunk in this group
        if boundary < n - 1:
            mask[boundary] = 1

    return _make_plan(model_name, k, mask, groups, chunk_times, has_timing)


def policy_aware_balanced_split(
    chunk_times: List[float],
    target_k: int,
    enabled_boundaries: List[int],
    model_name: str = "",
) -> SplitPlan:
    """
    Compute a balanced K-chunk partition restricted to enabled boundaries.

    Disabled boundaries are forced to 0. If target_k asks for more chunks than
    the policy can materialize, K is clamped to len(enabled_boundaries) + 1.
    For an unrestricted policy this is equivalent to balanced_split().
    """
    n = len(chunk_times)
    if n <= 1:
        return _make_plan(model_name, 1, [], [list(range(n))], chunk_times, any(t > 0.0 for t in chunk_times))

    enabled = sorted({b for b in enabled_boundaries if isinstance(b, int) and 0 <= b < n - 1})
    if len(enabled) == n - 1:
        return balanced_split(chunk_times, target_k, model_name=model_name)

    has_timing = any(t > 0.0 for t in chunk_times)
    max_k = len(enabled) + 1
    k = max(1, min(target_k, max_k))

    if k == 1:
        return _make_plan(model_name, k, [0] * (n - 1), [list(range(n))], chunk_times, has_timing)

    cut_positions = [b + 1 for b in enabled]
    positions = [0] + cut_positions + [n]
    end_idx = len(positions) - 1

    if k == max_k:
        cuts = cut_positions
        return _plan_from_cuts(chunk_times, cuts, model_name, k, has_timing)

    if not has_timing:
        return _policy_equal_size_plan(chunk_times, k, enabled, model_name)

    ps = _prefix_sums(chunk_times)

    def range_sum(lo_pos_idx: int, hi_pos_idx: int) -> float:
        return ps[positions[hi_pos_idx]] - ps[positions[lo_pos_idx]]

    INF = float("inf")
    # dp[a][j] = best max-group sum to reach positions[a] using j groups.
    dp = [[INF] * (k + 1) for _ in positions]
    prev = [[-1] * (k + 1) for _ in positions]
    dp[0][0] = 0.0

    for j in range(1, k + 1):
        for a in range(1, end_idx + 1):
            best = INF
            best_b = -1
            for b in range(0, a):
                if dp[b][j - 1] == INF:
                    continue
                val = max(dp[b][j - 1], range_sum(b, a))
                if val < best:
                    best = val
                    best_b = b
            dp[a][j] = best
            prev[a][j] = best_b

    if dp[end_idx][k] == INF:
        return _policy_equal_size_plan(chunk_times, k, enabled, model_name)

    cuts: List[int] = []
    a = end_idx
    for j in range(k, 0, -1):
        b = prev[a][j]
        if b <= 0:
            a = b
            continue
        cuts.append(positions[b])
        a = b
    cuts.reverse()

    return _plan_from_cuts(chunk_times, cuts, model_name, k, has_timing)


def _make_plan(
    model_name: str,
    k: int,
    mask: List[int],
    groups: List[List[int]],
    chunk_times: List[float],
    has_timing: bool,
    fallback: bool = False,
) -> SplitPlan:
    group_times = [sum(chunk_times[i] for i in grp) for grp in groups]
    total = sum(group_times)
    max_t = max(group_times) if group_times else 0.0
    avg = total / k if k > 0 else 0.0
    imbalance = (max_t / avg) if avg > 0 else 1.0
    return SplitPlan(
        model_name=model_name,
        target_k=k,
        actual_k=len(groups),
        mask=mask,
        groups=groups,
        group_times_ms=group_times,
        max_group_ms=max_t,
        total_ms=total,
        imbalance_ratio=imbalance,
        objective_score=max_t,
        has_timing=has_timing,
        fallback_equal=fallback,
    )


def _equal_size_plan(
    chunk_times: List[float],
    k: int,
    model_name: str,
) -> SplitPlan:
    """Fallback: divide N chunks into K approximately equal-size groups."""
    n = len(chunk_times)
    base_size = n // k
    remainder = n % k
    groups: List[List[int]] = []
    start = 0
    for i in range(k):
        size = base_size + (1 if i < remainder else 0)
        groups.append(list(range(start, start + size)))
        start += size

    mask = [0] * (n - 1)
    for g in groups[:-1]:
        mask[g[-1]] = 1

    return _make_plan(model_name, k, mask, groups, chunk_times, has_timing=False, fallback=True)


def _plan_from_cuts(
    chunk_times: List[float],
    cuts: List[int],
    model_name: str,
    k: int,
    has_timing: bool,
    fallback: bool = False,
) -> SplitPlan:
    n = len(chunk_times)
    groups: List[List[int]] = []
    start = 0
    for cut in cuts + [n]:
        groups.append(list(range(start, cut)))
        start = cut

    mask = [0] * (n - 1)
    for cut in cuts:
        boundary = cut - 1
        if 0 <= boundary < n - 1:
            mask[boundary] = 1

    return _make_plan(model_name, k, mask, groups, chunk_times, has_timing, fallback=fallback)


def _policy_equal_size_plan(
    chunk_times: List[float],
    k: int,
    enabled_boundaries: List[int],
    model_name: str,
) -> SplitPlan:
    """Fallback for no-timing data: choose approximately even enabled cuts."""
    max_k = len(enabled_boundaries) + 1
    k = max(1, min(k, max_k))
    if k == 1:
        return _make_plan(
            model_name, k, [0] * (len(chunk_times) - 1), [list(range(len(chunk_times)))],
            chunk_times, has_timing=False, fallback=True
        )

    cut_count = k - 1
    if cut_count == len(enabled_boundaries):
        cuts = [b + 1 for b in enabled_boundaries]
    else:
        n = len(chunk_times)
        chosen: List[int] = []
        for i in range(1, k):
            ideal_boundary = round((i * n / k) - 1)
            candidates = [b for b in enabled_boundaries if b not in chosen]
            chosen.append(min(candidates, key=lambda b: abs(b - ideal_boundary)))
        cuts = sorted(b + 1 for b in chosen)

    return _plan_from_cuts(
        chunk_times, cuts, model_name, k, has_timing=False, fallback=True
    )
