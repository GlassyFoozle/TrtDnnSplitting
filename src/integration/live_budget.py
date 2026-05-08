"""
live_budget.py — Global live-profiling safety budget for one script run.

Shared across all tasksets × algorithms × tasks. Cache hits never count
against the budget.

Three independent safety controls can be combined:
  cache_only             — skip any mask that is not already cached
  global_max_real_profiles N — hard cap on total new TRT profiles
  stop_on_first_build    — stop the entire run at the first cache miss
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LiveProfileBudget:
    cache_only: bool = False
    global_max_real_profiles: Optional[int] = None   # None = unlimited
    stop_on_first_build: bool = False

    # Runtime counters — mutated in place, shared across all callers
    used_real_profiles: int = field(default=0, compare=False)
    skipped_cache_misses: int = field(default=0, compare=False)
    stopped: bool = field(default=False, compare=False)
    stop_model: str = field(default="", compare=False)
    stop_variant: str = field(default="", compare=False)

    def budget_exhausted(self) -> bool:
        if self.global_max_real_profiles is None:
            return False
        return self.used_real_profiles >= self.global_max_real_profiles

    def check_before_real_eval(self, model_name: str, variant_name: str) -> Optional[str]:
        """
        Call this when a cache miss is about to trigger a real TRT pipeline run.

        Returns a skip-reason string if the eval should be suppressed, or None
        if the eval may proceed.

        Side-effect: sets self.stopped = True on stop_on_first_build.
        """
        if self.stopped:
            return "stopped_on_first_build"
        if self.cache_only:
            return "cache_miss_live_disabled"
        if self.budget_exhausted():
            return "global_budget_exhausted"
        if self.stop_on_first_build:
            self.stopped = True
            self.stop_model = model_name
            self.stop_variant = variant_name
            return "stop_on_first_build"
        return None

    def record_real_profile(self) -> None:
        self.used_real_profiles += 1

    def record_skip(self) -> None:
        self.skipped_cache_misses += 1

    def remaining(self) -> Optional[int]:
        if self.global_max_real_profiles is None:
            return None
        return max(0, self.global_max_real_profiles - self.used_real_profiles)

    def to_dict(self) -> dict:
        return {
            "live_cache_only": self.cache_only,
            "global_max_real_profiles": self.global_max_real_profiles,
            "stop_on_first_build": self.stop_on_first_build,
            "global_profile_budget_used": self.used_real_profiles,
            "global_profile_budget_remaining": self.remaining(),
            "skipped_cache_misses": self.skipped_cache_misses,
            "live_build_attempted": self.stopped,
            "stop_model": self.stop_model,
            "stop_variant": self.stop_variant,
        }
