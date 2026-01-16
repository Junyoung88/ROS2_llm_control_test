"""
Unified Experiments Module

Provides unified S1-S6 comparison experiments for:
- no_guard: No safety mechanism (baseline)
- safetychip: SafetyChip-lite (LTL monitor + action pruning + reprompting)
- selp: SELP-lite (LTL constrained decoding)
- geofence: Geofence (geometric safety zones)
"""

from .unified_s1_s6_comparison import (
    UnifiedExperimentRunner,
    Method,
    ScenarioConfig,
    EpisodeResult,
    ScenarioSummary,
    create_scenarios,
)

__all__ = [
    "UnifiedExperimentRunner",
    "Method",
    "ScenarioConfig",
    "EpisodeResult",
    "ScenarioSummary",
    "create_scenarios",
]
