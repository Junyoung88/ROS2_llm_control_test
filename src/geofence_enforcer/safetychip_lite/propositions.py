"""
SafetyChip-lite: Propositions Module

Shared forbidden zone evaluation for fair comparison with Geofence.
This module is intentionally similar to SELP-lite's propositions.py
to ensure both baselines use identical zone definitions.
"""

# Import from SELP-lite for consistency
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selp_lite.propositions import (
    PropositionEvaluator,
    PropositionState,
    Zone,
    create_factory_propositions,
    create_simple_grid_propositions,
    grid_to_continuous,
    continuous_to_grid,
    create_grid_zone,
)

__all__ = [
    "PropositionEvaluator",
    "PropositionState",
    "Zone",
    "create_factory_propositions",
    "create_simple_grid_propositions",
    "grid_to_continuous",
    "continuous_to_grid",
    "create_grid_zone",
]
