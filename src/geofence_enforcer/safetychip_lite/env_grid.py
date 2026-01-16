"""
SafetyChip-lite: Grid World Environment

Shared environment for fair comparison with Geofence and SELP-lite.
Reuses SELP-lite's environment implementation.
"""

# Import from SELP-lite for consistency
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selp_lite.env_grid import (
    GridWorld,
    GridState,
    Action,
    StepResult,
    create_factory_environment,
    create_simple_environment,
)

__all__ = [
    "GridWorld",
    "GridState",
    "Action",
    "StepResult",
    "create_factory_environment",
    "create_simple_environment",
]
