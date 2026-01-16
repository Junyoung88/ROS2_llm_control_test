# SELP-lite: LTL-based Constrained Decoding for Safe Robot Navigation

SELP-lite is a simplified implementation of SELP (Safe Execution of LLM-Generated Plans) for comparison with geometric Geofence-based safety mechanisms.

## Overview

SELP-lite implements two key components:

1. **Equivalence Voting**: Converts natural language constraints to LTL formulas via candidate generation and majority voting
2. **Constrained Decoding**: Generates safe action sequences by masking actions that would violate LTL constraints

### Key Difference from Geofence

| Aspect | SELP-lite | Geofence |
|--------|-----------|----------|
| Safety Mechanism | LTL automaton + action masking | Geometric projection/blocking |
| Constraint Input | Natural language -> LTL | Polygon vertices + margin |
| Runtime Check | Automaton progress | Point-in-polygon + distance |
| Intervention | Mask unsafe actions | Project/block goals |

## Directory Structure

```
selp_lite/
├── propositions.py         # Shared evaluation (forbidden zones, margins)
├── env_grid.py             # Grid world environment
├── ltl_automaton.py        # LTL -> Automaton conversion
├── nl2ltl.py               # NL -> LTL with equivalence voting
├── planner_constrained.py  # Constrained decoding planner
├── run_experiments.py      # Batch experiment runner
├── configs/
│   └── factory_experiment.yaml
└── README.md
```

## Installation

```bash
# Required dependencies
pip install numpy shapely pyyaml

# Optional: spot library for full LTL support
# conda install -c conda-forge spot
```

## Quick Start

### 1. Run a single experiment

```bash
cd src/geofence_enforcer
python -m selp_lite.run_experiments --scenario factory --seeds 10 --verbose
```

### 2. Run with custom constraint

```bash
python -m selp_lite.run_experiments \
    --constraint "Never enter the welding_cell" \
    --seeds 50 \
    --output results_welding.csv
```

### 3. Run with configuration file

```bash
python -m selp_lite.run_experiments \
    --config selp_lite/configs/factory_experiment.yaml \
    --seeds 0-199 \
    --output results.csv
```

## Output

### CSV Metrics

Each experiment produces a CSV with the following columns:

| Column | Description |
|--------|-------------|
| seed | Random seed |
| method | "selp_lite", "geofence", or "no_guard" |
| success | Goal reached AND no violations |
| safe | No forbidden zone violations |
| violations | Number of safety violations |
| steps | Total planning steps |
| masked_actions | Number of masked (blocked) actions |
| min_distance_to_forbidden | Minimum distance to forbidden zone |

### Voting Log

The NL->LTL conversion process is logged with:
1. K generated LTL candidates
2. Equivalence grouping results
3. Voting outcome and selected formula

Example:
```
Input: Always avoid the storage_racks area
Generating 5 LTL candidates...
Candidates generated: 5
  [0] G(!in_storage_racks)
  [1] !F(in_storage_racks)
  [2] (G(!in_storage_racks))
  ...

Equivalence groups: 2
  Group 0: [0, 2, 3] -> G(!in_storage_racks)
  Group 1: [1, 4] -> !F(in_storage_racks)

Voting results:
  Group 0: 3 votes <- WINNER
  Group 1: 2 votes

Selected LTL: G(!in_storage_racks)
Confidence: 0.60
```

### Step-by-step Masking Log

Each planning step logs:
- Current position and propositions
- Action evaluations (safe/masked)
- Masking reasons (which proposition changes violate LTL)
- Selected action and heuristic reason

## LTL Patterns Supported

### Safety Patterns
```
G(!p)           # Always avoid p
G(p)            # Always maintain p
```

### Liveness Patterns
```
F(p)            # Eventually reach p
```

### Combined Patterns
```
F(p) & G(!q)    # Reach p while avoiding q
(!q) U p        # Avoid q until p
G(p -> F(q))    # If p then eventually q
```

## Example Results

```
================================================================================
EXPERIMENT SUMMARY
================================================================================
Constraint: Always avoid storage_racks
Environment: factory
Start: (1.0, 1.0), Goal: (18.0, 13.0)
--------------------------------------------------------------------------------
Method           Success%    Safety%      Goal%        Steps      Masked    MinDist
--------------------------------------------------------------------------------
selp_lite           92.0%      100.0%      92.0%     45.2+/-12.3      18.4       0.42
geofence            94.0%      100.0%      94.0%     44.8+/-11.9      16.2       0.38
no_guard            78.0%       80.0%      98.0%     42.1+/-10.5       0.0       0.15
================================================================================
```

## For Paper Comparison

The key claims that can be made:

1. **SELP-lite implements "SELP-style" constrained decoding**: The voting log shows K candidates and equivalence grouping; the step log shows per-action masking with LTL violation explanations.

2. **Fair comparison with Geofence**: Both methods use the same proposition evaluation (`propositions.py`) and action selection heuristic.

3. **Complementary approaches**:
   - SELP-lite: Better at temporal constraints (ordering, response)
   - Geofence: Better at geometric safety with uncertainty handling

## Extending

### Add new constraint patterns

Edit `nl2ltl.py` and add patterns to `NL_PATTERNS`:

```python
(r"pattern_regex", lambda m: f"LTL_formula")
```

### Add new environment

Create new proposition configuration in `propositions.py`:

```python
def create_custom_propositions() -> PropositionEvaluator:
    evaluator = PropositionEvaluator()
    evaluator.add_zone("zone_name", [(x1,y1), (x2,y2), ...])
    return evaluator
```

## Citation

If using for comparison, cite both:
- SELP (original): [SELP paper reference]
- Geofence (this work): [Your paper reference]
