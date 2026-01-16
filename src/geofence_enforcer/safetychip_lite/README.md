# SafetyChip-lite: LTL-based Safety Enforcement for Robot Navigation

SafetyChip-lite is a simplified implementation of SafetyChip for comparison with geometric Geofence-based safety mechanisms.

## Overview

SafetyChip-lite implements the core SafetyChip architecture:

1. **NL→LTL Translation**: Converts natural language constraints to LTL formulas
2. **Constraint Monitoring**: Runtime LTL monitoring using progression semantics
3. **Action Pruning**: Blocks unsafe actions BEFORE execution
4. **Reprompting**: Explains violations and requests alternative actions

### Key Guarantee

**100% Safety**: Unsafe actions are NEVER executed. All constraint violations are detected and blocked before the action takes effect.

### Key Difference from Geofence

| Aspect | SafetyChip-lite | Geofence |
|--------|-----------------|----------|
| Safety Mechanism | LTL monitor + action pruning | Geometric projection/blocking |
| Constraint Input | Natural language → LTL | Polygon vertices + margin |
| Runtime Check | LTL progression | Point-in-polygon + distance |
| Intervention | Prune + explain + reprompt | Project/block goals |
| Temporal Constraints | Yes (precedence, trigger-response) | No |

## Directory Structure

```
safetychip_lite/
├── nl2ltl.py              # NL → LTL rule-based translator
├── ltl_monitor.py         # Progression-based LTL monitor
├── planner.py             # Heuristic/LLM action planner
├── agent_loop.py          # Main execution loop (prune + reprompt)
├── propositions.py        # Shared zone evaluation (same as Geofence)
├── env_grid.py            # Grid world environment
├── run_experiments.py     # Batch experiment runner
├── configs/
│   └── factory_experiment.yaml
└── README.md
```

## Installation

```bash
# Required dependencies
pip install numpy shapely pyyaml

# No additional dependencies (spot not required)
# Uses progression-based monitoring for supported patterns
```

## Quick Start

### 1. Run a single experiment

```bash
cd src/geofence_enforcer
python -m safetychip_lite.run_experiments --scenario factory --seeds 10 --verbose
```

### 2. Run with custom constraint

```bash
python -m safetychip_lite.run_experiments \
    --constraint "Never enter the welding_cell" \
    --seeds 50 \
    --output results_welding.csv
```

### 3. Run with configuration file

```bash
python -m safetychip_lite.run_experiments \
    --config safetychip_lite/configs/factory_experiment.yaml \
    --seeds 0-199 \
    --output results.csv
```

## Supported Constraint Patterns

### 1. Always Avoid: `G(!p)`

```
"Never enter the storage_racks"     → G(!in_storage_racks)
"Avoid the welding_cell"            → G(!in_welding_cell)
"Storage is forbidden"              → G(!in_storage)
```

### 2. Precedence: `(!q) U p`

```
"Visit checkpoint before goal"      → (!in_goal) U in_checkpoint
"Go to A first then B"              → (!in_b) U in_a
```

### 3. Trigger-Response: `G(p → F(q))`

```
"If enter warning_zone then must reach exit"
    → G(in_warning_zone → F(in_exit))
```

## Output

### CSV Metrics

Each experiment produces a CSV with the following columns:

| Column | Description |
|--------|-------------|
| seed | Random seed |
| method | "safetychip", "geofence", or "no_guard" |
| success | Goal reached AND no violations |
| safety_violation | Constraint zone entered (should be False for SafetyChip) |
| pruned_count | Number of actions blocked before execution |
| reprompt_count | Number of times planner was asked for alternatives |
| steps | Total planning steps |
| min_distance_to_constraint_zone | Closest approach to constraint zone |

### Step-by-Step Log

Each planning step logs:
1. Current position and propositions
2. Proposed action from planner
3. Monitor verdict (safe/unsafe)
4. If unsafe:
   - Which constraint was violated
   - Which proposition changes caused it
   - Explanation text
   - Reprompt text sent to planner
5. Final executed action

Example log:
```
============================================================
SafetyChip-lite Agent Starting
============================================================
Constraints: ['Never enter the storage_racks']
LTL: G(!in_storage_racks)

Step 15: Planner proposes MOVE_EAST
  Reasoning: Moves toward goal (distance: 4.24)
  -> UNSAFE: Action PRUNED
  Reason: Entered forbidden zone: storage_racks
  Reprompting (1/5)...

Step 15: Planner proposes MOVE_NORTH
  Reasoning: Moves toward goal (distance: 4.47)
  -> SAFE: Action approved
--------------------------------------------------
```

## Example Results

```
=======================================================================================
SAFETYCHIP-LITE vs GEOFENCE EXPERIMENT SUMMARY
=======================================================================================
Constraint: Never enter the storage_racks
LTL: G(!in_storage_racks)
Environment: factory
Start: (1.0, 1.0), Goal: (18.0, 13.0)
---------------------------------------------------------------------------------------
Method            Success%    Safety%      Goal%        Steps     Pruned  Reprompt  ConstDist
---------------------------------------------------------------------------------------
safetychip          100.0%     100.0%     100.0%    29.0+/-0.0      5.2       5.2       0.00
geofence            100.0%     100.0%     100.0%    29.0+/-0.0      8.9       0.0       1.00
no_guard              0.0%       0.0%     100.0%    29.0+/-0.0      0.0       0.0      -1.00
=======================================================================================
Key SafetyChip metrics:
  - Safety%: Percentage of episodes with NO constraint zone violations
  - Pruned: Actions blocked BEFORE execution (SafetyChip guarantee)
  - Reprompt: Times planner was asked to propose alternative action
=======================================================================================
```

## For Paper Comparison

The key claims that can be made:

1. **SafetyChip-lite implements the SafetyChip architecture**:
   - NL→LTL translation with rule-based patterns
   - LTL monitoring via progression semantics
   - Action pruning before execution
   - Violation explanation and reprompting

2. **100% Safety Guarantee**: The monitor checks actions BEFORE execution, ensuring no unsafe actions are ever taken.

3. **Fair comparison with Geofence**: Both methods use identical proposition evaluation (`propositions.py`) and the same underlying environment.

4. **Complementary approaches**:
   - SafetyChip-lite: Better for temporal constraints (ordering, trigger-response)
   - Geofence: Better for geometric safety with uncertainty handling

## Extending

### Add new constraint patterns

Edit `nl2ltl.py` and add patterns to the appropriate category:

```python
AVOID_PATTERNS = [
    (r"your_regex_pattern", ConstraintType.ALWAYS_AVOID),
]
```

### Add new monitoring logic

Edit `ltl_monitor.py` and implement the pattern in `PatternMonitor`:

```python
def _step_your_pattern(self, props):
    # Your progression logic
    pass

def _check_your_pattern(self, props):
    # Your violation checking logic
    pass
```

## Limitations

This is a simplified implementation:

1. **Limited LTL Patterns**: Only supports the three main patterns (avoid, precedence, trigger-response) without full LTL semantics.

2. **Progression-based**: Uses progression semantics instead of full automaton construction (no spot dependency).

3. **No Temporal Deadlines**: Trigger-response obligations are tracked but not enforced with deadlines.

## Citation

If using for comparison, cite both:
- SafetyChip (original): [SafetyChip paper reference]
- Geofence (this work): [Your paper reference]
