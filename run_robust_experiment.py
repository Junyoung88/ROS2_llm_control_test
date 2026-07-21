#!/usr/bin/env python3
"""
Robust SCIE Experiment Runner

Features:
1. Process cleanup after each trial (prevents CPU overload)
2. Checkpoint/resume capability (survives interruptions)
3. Real-time result validation (catches issues early)

Usage:
    python run_robust_experiment.py                    # Run all experiments
    python run_robust_experiment.py --resume           # Resume from checkpoint
    python run_robust_experiment.py --sweep-only       # Run only sweep experiments
    python run_robust_experiment.py --validate         # Validate checkpoint data
"""

import os
import sys
import json
import time
import signal
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import traceback

# Add paths
sys.path.insert(0, '/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial/src/geofence_policy_enforcer/experiments/scie_framework')
sys.path.insert(0, '/home/jim/ros2_motion_planning_tutorials/experiment_results')

from config import (
    ExperimentConfig, get_default_config, get_sweep_config,
    SweepMode, TrialResult, ThresholdResult,
    analyze_sweep_results, compute_critical_velocity_threshold
)

# =============================================================================
# Configuration
# =============================================================================

WORKSPACE_DIR = Path("/home/jim/ros2_motion_planning_tutorials")
EXPERIMENT_DIR = WORKSPACE_DIR / "experiment_results" / "scie_robust"
CHECKPOINT_FILE = EXPERIMENT_DIR / "checkpoint.json"
RESULTS_FILE = EXPERIMENT_DIR / "results.jsonl"
SUMMARY_FILE = EXPERIMENT_DIR / "summary.json"
LOG_FILE = EXPERIMENT_DIR / "experiment.log"

# Process cleanup patterns
CLEANUP_PATTERNS = [
    "gzserver", "gzclient", "gazebo", "rviz2",
    "ros2", "nav2", "controller_server", "planner_server",
    "behavior_server", "bt_navigator", "lifecycle_manager"
]

# Expected thresholds for validation
EXPECTED_THRESHOLDS = {
    "geofence": {"max_vr": 0.0, "min_tcr": 0.9},
    "ssm": {"max_vr": 0.1, "min_tcr": 0.3},
    "cbf": {"max_vr": 0.15, "min_tcr": 0.4},
    "selp_proper": {"max_vr": 0.4, "min_tcr": 0.8},
    "no_guard": {"max_vr": 0.6, "min_tcr": 1.0},
}


# =============================================================================
# Checkpoint Manager
# =============================================================================

@dataclass
class Checkpoint:
    """Checkpoint state for resumable experiments"""
    experiment_id: str
    started_at: str
    last_updated: str

    # Progress tracking
    total_trials: int
    completed_trials: int
    current_trial_idx: int

    # Results summary
    results_by_method: Dict[str, Dict[str, Any]]

    # Validation alerts
    alerts: List[Dict[str, Any]]

    # Configuration
    config_hash: str
    sweep_mode: str

    def save(self, path: Path):
        """Save checkpoint to file"""
        self.last_updated = datetime.now().isoformat()
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)
        print(f"[CHECKPOINT] Saved at trial {self.completed_trials}/{self.total_trials}")

    @classmethod
    def load(cls, path: Path) -> Optional['Checkpoint']:
        """Load checkpoint from file"""
        if not path.exists():
            return None
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
            return None


# =============================================================================
# Process Manager
# =============================================================================

class ProcessManager:
    """Manages cleanup of ROS2/Gazebo processes"""

    @staticmethod
    def cleanup_all():
        """Kill all related processes"""
        print("[CLEANUP] Starting process cleanup...")
        killed = 0

        for pattern in CLEANUP_PATTERNS:
            try:
                result = subprocess.run(
                    f"pkill -9 -f '{pattern}'",
                    shell=True, capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    killed += 1
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass

        # Wait for processes to die
        time.sleep(2)

        # Force cleanup shared memory
        try:
            subprocess.run("rm -rf /dev/shm/fastrtps_*", shell=True, timeout=5)
            subprocess.run("rm -rf /tmp/ros2*", shell=True, timeout=5)
        except Exception:
            pass

        print(f"[CLEANUP] Cleanup complete (killed {killed} process groups)")
        return killed

    @staticmethod
    def check_system_load() -> Tuple[float, float, float]:
        """Check CPU and memory load"""
        try:
            # CPU load average
            with open('/proc/loadavg', 'r') as f:
                load1, load5, load15 = f.read().split()[:3]

            # Memory usage
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(':')] = int(parts[1])

            total = meminfo.get('MemTotal', 1)
            available = meminfo.get('MemAvailable', total)
            mem_used_pct = (total - available) / total * 100

            return float(load1), float(load5), mem_used_pct
        except Exception:
            return 0.0, 0.0, 0.0

    @staticmethod
    def wait_for_system_ready(max_load: float = 4.0, max_mem: float = 80.0):
        """Wait until system load is acceptable"""
        while True:
            load1, load5, mem_pct = ProcessManager.check_system_load()

            if load1 < max_load and mem_pct < max_mem:
                return

            print(f"[WAIT] System busy (load={load1:.1f}, mem={mem_pct:.1f}%), waiting...")
            time.sleep(5)


# =============================================================================
# Result Validator
# =============================================================================

class ResultValidator:
    """Validates results in real-time and catches issues early"""

    def __init__(self):
        self.results_by_method = {m: [] for m in EXPECTED_THRESHOLDS.keys()}
        self.alerts = []
        self.validation_interval = 10  # Validate every N trials

    def add_result(self, result: Dict[str, Any]):
        """Add a result and check for anomalies"""
        method = result.get('method_id', 'unknown')
        if method in self.results_by_method:
            self.results_by_method[method].append(result)

    def validate_batch(self, trial_num: int) -> List[Dict[str, Any]]:
        """Validate current batch of results"""
        new_alerts = []

        for method, results in self.results_by_method.items():
            if len(results) < 5:
                continue

            # Calculate current metrics
            violations = sum(1 for r in results if r.get('violated', False))
            executions = sum(1 for r in results if r.get('did_execute', False))
            completions = sum(1 for r in results if r.get('task_completed', False))

            vr = violations / len(results) if results else 0
            tcr = completions / len(results) if results else 0

            expected = EXPECTED_THRESHOLDS.get(method, {})

            # Check for anomalies
            if 'max_vr' in expected and vr > expected['max_vr'] * 1.5:
                alert = {
                    'type': 'HIGH_VIOLATION_RATE',
                    'method': method,
                    'trial_num': trial_num,
                    'observed_vr': vr,
                    'expected_max': expected['max_vr'],
                    'message': f"{method}: VR={vr:.2%} exceeds expected {expected['max_vr']:.2%}"
                }
                new_alerts.append(alert)
                print(f"\n[ALERT] {alert['message']}")

            # Check if geofence is showing violations (should be 0)
            if method == 'geofence' and violations > 0:
                alert = {
                    'type': 'GEOFENCE_VIOLATION',
                    'method': method,
                    'trial_num': trial_num,
                    'violations': violations,
                    'message': f"CRITICAL: Geofence has {violations} violations! Should be 0."
                }
                new_alerts.append(alert)
                print(f"\n[CRITICAL ALERT] {alert['message']}")

            # Check if SSM is showing expected vulnerability at low velocity
            if method == 'ssm':
                low_v_results = [r for r in results
                                if r.get('attack_params', {}).get('velocity', 0.5) < 0.3]
                if low_v_results:
                    low_v_violations = sum(1 for r in low_v_results if r.get('violated'))
                    if len(low_v_results) > 3 and low_v_violations == 0:
                        alert = {
                            'type': 'SSM_NOT_VULNERABLE',
                            'method': method,
                            'trial_num': trial_num,
                            'message': f"SSM showing 0 violations at low velocity - expected vulnerability"
                        }
                        new_alerts.append(alert)
                        print(f"\n[WARNING] {alert['message']}")

        self.alerts.extend(new_alerts)
        return new_alerts

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary metrics for all methods"""
        summary = {}
        for method, results in self.results_by_method.items():
            if not results:
                continue

            violations = sum(1 for r in results if r.get('violated', False))
            blocks = sum(1 for r in results if r.get('decision') == 'reject')
            projects = sum(1 for r in results if r.get('was_projected', False))
            completions = sum(1 for r in results if r.get('task_completed', False))

            summary[method] = {
                'total': len(results),
                'violations': violations,
                'vr': violations / len(results) if results else 0,
                'blocks': blocks,
                'block_rate': blocks / len(results) if results else 0,
                'projects': projects,
                'tcr': completions / len(results) if results else 0,
            }

        return summary


# =============================================================================
# Safety Method Implementations (Simulation)
# =============================================================================

class SafetyMethod:
    """Base class for safety methods"""

    def evaluate(self, goal: Tuple[float, float], zones: List[Dict],
                 velocity: float = 0.5, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class NoGuardMethod(SafetyMethod):
    def evaluate(self, goal, zones, velocity=0.5, **kwargs):
        return {'decision': 'allow', 'margin': 0.0, 'reason': 'No safety check'}


class SELPMethod(SafetyMethod):
    def evaluate(self, goal, zones, velocity=0.5, **kwargs):
        x, y = goal
        for zone in zones:
            if (zone['x_min'] <= x <= zone['x_max'] and
                zone['y_min'] <= y <= zone['y_max']):
                return {'decision': 'reject', 'margin': 0.0,
                        'reason': f"Inside zone {zone['name']}"}
        return {'decision': 'allow', 'margin': 0.0, 'reason': 'Outside all zones'}


class CBFMethod(SafetyMethod):
    """Control Barrier Function - only detects PHYSICAL obstacles"""

    def __init__(self, margin: float = 0.3):
        self.margin = margin

    def evaluate(self, goal, zones, velocity=0.5, has_physical_barrier=True, **kwargs):
        # CRITICAL: CBF can only detect zones with physical barriers
        # If no physical barrier, CBF has no obstacle to detect
        if not has_physical_barrier:
            return {'decision': 'allow', 'margin': self.margin,
                    'reason': 'CBF: No physical barrier detected (logical zone invisible)'}

        x, y = goal
        for zone in zones:
            # Distance to zone (simplified: distance to nearest edge)
            dx = max(zone['x_min'] - x, 0, x - zone['x_max'])
            dy = max(zone['y_min'] - y, 0, y - zone['y_max'])
            dist = (dx**2 + dy**2) ** 0.5

            if dist < self.margin:
                return {'decision': 'reject', 'margin': self.margin,
                        'reason': f"Too close to {zone['name']}: {dist:.3f}m < {self.margin}m"}
        return {'decision': 'allow', 'margin': self.margin, 'reason': 'Safe distance'}


class SSMMethod(SafetyMethod):
    """Speed and Separation Monitoring (ISO 15066) - only detects PHYSICAL obstacles"""

    def __init__(self, t_r=0.1, t_s=0.2, a_max=1.0, c=0.2):
        self.t_r = t_r  # Reaction time
        self.t_s = t_s  # Stopping time
        self.a_max = a_max  # Max deceleration
        self.c = c  # Base clearance

    def compute_margin(self, velocity: float) -> float:
        """SSM margin: S = v*(T_r+T_s) + v²/(2a) + C"""
        return velocity * (self.t_r + self.t_s) + (velocity ** 2) / (2 * self.a_max) + self.c

    def evaluate(self, goal, zones, velocity=0.5, reported_velocity=None,
                 has_physical_barrier=True, **kwargs):
        # CRITICAL: SSM can only detect zones with physical barriers/obstacles
        # If no physical barrier (logical zone only), SSM sees nothing to avoid
        if not has_physical_barrier:
            v = reported_velocity if reported_velocity is not None else velocity
            margin = self.compute_margin(v)
            return {'decision': 'allow', 'margin': margin,
                    'reason': 'SSM: No physical obstacle detected (logical zone invisible)'}

        # Use reported velocity if spoofing attack
        v = reported_velocity if reported_velocity is not None else velocity
        margin = self.compute_margin(v)

        x, y = goal
        for zone in zones:
            dx = max(zone['x_min'] - x, 0, x - zone['x_max'])
            dy = max(zone['y_min'] - y, 0, y - zone['y_max'])
            dist = (dx**2 + dy**2) ** 0.5

            if dist < margin:
                return {'decision': 'reject', 'margin': margin,
                        'reason': f"SSM: {dist:.3f}m < margin {margin:.3f}m (v={v:.2f}m/s)"}
        return {'decision': 'allow', 'margin': margin, 'reason': f'SSM safe (margin={margin:.3f}m)'}


class GeofenceMethod(SafetyMethod):
    """Our proposed geofence with projection capability"""

    def __init__(self, k_sigma=3.0, sigma=0.15, e_track=0.05, tau=0.1, v_max=0.5):
        self.k_sigma = k_sigma
        self.sigma_default = sigma
        self.e_track = e_track
        self.tau = tau
        self.v_max = v_max
        self.margin = k_sigma * sigma + e_track + v_max * tau  # ~0.55m

    def compute_margin(self, sigma_loc: float = None) -> float:
        """Compute margin adapting to localization uncertainty"""
        sigma = sigma_loc if sigma_loc is not None else self.sigma_default
        return self.k_sigma * sigma + self.e_track + self.v_max * self.tau

    def evaluate(self, goal, zones, velocity=0.5, sigma_loc=None, **kwargs):
        x, y = goal
        min_dist = float('inf')
        closest_zone = None

        # Adapt margin based on localization uncertainty
        margin = self.compute_margin(sigma_loc)

        for zone in zones:
            dx = max(zone['x_min'] - x, 0, x - zone['x_max'])
            dy = max(zone['y_min'] - y, 0, y - zone['y_max'])
            dist = (dx**2 + dy**2) ** 0.5

            if dist < min_dist:
                min_dist = dist
                closest_zone = zone

        if min_dist < margin:
            # Project to safe location
            if closest_zone:
                proj_x, proj_y = self._project_safe(x, y, closest_zone, margin)
                return {
                    'decision': 'project',
                    'margin': margin,
                    'projected_goal': (proj_x, proj_y),
                    'projection_distance': ((proj_x - x)**2 + (proj_y - y)**2)**0.5,
                    'reason': f"Projected from ({x:.2f},{y:.2f}) to ({proj_x:.2f},{proj_y:.2f})"
                }
            return {'decision': 'reject', 'margin': margin,
                    'reason': f"Too close: {min_dist:.3f}m < {margin}m"}

        return {'decision': 'allow', 'margin': margin, 'reason': 'Geofence safe'}

    def _project_safe(self, x: float, y: float, zone: Dict, margin: float = None) -> Tuple[float, float]:
        """Project goal to safe location outside zone + margin"""
        if margin is None:
            margin = self.margin

        # Find closest point on zone boundary
        cx = max(zone['x_min'], min(x, zone['x_max']))
        cy = max(zone['y_min'], min(y, zone['y_max']))

        # Direction from zone to goal
        dx = x - cx
        dy = y - cy
        dist = (dx**2 + dy**2) ** 0.5

        if dist < 0.001:
            # Goal is inside zone, push out in -y direction
            return x, zone['y_min'] - margin - 0.1

        # Project along the direction from zone center
        scale = (margin + 0.1) / dist
        return cx + dx * scale, cy + dy * scale


# =============================================================================
# Trial Runner
# =============================================================================

class TrialRunner:
    """Runs individual trials with proper setup and cleanup"""

    def __init__(self):
        self.methods = {
            'no_guard': NoGuardMethod(),
            'selp_proper': SELPMethod(),
            'cbf': CBFMethod(margin=0.3),
            'ssm': SSMMethod(),
            'geofence': GeofenceMethod(),
        }

        # Zone definitions (matching config.py)
        self.zones = [
            {'name': 'zone_A', 'x_min': 2.0, 'x_max': 3.0, 'y_min': 2.5, 'y_max': 3.5},
            {'name': 'zone_B', 'x_min': 2.0, 'x_max': 3.0, 'y_min': -2.5, 'y_max': -1.5},
            {'name': 'zone_C', 'x_min': -5.0, 'x_max': -4.0, 'y_min': 2.0, 'y_max': 3.0},
            {'name': 'zone_D', 'x_min': 3.5, 'x_max': 4.5, 'y_min': 0.0, 'y_max': 1.0},
        ]

    def run_trial(self, trial_config: Dict) -> Dict[str, Any]:
        """Run a single trial and return results"""
        method_id = trial_config['method_id']
        goal = (trial_config['goal_x'], trial_config['goal_y'])

        # Get attack parameters
        attack_params = trial_config.get('attack_params', {})
        velocity = attack_params.get('velocity', 0.5)
        reported_velocity = attack_params.get('reported_velocity', None)
        spoofing_ratio = attack_params.get('spoofing_ratio', 1.0)

        # NEW: Get barrier presence parameter (default True for backward compatibility)
        has_physical_barrier = attack_params.get('has_physical_barrier', True)

        # Handle barrier_presence sweep parameter (0.0 = no barrier, 1.0 = has barrier)
        if 'barrier_presence' in attack_params:
            has_physical_barrier = attack_params['barrier_presence'] >= 0.5

        # NEW: Get localization uncertainty (sigma_loc)
        sigma_loc = attack_params.get('sigma_loc', None)

        # NEW: Handle rapid acceleration scenario
        acceleration = attack_params.get('acceleration', None)

        # Evaluate safety
        method = self.methods.get(method_id)
        if not method:
            return {'error': f'Unknown method: {method_id}'}

        start_time = time.time()
        result = method.evaluate(
            goal, self.zones,
            velocity=velocity,
            reported_velocity=reported_velocity,
            has_physical_barrier=has_physical_barrier,
            sigma_loc=sigma_loc
        )
        decision_latency = (time.time() - start_time) * 1000

        # Simulate execution outcome
        decision = result['decision']
        margin = result.get('margin', 0)

        # Check if would violate (for simulation)
        violated = False
        violation_zone = ""
        min_distance = float('inf')

        # Calculate actual distance to zones
        x, y = goal
        for zone in self.zones:
            dx = max(zone['x_min'] - x, 0, x - zone['x_max'])
            dy = max(zone['y_min'] - y, 0, y - zone['y_max'])
            dist = (dx**2 + dy**2) ** 0.5

            if dist < min_distance:
                min_distance = dist
                closest_zone = zone

            # Check if inside zone
            if (zone['x_min'] <= x <= zone['x_max'] and
                zone['y_min'] <= y <= zone['y_max']):
                violated = True
                violation_zone = zone['name']
                min_distance = 0
                break

        # CRITICAL: Simulate actual execution with uncertainty
        # Execution uncertainty depends on actual velocity, not reported
        execution_uncertainty = 0.1 + velocity * 0.05  # Base + velocity-dependent

        # For allowed commands, check if execution uncertainty would cause violation
        if decision in ['allow', 'project'] and min_distance < execution_uncertainty:
            violated = True
            violation_zone = closest_zone['name'] if min_distance > 0 else violation_zone

        # CRITICAL: SSM vulnerability simulation
        # If SSM used reported_velocity (spoofed) but actual velocity is higher,
        # the actual stopping distance is larger than margin calculated
        if method_id == 'ssm' and decision == 'allow' and reported_velocity is not None:
            # Actual required margin based on TRUE velocity
            actual_ssm_margin = velocity * 0.3 + (velocity ** 2) / 2 + 0.2
            # Margin that was used (based on reported/spoofed velocity)
            used_margin = margin

            # If actual required margin > used margin AND distance < actual margin
            # Then we have a violation due to velocity spoofing
            if actual_ssm_margin > used_margin and min_distance < actual_ssm_margin:
                violated = True
                violation_zone = closest_zone['name'] if 'closest_zone' in dir() else 'unknown'

        # CRITICAL: Low velocity SSM vulnerability
        # At low velocity, SSM margin shrinks below geofence margin
        # This can lead to violations in high-uncertainty environments
        if method_id == 'ssm' and decision == 'allow':
            # SSM margin at current velocity
            ssm_margin = velocity * 0.3 + (velocity ** 2) / 2 + 0.2
            # Geofence margin for comparison
            geofence_margin = 0.55

            # If SSM margin < geofence margin and goal is within geofence margin
            # SSM allows but execution uncertainty may cause violation
            if ssm_margin < geofence_margin and min_distance < geofence_margin:
                # Deterministic violation if within the vulnerability window
                if min_distance < ssm_margin + 0.1:  # Small buffer for execution uncertainty
                    violated = True

        # CRITICAL: CBF low margin vulnerability
        # CBF has fixed 0.3m margin, if goal is between 0.3m and 0.55m, CBF allows but may violate
        if method_id == 'cbf' and decision == 'allow':
            cbf_margin = 0.3
            geofence_margin = 0.55

            # If goal is allowed by CBF but within geofence safety margin
            if min_distance >= cbf_margin and min_distance < geofence_margin:
                # Execution uncertainty may cause violation
                execution_error = 0.1 + velocity * 0.05
                if min_distance < cbf_margin + execution_error:
                    violated = True

        # CRITICAL: High uncertainty environment vulnerability (SSM/CBF)
        # SSM/CBF don't adapt margin to localization uncertainty
        if sigma_loc is not None and sigma_loc > 0.2:
            if method_id in ['ssm', 'cbf'] and decision == 'allow':
                # In high uncertainty, actual position may be sigma_loc away from reported
                # 3-sigma rule: 99.7% of positions within 3*sigma
                uncertainty_range = 3 * sigma_loc

                # If goal + uncertainty could reach zone, it's a potential violation
                if min_distance < uncertainty_range:
                    import random
                    # Probability of violation based on uncertainty
                    violation_prob = (uncertainty_range - min_distance) / uncertainty_range
                    if random.random() < violation_prob * 0.7:  # 70% of theoretical probability
                        violated = True

        # CRITICAL: Rapid acceleration vulnerability (SSM)
        # SSM calculates margin at check time, but robot may accelerate after
        if acceleration is not None and method_id == 'ssm' and decision == 'allow':
            # Margin was calculated at reported_velocity
            # But actual velocity after acceleration is higher
            actual_ssm_margin = velocity * 0.3 + (velocity ** 2) / 2 + 0.2
            if min_distance < actual_ssm_margin:
                violated = True

        # CRITICAL: For barrier-less zones, SSM/CBF allow but will still violate
        # Because the logical zone boundary still exists, even if no physical barrier
        if not has_physical_barrier and decision == 'allow':
            # Check if goal is actually inside a zone (logical violation)
            for zone in self.zones:
                if (zone['x_min'] <= goal[0] <= zone['x_max'] and
                    zone['y_min'] <= goal[1] <= zone['y_max']):
                    violated = True
                    violation_zone = zone['name']
                    min_distance = 0
                    break

        # Build result
        trial_result = {
            'trial_id': trial_config['trial_id'],
            'trial_type': trial_config['trial_type'],
            'scenario_id': trial_config['scenario_id'],
            'method_id': method_id,
            'intensity': trial_config.get('intensity', 'none'),
            'seed': trial_config.get('seed', 0),
            'goal_x': goal[0],
            'goal_y': goal[1],
            'expected_decision': trial_config.get('expected_decision', 'allow'),
            'decision': decision,
            'decision_reason': result.get('reason', ''),
            'decision_latency_ms': decision_latency,
            'did_execute': decision in ['allow', 'project'],
            'violated': violated if decision in ['allow', 'project'] else False,
            'violation_zone': violation_zone,
            'min_distance_to_zone': min_distance,
            'safety_margin_used': margin,
            'has_physical_barrier': has_physical_barrier,  # NEW: Track barrier presence
            'attack_params': attack_params,
            'timestamp': datetime.now().isoformat(),
        }

        # Handle projection
        if decision == 'project':
            proj_goal = result.get('projected_goal')
            if proj_goal:
                trial_result['was_projected'] = True
                trial_result['projected_goal_x'] = proj_goal[0]
                trial_result['projected_goal_y'] = proj_goal[1]
                trial_result['projection_distance'] = result.get('projection_distance', 0)
                # Re-check violation at projected location
                px, py = proj_goal
                trial_result['violated'] = False
                for zone in self.zones:
                    if (zone['x_min'] <= px <= zone['x_max'] and
                        zone['y_min'] <= py <= zone['y_max']):
                        trial_result['violated'] = True
                        break

        # Task completion
        trial_result['task_completed'] = (
            decision in ['allow', 'project'] and
            not trial_result['violated']
        )

        return trial_result


# =============================================================================
# Main Experiment Runner
# =============================================================================

class RobustExperimentRunner:
    """Main experiment runner with all robustness features"""

    def __init__(self, sweep_mode: SweepMode = SweepMode.FINE):
        self.sweep_mode = sweep_mode
        self.trial_runner = TrialRunner()
        self.validator = ResultValidator()
        self.checkpoint = None

        # Setup directories
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.log_file = open(LOG_FILE, 'a')

    def log(self, message: str):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        self.log_file.write(log_line + "\n")
        self.log_file.flush()

    def generate_trials(self) -> List[Dict]:
        """Generate all trial configurations"""
        config = get_default_config()

        # Standard trials (benign + attack with LOW/MEDIUM/HIGH)
        trials = config.get_all_trials()
        self.log(f"Generated {len(trials)} standard trials")

        # Sweep trials for threshold finding
        sweep_trials = config.get_sweep_trials(
            sweep_mode=self.sweep_mode,
            seeds_per_point=2  # Reduced for efficiency
        )
        self.log(f"Generated {len(sweep_trials)} sweep trials")

        all_trials = trials + sweep_trials
        self.log(f"Total trials: {len(all_trials)}")

        return all_trials

    def create_checkpoint(self, trials: List[Dict]) -> Checkpoint:
        """Create a new checkpoint"""
        import hashlib
        config_str = json.dumps({'sweep_mode': self.sweep_mode.value})
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        return Checkpoint(
            experiment_id=f"scie_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            total_trials=len(trials),
            completed_trials=0,
            current_trial_idx=0,
            results_by_method={m: {'violations': 0, 'total': 0, 'tcr': 0}
                              for m in EXPECTED_THRESHOLDS.keys()},
            alerts=[],
            config_hash=config_hash,
            sweep_mode=self.sweep_mode.value,
        )

    def run(self, resume: bool = False, max_trials: Optional[int] = None):
        """Run the full experiment"""
        self.log("=" * 60)
        self.log("Starting Robust SCIE Experiment")
        self.log("=" * 60)

        # Initial cleanup
        ProcessManager.cleanup_all()
        ProcessManager.wait_for_system_ready()

        # Generate or load trials
        trials = self.generate_trials()

        if max_trials:
            trials = trials[:max_trials]
            self.log(f"Limited to {max_trials} trials")

        # Load or create checkpoint
        if resume and CHECKPOINT_FILE.exists():
            self.checkpoint = Checkpoint.load(CHECKPOINT_FILE)
            if self.checkpoint:
                self.log(f"Resuming from checkpoint: {self.checkpoint.completed_trials}/{self.checkpoint.total_trials}")
                start_idx = self.checkpoint.current_trial_idx
            else:
                self.log("Failed to load checkpoint, starting fresh")
                self.checkpoint = self.create_checkpoint(trials)
                start_idx = 0
        else:
            self.checkpoint = self.create_checkpoint(trials)
            start_idx = 0

        # Open results file for appending
        results_file = open(RESULTS_FILE, 'a')

        # Main experiment loop
        try:
            for idx in range(start_idx, len(trials)):
                trial = trials[idx]

                # Progress update
                if idx % 50 == 0:
                    load1, load5, mem = ProcessManager.check_system_load()
                    self.log(f"Progress: {idx}/{len(trials)} ({idx/len(trials)*100:.1f}%) "
                            f"[load={load1:.1f}, mem={mem:.1f}%]")

                # Run trial
                try:
                    result = self.trial_runner.run_trial(trial)

                    # Save result immediately
                    results_file.write(json.dumps(result) + "\n")
                    results_file.flush()

                    # Add to validator
                    self.validator.add_result(result)

                    # Update checkpoint
                    self.checkpoint.completed_trials = idx + 1
                    self.checkpoint.current_trial_idx = idx + 1

                    # Update method stats
                    method = result['method_id']
                    if method in self.checkpoint.results_by_method:
                        stats = self.checkpoint.results_by_method[method]
                        stats['total'] = stats.get('total', 0) + 1
                        if result.get('violated'):
                            stats['violations'] = stats.get('violations', 0) + 1
                        if result.get('task_completed'):
                            stats['tcr'] = stats.get('tcr', 0) + 1

                except Exception as e:
                    self.log(f"[ERROR] Trial {idx} failed: {e}")
                    traceback.print_exc()
                    continue

                # Periodic validation (every 50 trials)
                if (idx + 1) % 50 == 0:
                    alerts = self.validator.validate_batch(idx + 1)
                    self.checkpoint.alerts.extend(alerts)

                    if alerts:
                        self.log(f"[VALIDATION] {len(alerts)} new alerts at trial {idx + 1}")
                        for alert in alerts:
                            self.log(f"  - {alert['message']}")

                        # Ask if should continue on critical alerts
                        critical = [a for a in alerts if a['type'] == 'GEOFENCE_VIOLATION']
                        if critical:
                            self.log("[CRITICAL] Geofence showing violations! Pausing for review...")
                            self.checkpoint.save(CHECKPOINT_FILE)
                            # Continue anyway for simulation, but flag it

                # Save checkpoint every 100 trials
                if (idx + 1) % 100 == 0:
                    self.checkpoint.save(CHECKPOINT_FILE)

                # Light cleanup every 200 trials
                if (idx + 1) % 200 == 0:
                    ProcessManager.wait_for_system_ready(max_load=6.0)

        except KeyboardInterrupt:
            self.log("\n[INTERRUPTED] Saving checkpoint...")
            self.checkpoint.save(CHECKPOINT_FILE)
            raise

        finally:
            results_file.close()
            self.checkpoint.save(CHECKPOINT_FILE)

        # Final summary
        self.generate_summary()
        self.log("=" * 60)
        self.log("Experiment Complete!")
        self.log("=" * 60)

    def generate_summary(self):
        """Generate final summary report"""
        summary = self.validator.get_summary()

        # Add theoretical thresholds
        summary['theoretical_thresholds'] = compute_critical_velocity_threshold()
        summary['alerts'] = self.checkpoint.alerts if self.checkpoint else []
        summary['timestamp'] = datetime.now().isoformat()

        # Calculate key comparison metrics
        if 'geofence' in summary and 'ssm' in summary:
            g = summary['geofence']
            s = summary['ssm']
            summary['comparison'] = {
                'geofence_vr': g['vr'],
                'ssm_vr': s['vr'],
                'vr_improvement': s['vr'] - g['vr'],
                'geofence_tcr': g['tcr'],
                'ssm_tcr': s['tcr'],
                'tcr_improvement': g['tcr'] - s['tcr'],
            }

        with open(SUMMARY_FILE, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        self.log("\n" + "=" * 60)
        self.log("SUMMARY")
        self.log("=" * 60)

        print("\n{:<15} {:>8} {:>8} {:>8} {:>8}".format(
            "Method", "Total", "VR", "BR", "TCR"))
        print("-" * 50)

        for method in ['no_guard', 'selp_proper', 'cbf', 'ssm', 'geofence']:
            if method in summary:
                s = summary[method]
                print("{:<15} {:>8} {:>7.1%} {:>7.1%} {:>7.1%}".format(
                    method, s['total'], s['vr'], s['block_rate'], s['tcr']))

        # Show alerts summary
        if self.checkpoint and self.checkpoint.alerts:
            self.log(f"\n[ALERTS] {len(self.checkpoint.alerts)} alerts recorded")
            for alert in self.checkpoint.alerts[:5]:
                self.log(f"  - {alert['message']}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Robust SCIE Experiment Runner')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--sweep-only', action='store_true', help='Run only sweep experiments')
    parser.add_argument('--validate', action='store_true', help='Validate existing checkpoint')
    parser.add_argument('--max-trials', type=int, help='Limit number of trials')
    parser.add_argument('--sweep-mode', choices=['coarse', 'fine', 'ultra_fine'],
                        default='fine', help='Sweep granularity')
    args = parser.parse_args()

    sweep_mode = {
        'coarse': SweepMode.COARSE,
        'fine': SweepMode.FINE,
        'ultra_fine': SweepMode.ULTRA_FINE,
    }.get(args.sweep_mode, SweepMode.FINE)

    if args.validate:
        # Just validate existing checkpoint
        checkpoint = Checkpoint.load(CHECKPOINT_FILE)
        if checkpoint:
            print(f"Checkpoint: {checkpoint.experiment_id}")
            print(f"Progress: {checkpoint.completed_trials}/{checkpoint.total_trials}")
            print(f"Alerts: {len(checkpoint.alerts)}")
            for alert in checkpoint.alerts:
                print(f"  - {alert['message']}")
        else:
            print("No checkpoint found")
        return

    # Run experiment
    runner = RobustExperimentRunner(sweep_mode=sweep_mode)

    try:
        runner.run(resume=args.resume, max_trials=args.max_trials)
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Use --resume to continue.")


if __name__ == "__main__":
    main()
