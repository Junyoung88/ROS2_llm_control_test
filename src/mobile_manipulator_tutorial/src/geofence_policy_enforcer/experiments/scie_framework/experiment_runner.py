"""
Factorial Experiment Runner

Executes full factorial experiments with:
- Automatic simulation management (Gazebo, Nav2, Geofence)
- Sequential trial execution with proper cleanup
- Progress tracking and resumption support
- Randomization for bias mitigation
"""

import os
import sys
import time
import signal
import random
import subprocess
import json
from datetime import datetime
from typing import List, Dict, Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
import threading

from .config import ExperimentConfig, ScenarioConfig, MethodConfig, ExpectedResult, AttackType
from .data_logger import (
    ExperimentLogger, TrialResult, TrialConditions,
    SafetyMetrics, PerformanceMetrics, MethodSpecificMetrics
)
from .metrics_collector import MetricsCollector, TimingContext
from .violation_monitor import ViolationMonitor, SimpleViolationTracker, ForbiddenZone


@dataclass
class ProcessHandles:
    """Handles to running processes"""
    gazebo: Optional[subprocess.Popen] = None
    nav2: Optional[subprocess.Popen] = None
    geofence: Optional[subprocess.Popen] = None
    attack: Optional[subprocess.Popen] = None  # S5/S6 attack nodes


class SimulationManager:
    """
    Manages simulation environment lifecycle.

    Handles starting, monitoring, and stopping:
    - Gazebo simulation
    - Nav2 navigation stack
    - Geofence goal_gate node
    """

    def __init__(self, workspace_path: str, config: ExperimentConfig):
        self.workspace_path = Path(workspace_path)
        self.config = config
        self.processes = ProcessHandles()
        self._current_method = None

    def start_gazebo(self) -> bool:
        """Start Gazebo simulation"""
        print("[INFO] Starting Gazebo simulation...")

        # Kill any existing instances
        self._kill_process("gz sim")

        # Source workspace and launch
        launch_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {self.workspace_path}/install/setup.bash && \
            ros2 launch mobile_manip_moveit_config mobile_manipulator.launch.py \
                use_sim_time:=true
        """

        self.processes.gazebo = subprocess.Popen(
            launch_cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        print(f"[INFO] Waiting for Gazebo to initialize ({self.config.gazebo_startup_timeout}s)...")
        time.sleep(self.config.gazebo_startup_timeout)

        if self.processes.gazebo.poll() is None:
            print("[INFO] Gazebo started successfully")
            return True
        else:
            print("[ERROR] Gazebo failed to start")
            return False

    def start_nav2(self) -> bool:
        """Start Nav2 navigation stack"""
        print("[INFO] Starting Nav2 navigation...")

        launch_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {self.workspace_path}/install/setup.bash && \
            ros2 launch mobile_manip_moveit_config navigation.launch.py \
                use_sim_time:=true
        """

        self.processes.nav2 = subprocess.Popen(
            launch_cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        print(f"[INFO] Waiting for Nav2 to initialize ({self.config.nav2_startup_timeout}s)...")
        time.sleep(self.config.nav2_startup_timeout)

        if self.processes.nav2.poll() is None:
            print("[INFO] Nav2 started successfully")
            return True
        else:
            print("[ERROR] Nav2 failed to start")
            return False

    def start_geofence(self, method: str) -> bool:
        """Start geofence goal_gate node with specified method"""
        print(f"[INFO] Starting geofence with method: {method}")

        self._current_method = method

        launch_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {self.workspace_path}/install/setup.bash && \
            ros2 launch geofence_policy_enforcer demo.launch.py \
                safety_method:={method}
        """

        self.processes.geofence = subprocess.Popen(
            launch_cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        print("[INFO] Waiting for geofence nodes (5s)...")
        time.sleep(5)

        if self.processes.geofence.poll() is None:
            print(f"[INFO] Geofence started with method: {method}")
            return True
        else:
            print("[ERROR] Geofence failed to start")
            return False

    def stop_all(self):
        """Stop all simulation processes"""
        print("[INFO] Cleaning up processes...")

        # Stop attack node first if running
        self.stop_attack_node()

        # Stop in reverse order
        for name, proc in [
            ("geofence", self.processes.geofence),
            ("nav2", self.processes.nav2),
            ("gazebo", self.processes.gazebo)
        ]:
            if proc and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass

        # Kill any remaining processes
        self._kill_process("gz sim")
        self._kill_process("ros2 launch")
        self._kill_process("goal_gate_node")
        self._kill_process("nav2")
        self._kill_process("attack_velocity_amplify")
        self._kill_process("attack_pose_spoofing")

        print(f"[INFO] Waiting for processes to terminate ({self.config.cleanup_timeout}s)...")
        time.sleep(self.config.cleanup_timeout)

        self.processes = ProcessHandles()
        print("[INFO] Cleanup complete")

    def _kill_process(self, pattern: str):
        """Kill processes matching pattern"""
        subprocess.run(
            f"pkill -9 -f '{pattern}'",
            shell=True,
            capture_output=True
        )

    def start_attack_node(self, attack_type: str, params: Dict = None) -> bool:
        """Start attack node for S5/S6 scenarios"""
        params = params or {}

        if attack_type == "velocity":
            # S5: Velocity manipulation attack
            amp_factor = params.get("amplification_factor", 2.0)
            target_x = params.get("target_x", -4.7)
            target_y = params.get("target_y", 5.6)

            launch_cmd = f"""
                source /opt/ros/jazzy/setup.bash && \
                source {self.workspace_path}/install/setup.bash && \
                ros2 run geofence_policy_enforcer attack_velocity_amplify \
                    --ros-args \
                    -p amplification_factor:={amp_factor} \
                    -p target_zone_x:={target_x} \
                    -p target_zone_y:={target_y}
            """
            print(f"[ATTACK] Starting velocity amplification attack (factor: {amp_factor}x)")

        elif attack_type == "spoofing":
            # S6: Pose spoofing attack
            spoof_x = params.get("spoof_offset_x", 3.0)
            spoof_y = params.get("spoof_offset_y", 0.0)

            launch_cmd = f"""
                source /opt/ros/jazzy/setup.bash && \
                source {self.workspace_path}/install/setup.bash && \
                ros2 run geofence_policy_enforcer attack_pose_spoofing \
                    --ros-args \
                    -p spoof_offset_x:={spoof_x} \
                    -p spoof_offset_y:={spoof_y}
            """
            print(f"[ATTACK] Starting pose spoofing attack (offset: {spoof_x}, {spoof_y})")

        else:
            print(f"[ATTACK] Unknown attack type: {attack_type}")
            return False

        self.processes.attack = subprocess.Popen(
            launch_cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        time.sleep(2)  # Wait for attack node to start

        if self.processes.attack.poll() is None:
            print(f"[ATTACK] Attack node started successfully")
            return True
        else:
            print(f"[ATTACK] Attack node failed to start")
            return False

    def stop_attack_node(self):
        """Stop attack node if running"""
        if self.processes.attack and self.processes.attack.poll() is None:
            try:
                os.killpg(os.getpgid(self.processes.attack.pid), signal.SIGTERM)
                print("[ATTACK] Attack node stopped")
            except ProcessLookupError:
                pass
            self.processes.attack = None

    def restart_with_method(self, method: str) -> bool:
        """Restart all processes with new safety method"""
        self.stop_all()
        time.sleep(3)

        if not self.start_gazebo():
            return False
        if not self.start_nav2():
            return False
        if not self.start_geofence(method):
            return False

        return True


class GoalSender:
    """Sends navigation goals to the robot"""

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)

    def send_goal(self, x: float, y: float, theta: float = 0.0,
                  timeout: float = 60.0) -> Tuple[bool, str]:
        """
        Send navigation goal and wait for result.

        Args:
            x, y: Goal position
            theta: Goal orientation (radians)
            timeout: Maximum wait time

        Returns:
            Tuple of (success, reason)
        """
        # Use navigate_to_pose_safe action (goes through goal_gate)
        goal_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {self.workspace_path}/install/setup.bash && \
            ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
                "{{pose: {{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}}}" \
                --feedback 2>&1
        """

        try:
            result = subprocess.run(
                goal_cmd,
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = result.stdout + result.stderr

            # Parse result
            if "Goal accepted" in output:
                if "SUCCEEDED" in output or "succeeded" in output:
                    return True, "Goal reached"
                elif "ABORTED" in output or "aborted" in output:
                    return False, "Goal aborted (rejected by safety system)"
                else:
                    return False, "Goal status unknown"
            else:
                return False, f"Goal not accepted: {output[:200]}"

        except subprocess.TimeoutExpired:
            return False, f"Goal timed out after {timeout}s"
        except Exception as e:
            return False, f"Error sending goal: {e}"

    def send_waypoints(self, waypoints: List[Tuple[float, float]],
                       timeout: float = 120.0) -> Tuple[bool, str, int]:
        """
        Send waypoints using FollowWaypoints action.

        Returns:
            Tuple of (success, reason, waypoints_completed)
        """
        if not waypoints:
            return False, "No waypoints provided", 0

        # Build poses array
        poses_str = ""
        for i, (x, y) in enumerate(waypoints):
            poses_str += f"{{header: {{frame_id: 'map'}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}"
            if i < len(waypoints) - 1:
                poses_str += ", "

        goal_cmd = f"""
            source /opt/ros/jazzy/setup.bash && \
            source {self.workspace_path}/install/setup.bash && \
            ros2 action send_goal /follow_waypoints_safe nav2_msgs/action/FollowWaypoints \
                "{{poses: [{poses_str}]}}" \
                --feedback 2>&1
        """

        try:
            result = subprocess.run(
                goal_cmd,
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = result.stdout + result.stderr

            # Count completed waypoints from output
            completed = output.count("Waypoint")

            if "SUCCEEDED" in output:
                return True, "All waypoints reached", len(waypoints)
            elif "ABORTED" in output:
                return False, "Waypoints aborted (rejected by safety system)", completed
            else:
                return False, "Unknown result", completed

        except subprocess.TimeoutExpired:
            return False, f"Waypoints timed out after {timeout}s", 0
        except Exception as e:
            return False, f"Error sending waypoints: {e}", 0


class FactorialExperimentRunner:
    """
    Runs full factorial experiments with all condition combinations.

    Features:
    - Automatic randomization of trial order
    - Progress tracking and ETA estimation
    - Resumption from checkpoints
    - Comprehensive logging
    """

    def __init__(self, config: ExperimentConfig, output_dir: str = None):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
            output_dir: Output directory (uses config default if None)
        """
        self.config = config
        self.output_dir = Path(output_dir or config.output_base_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate experiment ID
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create experiment-specific directory
        self.experiment_dir = self.output_dir / f"experiment_{self.experiment_id}"
        self.experiment_dir.mkdir(exist_ok=True)

        # Initialize components
        self.logger = ExperimentLogger(str(self.experiment_dir), self.experiment_id)
        self.sim_manager = SimulationManager(config.workspace_path, config)
        self.goal_sender = GoalSender(config.workspace_path)
        self.metrics = MetricsCollector()

        # Initialize violation monitor for tracking actual zone intrusions
        self.violation_monitor = ViolationMonitor(config.workspace_path)
        self.violation_tracker = SimpleViolationTracker()

        # Trial tracking
        self.completed_trials: List[str] = []
        self.failed_trials: List[str] = []
        self.current_method: Optional[str] = None

        # Progress tracking
        self.total_trials = 0
        self.trials_completed = 0
        self.start_time = 0.0

        # Callbacks
        self.on_trial_complete: Optional[Callable] = None
        self.on_method_change: Optional[Callable] = None

    def run(self,
            include_intensity: bool = False,
            include_noise: bool = False,
            randomize: bool = True,
            resume_from: str = None) -> Dict:
        """
        Run the full factorial experiment.

        Args:
            include_intensity: Include intensity factor
            include_noise: Include noise factor
            randomize: Randomize trial order
            resume_from: Path to checkpoint for resumption

        Returns:
            Summary statistics
        """
        # Save configuration
        self.config.save(str(self.experiment_dir / "config.yaml"))

        # Generate trial combinations
        combinations = self.config.get_all_combinations(
            include_intensity=include_intensity,
            include_noise=include_noise
        )

        self.total_trials = len(combinations)
        print(f"\n{'='*70}")
        print(f"FACTORIAL EXPERIMENT")
        print(f"{'='*70}")
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Total trials:  {self.total_trials}")
        print(f"Output:        {self.experiment_dir}")
        print(f"{'='*70}\n")

        # Randomize if requested
        if randomize:
            random.seed(self.config.random_seed)
            random.shuffle(combinations)

        # Load checkpoint if resuming
        if resume_from:
            self._load_checkpoint(resume_from)

        # Run trials
        self.start_time = time.time()

        try:
            for i, cond in enumerate(combinations):
                # Skip completed trials
                trial_key = self._get_trial_key(cond)
                if trial_key in self.completed_trials:
                    continue

                # Check if method changed
                if cond["method"] != self.current_method:
                    self._handle_method_change(cond["method"])

                # Run trial
                self._run_trial(cond, i + 1)

                # Save checkpoint periodically
                if (i + 1) % 10 == 0:
                    self._save_checkpoint()

        except KeyboardInterrupt:
            print("\n[INFO] Experiment interrupted by user")
            self._save_checkpoint()

        finally:
            self.sim_manager.stop_all()
            self.logger.save_full_results()
            summary = self.logger.save_summary()

        return summary

    def run_quick_test(self) -> Dict:
        """Run a quick test with minimal trials"""
        print("[INFO] Running quick test (S1 only, all methods, 1 rep)")

        # Override config for quick test
        original_reps = self.config.num_repetitions
        self.config.num_repetitions = 1

        # Filter to S1 only
        original_scenarios = self.config.scenarios
        self.config.scenarios = [s for s in original_scenarios if s.id == "S1"]

        try:
            result = self.run(include_intensity=False, include_noise=False)
        finally:
            # Restore original config
            self.config.num_repetitions = original_reps
            self.config.scenarios = original_scenarios

        return result

    def _run_trial(self, conditions: Dict, trial_num: int):
        """Run a single trial"""
        method = conditions["method"]
        scenario_id = conditions["scenario"]
        intensity = conditions.get("intensity", "medium")
        noise = conditions.get("noise_sigma", 0.15)
        rep = conditions["repetition"]

        # Find scenario config
        scenario = next((s for s in self.config.scenarios if s.id == scenario_id), None)
        if not scenario:
            print(f"[ERROR] Scenario {scenario_id} not found")
            return

        # Apply intensity variations
        goal_x = scenario.goal_x
        goal_y = scenario.goal_y
        waypoints = scenario.waypoints

        if intensity in scenario.intensity_variations:
            variations = scenario.intensity_variations[intensity]
            goal_x = variations.get("goal_x", goal_x)
            goal_y = variations.get("goal_y", goal_y)
            waypoints = variations.get("waypoints", waypoints)

        print(f"\n{'='*60}")
        print(f"Trial {trial_num}/{self.total_trials}")
        print(f"Method: {method}, Scenario: {scenario_id}, Rep: {rep}")
        print(f"Goal: ({goal_x:.2f}, {goal_y:.2f})")
        print(f"Expected: {scenario.expected_result.value}")
        print(f"{'='*60}")

        # Create trial result
        trial_conditions = TrialConditions(
            method=method,
            scenario=scenario_id,
            intensity=intensity,
            noise_sigma=noise,
            repetition=rep,
            random_seed=self.config.random_seed + rep
        )

        result = TrialResult(
            conditions=trial_conditions,
            goal_x=goal_x,
            goal_y=goal_y,
            waypoints=waypoints,
            started_at=datetime.now().isoformat()
        )

        # Start metrics collection
        self.metrics.reset()
        self.metrics.start_trial()
        self.metrics.start_decision_timer()

        # Start attack nodes for S5/S6 scenarios
        attack_launched = False
        if scenario.attack_type == AttackType.VELOCITY_MANIPULATION:
            # S5: Launch velocity amplification attack
            attack_params = {
                "amplification_factor": scenario.approach_velocity / 0.5,  # Based on intensity
                "target_x": goal_x,
                "target_y": goal_y,
            }
            attack_launched = self.sim_manager.start_attack_node("velocity", attack_params)
        elif scenario.attack_type == AttackType.POSE_SPOOFING:
            # S6: Launch pose spoofing attack
            attack_params = {
                "spoof_offset_x": 3.0,  # Make robot think it's 3m farther from zone
                "spoof_offset_y": 0.0,
            }
            attack_launched = self.sim_manager.start_attack_node("spoofing", attack_params)

        # Start violation monitoring
        trial_id = f"{method}_{scenario_id}_{rep}"
        self.violation_monitor.start_monitoring(trial_id, method, scenario_id)

        # Send goal
        if waypoints and scenario.attack_type.value == "indirect":
            success, reason, completed = self.goal_sender.send_waypoints(
                waypoints, timeout=self.config.goal_timeout
            )
        else:
            success, reason = self.goal_sender.send_goal(
                goal_x, goal_y, timeout=self.config.goal_timeout
            )

        # Stop violation monitoring and collect results
        violation_result = self.violation_monitor.stop_monitoring()

        # Stop attack node if launched
        if attack_launched:
            self.sim_manager.stop_attack_node()

        # End timing
        decision_latency = self.metrics.end_decision_timer()
        trial_duration = self.metrics.end_trial()

        # Collect metrics from audit log
        audit_metrics = self.metrics.collect_from_audit()

        # Determine decision - prioritize safety detection over navigation success
        decision = audit_metrics.get("decision", "unknown")
        reason_lower = reason.lower()

        # Check for rejection indicators in reason or audit
        if any(word in reason_lower for word in ["aborted", "rejected", "violation", "reject"]):
            decision = "reject"
        elif audit_metrics.get("violated_zone"):
            decision = "reject"
        elif decision == "unknown" and success:
            decision = "allow"
        elif decision == "unknown":
            decision = "reject" if not success else "allow"

        # Check if correct
        expected = scenario.expected_result.value

        # Special case: no_guard should allow everything
        if method == "no_guard":
            expected = "allow"

        is_correct = (decision == expected)

        # Get actual violation data from monitor
        actual_violation = False
        actual_violated_zone = None
        min_distance_actual = float('inf')

        if violation_result:
            actual_violation = violation_result.violated
            if violation_result.violated_zones:
                actual_violated_zone = violation_result.violated_zones[0]
            min_distance_actual = violation_result.min_distance_to_any_zone

        # Populate safety metrics with both predicted and actual violations
        result.safety = SafetyMetrics(
            decision=decision,
            expected_decision=expected,
            is_correct=is_correct,
            min_distance_to_forbidden=min(
                audit_metrics.get("min_distance_to_forbidden", float('inf')),
                min_distance_actual
            ),
            violated_zone=audit_metrics.get("violated_zone") or actual_violated_zone,
            actual_violation=actual_violation,
            actual_violation_count=violation_result.violation_count if violation_result else 0,
            actual_max_penetration=violation_result.max_penetration_depth if violation_result else 0.0
        )

        # Populate performance metrics
        result.performance = PerformanceMetrics(
            decision_latency_ms=decision_latency,
            total_execution_time_s=trial_duration,
            goal_reached=success
        )

        # Populate method-specific metrics
        result.method_specific = MethodSpecificMetrics(
            reprompt_count=audit_metrics.get("reprompt_count", 0),
            reprompt_text=audit_metrics.get("reprompt_text", ""),
            suggestions=audit_metrics.get("suggestions", []),
            automaton_state=audit_metrics.get("automaton_state", ""),
            noisy_perception_applied=audit_metrics.get("noisy_perception", False),
            proposition_changes=audit_metrics.get("proposition_changes", {})
        )

        result.reason = audit_metrics.get("reason", reason)
        result.success = True
        result.completed_at = datetime.now().isoformat()

        # Log result
        self.logger.log_trial(result)

        # Update tracking
        trial_key = self._get_trial_key(conditions)
        self.completed_trials.append(trial_key)
        self.trials_completed += 1

        # Print result
        status = "PASS" if is_correct else "FAIL"
        print(f"Result: {decision} - {'✓' if is_correct else '✗'} {status}")
        print(f"Reason: {result.reason[:60]}...")
        print(f"Latency: {decision_latency:.1f}ms")

        # Print actual violation info
        if actual_violation:
            print(f"⚠️  ACTUAL VIOLATION: Robot entered forbidden zone!")
            print(f"   - Penetration depth: {violation_result.max_penetration_depth:.3f}m")
            print(f"   - Violation count: {violation_result.violation_count}")
        else:
            print(f"✓  No actual zone intrusion (min dist: {min_distance_actual:.2f}m)")

        # Callback
        if self.on_trial_complete:
            self.on_trial_complete(result)

        # ETA
        self._print_progress()

    def _handle_method_change(self, new_method: str):
        """Handle transition to new safety method"""
        print(f"\n{'#'*60}")
        print(f"# Switching to method: {new_method}")
        print(f"{'#'*60}\n")

        if not self.sim_manager.restart_with_method(new_method):
            raise RuntimeError(f"Failed to start method: {new_method}")

        self.current_method = new_method

        if self.on_method_change:
            self.on_method_change(new_method)

    def _get_trial_key(self, conditions: Dict) -> str:
        """Generate unique key for trial conditions"""
        return f"{conditions['method']}_{conditions['scenario']}_{conditions.get('intensity', 'medium')}_{conditions.get('noise_sigma', 0.15)}_{conditions['repetition']}"

    def _save_checkpoint(self):
        """Save checkpoint for resumption"""
        checkpoint = {
            "experiment_id": self.experiment_id,
            "completed_trials": self.completed_trials,
            "failed_trials": self.failed_trials,
            "trials_completed": self.trials_completed,
            "timestamp": datetime.now().isoformat()
        }

        checkpoint_path = self.experiment_dir / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"[INFO] Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resumption"""
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        self.completed_trials = checkpoint.get("completed_trials", [])
        self.trials_completed = checkpoint.get("trials_completed", 0)

        print(f"[INFO] Resuming from checkpoint: {self.trials_completed} trials completed")

    def _print_progress(self):
        """Print progress and ETA"""
        elapsed = time.time() - self.start_time
        remaining = self.total_trials - self.trials_completed

        if self.trials_completed > 0:
            avg_time = elapsed / self.trials_completed
            eta_seconds = avg_time * remaining
            eta_minutes = eta_seconds / 60

            print(f"Progress: {self.trials_completed}/{self.total_trials} "
                  f"({100*self.trials_completed/self.total_trials:.1f}%) "
                  f"- ETA: {eta_minutes:.1f} min")


def run_experiment(config: ExperimentConfig = None,
                   quick_test: bool = False,
                   output_dir: str = None) -> Dict:
    """
    Convenience function to run experiments.

    Args:
        config: Experiment configuration (uses default if None)
        quick_test: Run quick test instead of full experiment
        output_dir: Output directory

    Returns:
        Summary statistics
    """
    if config is None:
        from .config import get_default_config
        config = get_default_config()

    runner = FactorialExperimentRunner(config, output_dir)

    if quick_test:
        return runner.run_quick_test()
    else:
        return runner.run()
