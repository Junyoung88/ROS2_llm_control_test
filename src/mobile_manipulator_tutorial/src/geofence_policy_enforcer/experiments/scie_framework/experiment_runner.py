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

from .config import ExperimentConfig, ScenarioConfig, MethodConfig, ExpectedResult
from .data_logger import (
    ExperimentLogger, TrialResult, TrialConditions,
    SafetyMetrics, PerformanceMetrics, MethodSpecificMetrics
)
from .metrics_collector import MetricsCollector, TimingContext


@dataclass
class ProcessHandles:
    """Handles to running processes"""
    gazebo: Optional[subprocess.Popen] = None
    nav2: Optional[subprocess.Popen] = None
    geofence: Optional[subprocess.Popen] = None


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

        # Send goal
        if waypoints and scenario.attack_type.value == "indirect":
            success, reason, completed = self.goal_sender.send_waypoints(
                waypoints, timeout=self.config.goal_timeout
            )
        else:
            success, reason = self.goal_sender.send_goal(
                goal_x, goal_y, timeout=self.config.goal_timeout
            )

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

        # Populate safety metrics
        result.safety = SafetyMetrics(
            decision=decision,
            expected_decision=expected,
            is_correct=is_correct,
            min_distance_to_forbidden=audit_metrics.get("min_distance_to_forbidden", float('inf')),
            violated_zone=audit_metrics.get("violated_zone")
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
