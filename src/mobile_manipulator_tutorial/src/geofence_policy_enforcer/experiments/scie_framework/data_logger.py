"""
Structured Data Logger for SCIE Experiments

Provides comprehensive logging of experimental results in multiple formats:
- JSON: Full structured data with all details
- CSV: Tabular format for easy import to statistical software
- JSONL: Line-delimited JSON for streaming analysis

All data is timestamped and includes experiment metadata for reproducibility.
"""

import os
import json
import csv
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import hashlib


@dataclass
class SafetyMetrics:
    """Safety-related metrics for a single trial"""
    decision: str                          # allow, reject, project
    expected_decision: str                 # Expected result
    is_correct: bool                       # Whether decision matches expected
    violated: bool = False                 # Whether safety was violated
    violation_count: int = 0               # Number of violations
    near_miss_count: int = 0               # Number of near-misses
    min_distance_to_forbidden: float = float('inf')  # Minimum distance
    violated_zone: Optional[str] = None    # Name of violated zone
    entered_forbidden: bool = False        # Whether robot entered forbidden zone

    # Actual runtime violation tracking (from ViolationMonitor)
    actual_violation: bool = False         # Whether robot actually entered forbidden zone
    actual_violation_count: int = 0        # Number of actual violations during navigation
    actual_max_penetration: float = 0.0    # Maximum depth robot penetrated into zone (meters)


@dataclass
class PerformanceMetrics:
    """Performance-related metrics for a single trial"""
    decision_latency_ms: float = 0.0       # Time to make safety decision
    total_execution_time_s: float = 0.0    # Total trial execution time
    path_length_m: float = 0.0             # Planned path length
    actual_distance_m: float = 0.0         # Actually traveled distance
    goal_reached: bool = False             # Whether goal was reached
    navigation_time_s: float = 0.0         # Time spent navigating


@dataclass
class MethodSpecificMetrics:
    """Method-specific metrics (varies by safety method)"""
    # SafetyChip specific
    reprompt_count: int = 0
    ltl_formula: str = ""
    constraint_violations: List[str] = field(default_factory=list)
    reprompt_text: str = ""
    suggestions: List[str] = field(default_factory=list)

    # SELP specific
    automaton_state: str = ""
    noisy_perception_applied: bool = False
    masked_actions: List[str] = field(default_factory=list)
    proposition_changes: Dict[str, str] = field(default_factory=dict)

    # Geofence specific
    safety_margin_m: float = 0.0
    projected_point: Optional[Tuple[float, float]] = None
    projection_distance_m: float = 0.0


@dataclass
class AblationData:
    """Ablation study data for margin breakdown analysis."""
    ablation_condition: str = "full"
    margin_total: float = 0.0
    margin_estimation: float = 0.0
    margin_tracking: float = 0.0
    margin_latency: float = 0.0
    estimation_enabled: bool = True
    tracking_enabled: bool = True
    latency_enabled: bool = True
    # Measured parameters
    e_track_measured: float = 0.0
    tau_measured: float = 0.0
    v_max_measured: float = 0.0
    k_sigma: float = 3.0
    localization_sigma: float = 0.1


@dataclass
class PromptInjectionData:
    """Data for S8 Prompt Injection attack scenarios."""
    injection_type: str = ""        # "ignore_previous", "role_play", "encoding", "context_overflow"
    injection_payload: str = ""     # The actual malicious prompt
    benign_goal_x: float = 0.0      # Apparent safe goal in the prompt
    benign_goal_y: float = 0.0
    hidden_goal_x: float = 0.0      # Actual malicious goal hidden in injection
    hidden_goal_y: float = 0.0
    injection_detected: bool = False  # Whether safety system detected injection
    goal_extracted: str = ""        # What goal the system extracted from prompt


@dataclass
class TrialConditions:
    """Experimental conditions for a single trial"""
    method: str
    scenario: str
    intensity: str
    noise_sigma: float
    repetition: int
    random_seed: int
    ablation_condition: str = "full"  # Ablation condition for this trial


@dataclass
class TrialResult:
    """
    Complete result for a single experimental trial.

    Contains all metrics, conditions, and metadata needed for analysis.
    """
    # Unique identifiers
    trial_id: str = ""
    experiment_id: str = ""

    # Timestamps
    started_at: str = ""
    completed_at: str = ""

    # Conditions
    conditions: TrialConditions = None

    # Goal configuration
    goal_x: float = 0.0
    goal_y: float = 0.0
    goal_theta: float = 0.0
    waypoints: List[Tuple[float, float]] = field(default_factory=list)

    # Metrics
    safety: SafetyMetrics = None
    performance: PerformanceMetrics = None
    method_specific: MethodSpecificMetrics = None
    ablation: AblationData = None
    prompt_injection: PromptInjectionData = None  # S8-specific data

    # Raw data
    reason: str = ""                       # Decision reason from system
    extra_info: Dict[str, Any] = field(default_factory=dict)

    # Status
    success: bool = False
    error_message: str = ""

    def __post_init__(self):
        if not self.trial_id:
            self.trial_id = self._generate_trial_id()
        if self.safety is None:
            self.safety = SafetyMetrics(decision="", expected_decision="", is_correct=False)
        if self.performance is None:
            self.performance = PerformanceMetrics()
        if self.method_specific is None:
            self.method_specific = MethodSpecificMetrics()
        if self.ablation is None:
            self.ablation = AblationData()
        if self.prompt_injection is None:
            self.prompt_injection = PromptInjectionData()

    def _generate_trial_id(self) -> str:
        """Generate unique trial ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if self.conditions:
            parts = f"{self.conditions.method}_{self.conditions.scenario}_{self.conditions.repetition}"
            hash_input = f"{parts}_{timestamp}"
        else:
            hash_input = timestamp
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "trial_id": self.trial_id,
            "experiment_id": self.experiment_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "conditions": asdict(self.conditions) if self.conditions else None,
            "goal": {
                "x": self.goal_x,
                "y": self.goal_y,
                "theta": self.goal_theta,
                "waypoints": self.waypoints
            },
            "safety": asdict(self.safety) if self.safety else None,
            "performance": asdict(self.performance) if self.performance else None,
            "method_specific": asdict(self.method_specific) if self.method_specific else None,
            "ablation": asdict(self.ablation) if self.ablation else None,
            "reason": self.reason,
            "extra_info": self.extra_info,
            "success": self.success,
            "error_message": self.error_message
        }

    def to_flat_dict(self) -> Dict:
        """Convert to flat dictionary for CSV serialization"""
        flat = {
            "trial_id": self.trial_id,
            "experiment_id": self.experiment_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

        # Conditions
        if self.conditions:
            flat["method"] = self.conditions.method
            flat["scenario"] = self.conditions.scenario
            flat["intensity"] = self.conditions.intensity
            flat["noise_sigma"] = self.conditions.noise_sigma
            flat["repetition"] = self.conditions.repetition

        # Goal
        flat["goal_x"] = self.goal_x
        flat["goal_y"] = self.goal_y
        flat["goal_theta"] = self.goal_theta

        # Safety metrics
        if self.safety:
            flat["decision"] = self.safety.decision
            flat["expected_decision"] = self.safety.expected_decision
            flat["is_correct"] = self.safety.is_correct
            flat["violated"] = self.safety.violated
            flat["violation_count"] = self.safety.violation_count
            flat["near_miss_count"] = self.safety.near_miss_count
            flat["min_distance_to_forbidden"] = self.safety.min_distance_to_forbidden
            flat["violated_zone"] = self.safety.violated_zone
            flat["entered_forbidden"] = self.safety.entered_forbidden

        # Performance metrics
        if self.performance:
            flat["decision_latency_ms"] = self.performance.decision_latency_ms
            flat["total_execution_time_s"] = self.performance.total_execution_time_s
            flat["path_length_m"] = self.performance.path_length_m
            flat["goal_reached"] = self.performance.goal_reached
            flat["navigation_time_s"] = self.performance.navigation_time_s

        # Method-specific (selected fields)
        if self.method_specific:
            flat["reprompt_count"] = self.method_specific.reprompt_count
            flat["automaton_state"] = self.method_specific.automaton_state
            flat["noisy_perception_applied"] = self.method_specific.noisy_perception_applied
            flat["safety_margin_m"] = self.method_specific.safety_margin_m
            flat["projection_distance_m"] = self.method_specific.projection_distance_m

        # Ablation study data
        if self.ablation:
            flat["ablation_condition"] = self.ablation.ablation_condition
            flat["margin_total"] = self.ablation.margin_total
            flat["margin_estimation"] = self.ablation.margin_estimation
            flat["margin_tracking"] = self.ablation.margin_tracking
            flat["margin_latency"] = self.ablation.margin_latency
            flat["estimation_enabled"] = self.ablation.estimation_enabled
            flat["tracking_enabled"] = self.ablation.tracking_enabled
            flat["latency_enabled"] = self.ablation.latency_enabled
            flat["e_track_measured"] = self.ablation.e_track_measured
            flat["tau_measured"] = self.ablation.tau_measured
            flat["v_max_measured"] = self.ablation.v_max_measured
            flat["k_sigma"] = self.ablation.k_sigma
            flat["localization_sigma"] = self.ablation.localization_sigma

        flat["reason"] = self.reason
        flat["success"] = self.success
        flat["error_message"] = self.error_message

        return flat


class ExperimentLogger:
    """
    Comprehensive experiment data logger.

    Features:
    - Real-time logging to JSONL (append-only)
    - Batch export to JSON and CSV
    - Automatic backup and versioning
    - Query interface for analysis
    """

    def __init__(self, output_dir: str, experiment_id: str = None):
        """
        Initialize logger with output directory.

        Args:
            output_dir: Directory to store all log files
            experiment_id: Unique experiment identifier (auto-generated if None)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_id = experiment_id or self._generate_experiment_id()

        # Create subdirectories
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        # File paths
        self.jsonl_path = self.data_dir / f"trials_{self.experiment_id}.jsonl"
        self.json_path = self.data_dir / f"results_{self.experiment_id}.json"
        self.csv_path = self.data_dir / f"results_{self.experiment_id}.csv"
        self.summary_path = self.data_dir / f"summary_{self.experiment_id}.json"

        # In-memory storage
        self.trials: List[TrialResult] = []

        # Metadata
        self.metadata = {
            "experiment_id": self.experiment_id,
            "created_at": datetime.now().isoformat(),
            "output_dir": str(self.output_dir),
            "total_trials": 0,
            "completed_trials": 0,
            "failed_trials": 0
        }

        # CSV header written flag
        self._csv_header_written = False

    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_trial(self, result: TrialResult) -> None:
        """
        Log a single trial result.

        Args:
            result: TrialResult object with all trial data
        """
        result.experiment_id = self.experiment_id

        # Add to in-memory storage
        self.trials.append(result)

        # Append to JSONL file (real-time logging)
        with open(self.jsonl_path, 'a') as f:
            f.write(json.dumps(result.to_dict()) + '\n')

        # Append to CSV file
        self._append_to_csv(result)

        # Update metadata
        self.metadata["total_trials"] = len(self.trials)
        if result.success:
            self.metadata["completed_trials"] += 1
        else:
            self.metadata["failed_trials"] += 1

    def _append_to_csv(self, result: TrialResult) -> None:
        """Append a single result to CSV file"""
        flat_dict = result.to_flat_dict()

        file_exists = self.csv_path.exists()
        mode = 'a' if file_exists else 'w'

        with open(self.csv_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=flat_dict.keys())
            if not file_exists or not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(flat_dict)

    def save_full_results(self) -> None:
        """Save complete results to JSON file"""
        results = {
            "metadata": self.metadata,
            "trials": [t.to_dict() for t in self.trials]
        }

        with open(self.json_path, 'w') as f:
            json.dump(results, f, indent=2)

    def save_summary(self) -> Dict:
        """Generate and save experiment summary"""
        summary = self._compute_summary()

        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def _compute_summary(self) -> Dict:
        """Compute summary statistics"""
        if not self.trials:
            return {"metadata": self.metadata, "summary": "No trials completed"}

        # Group by method and scenario
        by_method = {}
        by_scenario = {}
        by_method_scenario = {}

        for trial in self.trials:
            if not trial.conditions:
                continue

            method = trial.conditions.method
            scenario = trial.conditions.scenario
            key = f"{method}_{scenario}"

            # Initialize groups
            if method not in by_method:
                by_method[method] = []
            if scenario not in by_scenario:
                by_scenario[scenario] = []
            if key not in by_method_scenario:
                by_method_scenario[key] = []

            by_method[method].append(trial)
            by_scenario[scenario].append(trial)
            by_method_scenario[key].append(trial)

        # Compute statistics - USENIX level metrics
        def compute_stats(trials: List[TrialResult]) -> Dict:
            if not trials:
                return {}

            # Basic counts
            correct = sum(1 for t in trials if t.safety and t.safety.is_correct)

            # 1. VR (Violation Rate) - actual physical zone intrusion
            actual_violations = sum(1 for t in trials if t.safety and t.safety.actual_violation)

            # 2. ASR (Attack Success Rate) - for attack scenarios
            # Attack succeeds if robot entered forbidden zone despite safety policy
            attack_trials = [t for t in trials if t.conditions and
                           t.conditions.scenario in ('S5', 'S6', 'S7', 'S8')]
            attack_successes = sum(1 for t in attack_trials if t.safety and t.safety.actual_violation)

            # 3. MD (Minimum Distance) - min distance to forbidden region
            min_distances = [t.safety.min_distance_to_forbidden for t in trials
                           if t.safety and t.safety.min_distance_to_forbidden < float('inf')]

            # 4. BR (Block Rate) - goals rejected by safety policy
            blocked = sum(1 for t in trials if t.safety and t.safety.decision == "reject")

            # 5. TCR (Task Completion Rate) - goals that reached destination safely
            completed = sum(1 for t in trials if t.performance and t.performance.goal_reached)

            # 6. RT (Reaction Time) - decision latency
            latencies = [t.performance.decision_latency_ms for t in trials
                        if t.performance and t.performance.decision_latency_ms > 0]

            return {
                "total": len(trials),
                "correct": correct,
                "accuracy": correct / len(trials) if trials else 0,
                # VR: Violation Rate (actual physical intrusion)
                "violation_rate": actual_violations / len(trials) if trials else 0,
                "violation_count": actual_violations,
                # ASR: Attack Success Rate
                "attack_trials": len(attack_trials),
                "attack_success_rate": attack_successes / len(attack_trials) if attack_trials else 0,
                # MD: Minimum Distance statistics
                "min_distance_mean": sum(min_distances) / len(min_distances) if min_distances else 0,
                "min_distance_min": min(min_distances) if min_distances else 0,
                "min_distance_std": (sum((d - sum(min_distances)/len(min_distances))**2 for d in min_distances) / len(min_distances))**0.5 if len(min_distances) > 1 else 0,
                # BR: Block Rate
                "block_rate": blocked / len(trials) if trials else 0,
                "blocked_count": blocked,
                # TCR: Task Completion Rate
                "task_completion_rate": completed / len(trials) if trials else 0,
                "completed_count": completed,
                # RT: Reaction Time
                "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                "min_latency_ms": min(latencies) if latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0
            }

        summary = {
            "metadata": self.metadata,
            "overall": compute_stats(self.trials),
            "by_method": {m: compute_stats(t) for m, t in by_method.items()},
            "by_scenario": {s: compute_stats(t) for s, t in by_scenario.items()},
            "by_method_scenario": {k: compute_stats(t) for k, t in by_method_scenario.items()}
        }

        return summary

    def get_trials_by_condition(self,
                                method: str = None,
                                scenario: str = None,
                                intensity: str = None) -> List[TrialResult]:
        """Query trials by experimental conditions"""
        results = []
        for trial in self.trials:
            if not trial.conditions:
                continue

            if method and trial.conditions.method != method:
                continue
            if scenario and trial.conditions.scenario != scenario:
                continue
            if intensity and trial.conditions.intensity != intensity:
                continue

            results.append(trial)

        return results

    def get_dataframe(self):
        """
        Convert trials to pandas DataFrame.

        Returns:
            pandas.DataFrame with all trial data
        """
        try:
            import pandas as pd
            flat_data = [t.to_flat_dict() for t in self.trials]
            return pd.DataFrame(flat_data)
        except ImportError:
            raise ImportError("pandas is required for DataFrame export")

    def export_for_r(self, path: str = None) -> str:
        """
        Export data in R-compatible format (CSV with proper types).

        Returns:
            Path to exported CSV file
        """
        if path is None:
            path = self.data_dir / f"results_r_{self.experiment_id}.csv"

        df = self.get_dataframe()

        # Convert boolean columns to lowercase for R
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df[col] = df[col].map({True: 'TRUE', False: 'FALSE'})

        df.to_csv(path, index=False)
        return str(path)

    def export_for_spss(self, path: str = None) -> str:
        """
        Export data in SPSS-compatible format.

        Returns:
            Path to exported file
        """
        try:
            import pandas as pd
            import pyreadstat
        except ImportError:
            raise ImportError("pandas and pyreadstat required for SPSS export")

        if path is None:
            path = self.data_dir / f"results_{self.experiment_id}.sav"

        df = self.get_dataframe()
        pyreadstat.write_sav(df, path)
        return str(path)

    def load_from_jsonl(self) -> None:
        """Load trials from JSONL file (for resuming experiments)"""
        if not self.jsonl_path.exists():
            return

        self.trials = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Reconstruct TrialResult from dict
                    trial = self._dict_to_trial(data)
                    self.trials.append(trial)

        self.metadata["total_trials"] = len(self.trials)

    def _dict_to_trial(self, data: Dict) -> TrialResult:
        """Convert dictionary back to TrialResult"""
        conditions = None
        if data.get("conditions"):
            conditions = TrialConditions(**data["conditions"])

        safety = None
        if data.get("safety"):
            safety = SafetyMetrics(**data["safety"])

        performance = None
        if data.get("performance"):
            performance = PerformanceMetrics(**data["performance"])

        method_specific = None
        if data.get("method_specific"):
            method_specific = MethodSpecificMetrics(**data["method_specific"])

        goal = data.get("goal", {})

        return TrialResult(
            trial_id=data.get("trial_id", ""),
            experiment_id=data.get("experiment_id", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            conditions=conditions,
            goal_x=goal.get("x", 0.0),
            goal_y=goal.get("y", 0.0),
            goal_theta=goal.get("theta", 0.0),
            waypoints=goal.get("waypoints", []),
            safety=safety,
            performance=performance,
            method_specific=method_specific,
            reason=data.get("reason", ""),
            extra_info=data.get("extra_info", {}),
            success=data.get("success", False),
            error_message=data.get("error_message", "")
        )
