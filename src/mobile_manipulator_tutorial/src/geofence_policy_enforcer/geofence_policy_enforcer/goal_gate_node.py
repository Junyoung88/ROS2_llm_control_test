#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Goal Gate Node - Policy Enforcement Point for navigation goals.

Acts as an action proxy for NavigateToPose and FollowWaypoints.
Evaluates goals against multiple safety methods:
- no_guard: Pass all goals (baseline)
- geofence: Geometric zone checking with safety margins
- safetychip: LTL-based constraint monitoring with Reprompting
- selp: Automaton-based constraint checking with Noisy Perception

Evaluates goals and either:
- ALLOW: Forward to Nav2
- REJECT: Abort with reason
- PROJECT: Modify goal to nearest safe point and forward

Proper Version Features:
- SafetyChip: LTL Monitor + Action Pruning + Reprompting with explanation
- SELP: LTL Automaton + Action Masking + Noisy Zone Perception
"""

import math
import json
import random
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String, Float64
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import NavigateToPose, FollowWaypoints
# Note: ComputePathToPose is an ACTION, not a service. Path checking feature disabled.
# from nav2_msgs.action import ComputePathToPose
from std_srvs.srv import Trigger, SetBool
from visualization_msgs.msg import MarkerArray

from tf_transformations import euler_from_quaternion, quaternion_from_euler

from .geofence_core import (
    GeofencePolicy, PolicyAction, PolicyDecision, ZoneType,
    VelocityMonitor, LatencyMonitor, TrackingErrorMonitor
)

# Import new safety baselines (SELP, CBF, Static Margin, SSM)
from .safety_baselines import (
    SELPSpatialSafety, CBFSpatialSafety, StaticMarginSafety, SSMSpatialSafety,
    SafetyDecision, SafetyResult
)


# ============================================================================
# SafetyChip Proper: LTL Constraint Monitor with Progression Semantics
# ============================================================================

class ConstraintType(Enum):
    """Supported LTL constraint pattern types"""
    ALWAYS_AVOID = auto()       # G(!p) - always avoid p
    ALWAYS_MAINTAIN = auto()    # G(p) - always maintain p
    PRECEDENCE = auto()         # (!q) U p - p before q
    TRIGGER_RESPONSE = auto()   # G(p -> F(q)) - if p then eventually q
    EVENTUALLY = auto()         # F(p) - eventually reach p


class MonitorStatus(Enum):
    """Status of the constraint monitor"""
    SATISFIED = auto()      # Constraint is satisfied (for safety: not violated)
    VIOLATED = auto()       # Constraint has been violated
    PENDING = auto()        # Not yet satisfied but not violated (for liveness)
    UNKNOWN = auto()


@dataclass
class LTLConstraint:
    """A single LTL constraint with metadata"""
    original_nl: str
    ltl_formula: str
    constraint_type: ConstraintType
    propositions: Set[str] = field(default_factory=set)
    parameters: Dict = field(default_factory=dict)


@dataclass
class ViolationInfo:
    """Information about a constraint violation (Proper version)"""
    constraint: LTLConstraint
    violated_proposition: str
    proposition_change: Tuple[bool, bool]  # (before, after) values
    explanation: str
    suggested_action: str  # Suggested alternative


@dataclass
class MonitorState:
    """Current state of the monitor"""
    status: MonitorStatus
    violated_constraints: List[ViolationInfo] = field(default_factory=list)
    pending_obligations: List[str] = field(default_factory=list)
    step_count: int = 0


@dataclass
class ExplanationResult:
    """Result of violation explanation for reprompting (SafetyChip key feature)"""
    summary: str                          # Short summary
    details: List[str]                    # Detailed explanation lines
    violated_constraints_nl: List[str]    # Original NL constraints violated
    proposition_changes: Dict[str, Tuple[bool, bool]]  # Prop -> (before, after)
    suggestions: List[str]                # Suggested alternatives
    reprompt_text: str                    # Text to add to LLM reprompt


class PatternMonitor:
    """
    Pattern-based LTL monitor using progression semantics.

    For each supported pattern, we track a simple state machine:

    G(!p) - Always Avoid:
        Transition: if p becomes true -> violated

    G(p) - Always Maintain:
        Transition: if p becomes false -> violated

    (!q) U p - Precedence:
        Transition: if p becomes true -> satisfied
                    if q becomes true before p -> violated

    G(p -> F(q)) - Trigger Response:
        Transition: if p becomes true -> add pending q
                    if q becomes true -> clear pending
    """

    def __init__(self, constraint: LTLConstraint):
        self.constraint = constraint
        self.pattern_type = constraint.constraint_type
        self.params = constraint.parameters

        # State variables
        self._violated = False
        self._satisfied = False
        self._pending_responses: List[int] = []
        self._step_count = 0

    def reset(self) -> None:
        """Reset monitor to initial state"""
        self._violated = False
        self._satisfied = False
        self._pending_responses = []
        self._step_count = 0

    def would_violate(self, current_props: Dict[str, bool],
                      proposed_props: Dict[str, bool]) -> Tuple[bool, Optional[ViolationInfo]]:
        """Check if proposed propositions would cause a violation WITHOUT advancing state"""
        if self._violated:
            return True, self._make_violation_info("already_violated", current_props, proposed_props)

        if self.pattern_type == ConstraintType.ALWAYS_AVOID:
            avoid_prop = f"in_{self.params['avoid_zone']}"
            if proposed_props.get(avoid_prop, False):
                return True, self._make_violation_info(avoid_prop, current_props, proposed_props)

        elif self.pattern_type == ConstraintType.ALWAYS_MAINTAIN:
            maintain_prop = f"in_{self.params.get('maintain', 'safe_zone')}"
            if not proposed_props.get(maintain_prop, True):
                return True, self._make_violation_info(maintain_prop, current_props, proposed_props,
                    reason=f"Would fail to maintain {maintain_prop}")

        elif self.pattern_type == ConstraintType.PRECEDENCE:
            if not self._satisfied:
                first_prop = f"in_{self.params['first']}"
                second_prop = f"in_{self.params['second']}"
                if proposed_props.get(second_prop, False) and not proposed_props.get(first_prop, False):
                    return True, self._make_violation_info(second_prop, current_props, proposed_props,
                        reason=f"Would enter {self.params['second']} before visiting {self.params['first']}")

        return False, None

    def _make_violation_info(self, prop: str, current_props: Dict[str, bool],
                             proposed_props: Dict[str, bool], reason: str = None) -> ViolationInfo:
        """Create violation info with explanation"""
        before = current_props.get(prop, False)
        after = proposed_props.get(prop, False)

        if reason is None:
            if self.pattern_type == ConstraintType.ALWAYS_AVOID:
                reason = f"Would enter forbidden zone: {self.params['avoid_zone']}"
            elif self.pattern_type == ConstraintType.PRECEDENCE:
                reason = f"Would visit {self.params['second']} before {self.params['first']}"
            else:
                reason = f"Constraint violated: {self.constraint.ltl_formula}"

        # Generate suggested action
        if self.pattern_type == ConstraintType.ALWAYS_AVOID:
            suggestion = f"Choose a goal outside of {self.params['avoid_zone']} zone"
        elif self.pattern_type == ConstraintType.PRECEDENCE:
            suggestion = f"Visit {self.params['first']} first"
        else:
            suggestion = "Choose a different goal location"

        return ViolationInfo(
            constraint=self.constraint,
            violated_proposition=prop,
            proposition_change=(before, after),
            explanation=reason,
            suggested_action=suggestion
        )


class CompositeMonitor:
    """
    Composite monitor that combines multiple pattern monitors.
    All constraints must be satisfied (conjunction).
    """

    def __init__(self, constraints: List[LTLConstraint]):
        self.monitors: List[PatternMonitor] = []
        for c in constraints:
            self.monitors.append(PatternMonitor(c))

    def reset(self) -> None:
        """Reset all monitors"""
        for m in self.monitors:
            m.reset()

    def would_violate(self, current_props: Dict[str, bool],
                      proposed_props: Dict[str, bool]) -> Tuple[bool, List[ViolationInfo]]:
        """Check if proposed props would violate any constraint"""
        all_violations = []

        for m in self.monitors:
            would_viol, violation = m.would_violate(current_props, proposed_props)
            if would_viol and violation:
                all_violations.append(violation)

        return len(all_violations) > 0, all_violations


class SafetyChipMonitor:
    """
    SafetyChip Proper: LTL Monitor with Reprompting.

    Key features:
    1. LTL Monitor with progression semantics
    2. Action Pruning (pre-check before execution)
    3. Reprompting with detailed violation explanation

    The reprompt_text is generated to inform the LLM WHY an action was rejected,
    enabling it to make a better decision on the next attempt.
    """

    def __init__(self, constraints: List[LTLConstraint]):
        self.constraints = constraints
        self._composite_monitor = CompositeMonitor(constraints)
        self._reprompt_count = 0
        self._max_reprompts = 5

    def reset(self):
        """Reset monitor state"""
        self._composite_monitor.reset()
        self._reprompt_count = 0

    def would_violate(self, current_props: Dict[str, bool],
                      proposed_props: Dict[str, bool] = None) -> Tuple[bool, List[ViolationInfo]]:
        """
        Check if propositions would violate any constraint.

        Args:
            current_props: Current proposition values (before action)
            proposed_props: Proposed proposition values (after action)
                           If None, uses current_props for both (simplified check)

        Returns:
            Tuple of (would_violate, list of violations)
        """
        if proposed_props is None:
            proposed_props = current_props
        return self._composite_monitor.would_violate(current_props, proposed_props)

    def explain_violation(self, violations: List[ViolationInfo],
                         current_props: Dict[str, bool],
                         proposed_props: Dict[str, bool]) -> ExplanationResult:
        """
        Generate detailed explanation for constraint violation.

        This is a key SafetyChip feature: explaining WHY an action is unsafe
        to enable informed reprompting.
        """
        if not violations:
            return ExplanationResult(
                summary="No violation",
                details=[],
                violated_constraints_nl=[],
                proposition_changes={},
                suggestions=[],
                reprompt_text=""
            )

        details = []
        nl_constraints = []
        prop_changes = {}
        suggestions = []

        for v in violations:
            details.append(f"- {v.explanation}")
            nl_constraints.append(v.constraint.original_nl)
            prop_changes[v.violated_proposition] = v.proposition_change
            if v.suggested_action not in suggestions:
                suggestions.append(v.suggested_action)

        # Create summary
        if len(violations) == 1:
            summary = violations[0].explanation
        else:
            summary = f"Multiple constraint violations ({len(violations)})"

        # Create reprompt text for LLM
        reprompt_lines = [
            "SAFETY VIOLATION DETECTED:",
            f"The proposed navigation goal would violate the following constraint(s):",
        ]
        for nl in nl_constraints:
            reprompt_lines.append(f'  - "{nl}"')

        reprompt_lines.append("\nProposition changes that cause violation:")
        for prop, (before, after) in prop_changes.items():
            reprompt_lines.append(f"  - {prop}: {before} -> {after}")

        reprompt_lines.append("\nSuggested alternatives:")
        for s in suggestions:
            reprompt_lines.append(f"  - {s}")

        reprompt_lines.append("\nPlease propose a different goal that does not violate these constraints.")

        return ExplanationResult(
            summary=summary,
            details=details,
            violated_constraints_nl=nl_constraints,
            proposition_changes=prop_changes,
            suggestions=suggestions,
            reprompt_text="\n".join(reprompt_lines)
        )

    def handle_reprompt(self) -> bool:
        """
        Handle reprompt logic. Returns True if reprompt should be allowed.
        """
        self._reprompt_count += 1
        return self._reprompt_count <= self._max_reprompts

    @property
    def reprompt_count(self) -> int:
        return self._reprompt_count


# ============================================================================
# SELP Proper: LTL Automaton with Action Masking and Noisy Perception
# ============================================================================

class AutomatonState(Enum):
    """State of LTL automaton monitoring"""
    ACCEPTING = auto()      # Currently in accepting state
    NON_ACCEPTING = auto()  # Not in accepting state but can still reach one
    VIOLATED = auto()       # Constraint permanently violated
    UNKNOWN = auto()


@dataclass
class SELPMonitorState:
    """State of the SELP runtime monitor"""
    automaton_state: str              # Current automaton state name
    is_accepting: bool                # Whether current state is accepting
    can_reach_accepting: bool         # Whether accepting state is reachable
    status: AutomatonState            # Overall status
    noisy_perception_applied: bool = False  # Whether noisy perception was used


@dataclass
class ActionMaskInfo:
    """Information about why an action was masked (SELP logging)"""
    goal_x: float
    goal_y: float
    masked: bool
    reason: str
    proposition_changes: Dict[str, Tuple[bool, bool]] = field(default_factory=dict)
    automaton_status_before: AutomatonState = AutomatonState.UNKNOWN
    automaton_status_after: AutomatonState = AutomatonState.UNKNOWN


class SELPAutomaton:
    """
    SELP Proper: LTL Automaton with Noisy Zone Perception.

    Key features:
    1. LTL Automaton with state tracking (ACCEPTING, NON_ACCEPTING, VIOLATED)
    2. Constrained Decoding (mask invalid actions)
    3. Noisy Zone Perception (simulate LLM's imperfect zone understanding)

    Noisy Perception:
    - LLMs don't have perfect geometric understanding
    - Simulates this by adding noise to zone boundary perception
    - Uses estimated position with uncertainty (sigma = localization_sigma)
    """

    def __init__(self, constraints: List[LTLConstraint],
                 perception_noise_sigma: float = 0.3):
        """
        Args:
            constraints: LTL constraints to enforce
            perception_noise_sigma: Standard deviation for noisy perception (meters)
        """
        self.constraints = constraints
        self.perception_noise_sigma = perception_noise_sigma

        # Automaton state tracking
        self._current_state = "q0"  # Initial accepting state
        self._status = AutomatonState.ACCEPTING
        self._step_count = 0

        # Pattern states for each constraint
        self._pattern_states: Dict[str, Dict] = {}
        for c in constraints:
            zone_name = c.parameters.get('avoid_zone', 'unknown')
            self._pattern_states[zone_name] = {
                "violated": False,
                "satisfied": False
            }

    def reset(self) -> SELPMonitorState:
        """Reset automaton to initial state"""
        self._current_state = "q0"
        self._status = AutomatonState.ACCEPTING
        self._step_count = 0
        for zone_name in self._pattern_states:
            self._pattern_states[zone_name] = {"violated": False, "satisfied": False}
        return self._get_monitor_state(False)

    def would_violate(self, props: Dict[str, bool],
                      use_noisy_perception: bool = True) -> Tuple[bool, str, ActionMaskInfo]:
        """
        Check if propositions would move automaton to rejecting state.

        Args:
            props: Proposition values to check
            use_noisy_perception: If True, apply noisy perception to zone checks

        Returns:
            Tuple of (would_violate, reason, mask_info)
        """
        noisy_applied = False
        mask_info = ActionMaskInfo(
            goal_x=0.0, goal_y=0.0,
            masked=False, reason="",
            automaton_status_before=self._status
        )

        for c in self.constraints:
            if c.constraint_type == ConstraintType.ALWAYS_AVOID:
                avoid_prop = f"in_{c.parameters['avoid_zone']}"
                zone_name = c.parameters['avoid_zone']

                # Apply noisy perception
                prop_value = props.get(avoid_prop, False)
                if use_noisy_perception and self.perception_noise_sigma > 0:
                    # Noisy perception: near boundary, there's uncertainty
                    # Simulated by checking if we're "close" to the zone
                    near_boundary_prop = f"near_{zone_name}"
                    if props.get(near_boundary_prop, False):
                        # Near boundary: perception becomes uncertain
                        # Flip with probability based on distance/sigma
                        if random.random() < 0.3:  # 30% chance of perception error
                            prop_value = not prop_value
                            noisy_applied = True

                if prop_value:
                    mask_info.masked = True
                    mask_info.reason = f"Automaton rejects: {avoid_prop} would become True"
                    mask_info.proposition_changes[avoid_prop] = (False, True)
                    mask_info.automaton_status_after = AutomatonState.VIOLATED
                    return True, mask_info.reason, mask_info

        mask_info.automaton_status_after = self._status
        return False, "", mask_info

    def step(self, props: Dict[str, bool]) -> SELPMonitorState:
        """
        Advance automaton by one step given current propositions.

        Args:
            props: Dictionary mapping proposition names to truth values

        Returns:
            Updated monitor state
        """
        self._step_count += 1

        for c in self.constraints:
            if c.constraint_type == ConstraintType.ALWAYS_AVOID:
                avoid_prop = f"in_{c.parameters['avoid_zone']}"
                zone_name = c.parameters['avoid_zone']

                if props.get(avoid_prop, False):
                    self._pattern_states[zone_name]["violated"] = True
                    self._status = AutomatonState.VIOLATED
                    self._current_state = "q_reject"

        if self._status != AutomatonState.VIOLATED:
            self._status = AutomatonState.ACCEPTING

        return self._get_monitor_state(False)

    def can_satisfy(self) -> bool:
        """Check if the constraint can still be satisfied from current state"""
        return self._status != AutomatonState.VIOLATED

    def _get_monitor_state(self, noisy_applied: bool) -> SELPMonitorState:
        """Get current monitor state"""
        return SELPMonitorState(
            automaton_state=self._current_state,
            is_accepting=(self._status == AutomatonState.ACCEPTING),
            can_reach_accepting=(self._status != AutomatonState.VIOLATED),
            status=self._status,
            noisy_perception_applied=noisy_applied
        )


class GoalGateNode(Node):
    """
    Action proxy that enforces safety policy on navigation goals.

    Intercepts NavigateToPose and FollowWaypoints actions, evaluates them
    against the selected safety method, and either forwards, rejects, or projects them.

    Supported safety methods:
    - no_guard: Pass all goals without checking (baseline)
    - geofence: Geometric zone checking with uncertainty-aware margins
    - safetychip: LTL-based constraint monitoring using G(!in_zone)
    - selp: Automaton-based constraint checking
    """

    def __init__(self):
        super().__init__('goal_gate')

        # Declare parameters
        self.declare_parameter('geofence_config', '')
        self.declare_parameter('enable_projection', True)
        self.declare_parameter('audit_log_file', '/tmp/geofence_audit.jsonl')
        self.declare_parameter('nav2_action_name', 'navigate_to_pose')
        self.declare_parameter('waypoints_action_name', 'follow_waypoints')
        self.declare_parameter('safety_method', 'geofence')  # no_guard, geofence, safetychip, selp

        # Ablation study parameters
        self.declare_parameter('enable_estimation_term', True)
        self.declare_parameter('enable_tracking_term', True)
        self.declare_parameter('enable_latency_term', True)
        self.declare_parameter('ablation_condition', 'full')

        # Runtime monitoring parameters (for velocity-dependent safety during navigation)
        self.declare_parameter('enable_runtime_monitoring', False)
        self.declare_parameter('runtime_monitoring_rate', 10.0)  # Hz

        # Dynamic v_max parameters
        self.declare_parameter('use_dynamic_v_max', True)
        self.declare_parameter('v_max_window_size', 100)

        # Dynamic latency (τ) parameters
        self.declare_parameter('use_dynamic_tau', True)
        self.declare_parameter('tau_window_size', 50)

        # Dynamic tracking error (e_track) parameters
        self.declare_parameter('use_dynamic_e_track', True)
        self.declare_parameter('e_track_window_size', 200)

        # Uncertainty/margin parameters (can be overridden for method-specific configs)
        self.declare_parameter('k_sigma', 3.0)
        self.declare_parameter('localization_sigma', 0.15)
        self.declare_parameter('tracking_error', 0.05)
        self.declare_parameter('v_max', 0.5)
        self.declare_parameter('latency', 0.1)

        # Load geofence policy
        config_path = self.get_parameter('geofence_config').get_parameter_value().string_value
        self.policy = GeofencePolicy(config_path if config_path else None)

        # Apply uncertainty/margin parameters to policy
        self.policy.uncertainty.k_sigma = self.get_parameter('k_sigma').value
        self.policy.uncertainty.localization_sigma = self.get_parameter('localization_sigma').value
        self.policy.uncertainty.tracking_error = self.get_parameter('tracking_error').value
        self.policy.uncertainty.v_max = self.get_parameter('v_max').value
        self.policy.uncertainty.latency = self.get_parameter('latency').value

        # Apply ablation settings to policy
        self.policy.uncertainty.enable_estimation_term = self.get_parameter('enable_estimation_term').value
        self.policy.uncertainty.enable_tracking_term = self.get_parameter('enable_tracking_term').value
        self.policy.uncertainty.enable_latency_term = self.get_parameter('enable_latency_term').value
        self.policy.uncertainty.use_dynamic_v_max = self.get_parameter('use_dynamic_v_max').value
        self.policy.uncertainty.use_dynamic_tau = self.get_parameter('use_dynamic_tau').value
        self.policy.uncertainty.use_dynamic_e_track = self.get_parameter('use_dynamic_e_track').value
        self._ablation_condition = self.get_parameter('ablation_condition').value

        # Runtime monitoring settings
        self.enable_runtime_monitoring = self.get_parameter('enable_runtime_monitoring').value
        self.runtime_monitoring_rate = self.get_parameter('runtime_monitoring_rate').value
        self._runtime_violations = []  # Track violations during navigation
        # Note: Runtime monitoring log moved after safety_method initialization (line ~642)

        # Initialize velocity monitor for dynamic v_max
        self._velocity_monitor = VelocityMonitor(
            window_size=self.get_parameter('v_max_window_size').value
        )

        # Initialize latency monitor for dynamic τ
        self._latency_monitor = LatencyMonitor(
            window_size=self.get_parameter('tau_window_size').value
        )

        # Initialize tracking error monitor for dynamic e_track
        self._tracking_error_monitor = TrackingErrorMonitor(
            window_size=self.get_parameter('e_track_window_size').value
        )

        # Current robot position for tracking error measurement
        self._current_x: float = 0.0
        self._current_y: float = 0.0

        self.enable_projection = self.get_parameter('enable_projection').get_parameter_value().bool_value
        self.audit_log_file = self.get_parameter('audit_log_file').get_parameter_value().string_value
        self.safety_method = self.get_parameter('safety_method').get_parameter_value().string_value

        # Log runtime monitoring settings (after safety_method is set)
        if self.enable_runtime_monitoring:
            self.get_logger().info(
                f'[RUNTIME MONITORING] Enabled for method {self.safety_method} at {self.runtime_monitoring_rate}Hz'
            )

        # Initialize SafetyChip and SELP from geofence zones
        self.safetychip_monitor = None
        self.selp_automaton = None
        self._init_ltl_monitors()

        # Initialize new safety baselines (CBF, Static Margin, SSM, SELP-proper)
        self._init_safety_baselines()

        # Callback group for concurrent action handling
        self.cb_group = ReentrantCallbackGroup()

        # Action clients to forward to Nav2
        nav2_action = self.get_parameter('nav2_action_name').get_parameter_value().string_value
        waypoints_action = self.get_parameter('waypoints_action_name').get_parameter_value().string_value

        self._nav_client = ActionClient(
            self, NavigateToPose, nav2_action,
            callback_group=self.cb_group
        )
        self._waypoints_client = ActionClient(
            self, FollowWaypoints, waypoints_action,
            callback_group=self.cb_group
        )

        # Service client for path planning (disabled - ComputePathToPose is an action, not service)
        # Path-based zone checking feature is disabled for now
        # self._compute_path_client = self.create_client(
        #     ComputePathToPose, '/compute_path_to_pose',
        #     callback_group=self.cb_group
        # )
        self._compute_path_client = None

        # Action servers (proxy entry points)
        self._nav_server = ActionServer(
            self, NavigateToPose, 'navigate_to_pose_safe',
            execute_callback=self.execute_nav_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group
        )
        self._waypoints_server = ActionServer(
            self, FollowWaypoints, 'follow_waypoints_safe',
            execute_callback=self.execute_waypoints_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group
        )

        # Publishers for metrics and visualization
        self.metrics_pub = self.create_publisher(String, '/geofence/goal_events', 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/geofence/zones_viz', 10)
        self.status_pub = self.create_publisher(String, '/geofence/status', 10)
        self.margin_pub = self.create_publisher(String, '/geofence/margin_breakdown', 10)

        # Navigation state publisher for cmd_vel_guard integration
        # Enables unapproved motion detection: guard blocks cmd_vel not authorized by goal_gate
        self._nav_state_pub = self.create_publisher(String, '/geofence/nav_state', 10)

        # Subscriber for velocity monitoring (dynamic v_max) and latency measurement
        self._odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, 10
        )

        # Subscriber for cmd_vel (latency measurement)
        self._cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_callback, 10
        )

        # Subscriber for planned path (tracking error measurement)
        self._plan_sub = self.create_subscription(
            Path, '/plan', self._plan_callback, 10
        )

        # Subscriber for approach velocity override (S4 attack scenario)
        # When set, SSM will use this velocity instead of current robot velocity
        self._approach_velocity_override = None
        self._approach_velocity_sub = self.create_subscription(
            Float64, '/approach_velocity', self._approach_velocity_callback, 10
        )

        # Service for reloading geofence config
        self.reload_srv = self.create_service(
            Trigger, '/geofence/reload', self.reload_callback
        )

        # Service for changing safety method at runtime
        self.method_srv = self.create_service(
            Trigger, '/geofence/set_method', self.set_method_callback
        )

        # Visualization timer
        self.create_timer(2.0, self.publish_visualization)

        # Statistics
        self.stats = {
            'total_goals': 0,
            'allowed': 0,
            'rejected': 0,
            'projected': 0,
        }

        self.get_logger().info('=' * 60)
        self.get_logger().info('Goal Gate node initialized')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Safety method: {self.safety_method}')
        self.get_logger().info(f'Listening on: navigate_to_pose_safe, follow_waypoints_safe')
        self.get_logger().info(f'Forwarding to: {nav2_action}, {waypoints_action}')
        if config_path:
            self.get_logger().info(f'Geofence config: {config_path}')
            self.get_logger().info(f'Loaded {len(self.policy.zones)} zones')
            self.get_logger().info(f'Safety margin: {self.policy.get_safety_margin():.3f}m')
            if self.safetychip_monitor:
                self.get_logger().info(f'SafetyChip constraints: {len(self.safetychip_monitor.constraints)}')
            if self.selp_automaton:
                self.get_logger().info(f'SELP automaton constraints: {len(self.selp_automaton.constraints)}')
        self.get_logger().info('-' * 60)
        self.get_logger().info('Ablation Study Configuration:')
        self.get_logger().info(f'  Condition:        {self._ablation_condition}')
        self.get_logger().info(f'  Estimation term:  {"ENABLED" if self.policy.uncertainty.enable_estimation_term else "DISABLED"}')
        self.get_logger().info(f'  Tracking term:    {"ENABLED" if self.policy.uncertainty.enable_tracking_term else "DISABLED"}')
        self.get_logger().info(f'  Latency term:     {"ENABLED" if self.policy.uncertainty.enable_latency_term else "DISABLED"}')
        self.get_logger().info('-' * 60)
        self.get_logger().info('Dynamic Parameter Measurement:')
        self.get_logger().info(f'  Dynamic v_max:    {self.policy.uncertainty.use_dynamic_v_max}')
        self.get_logger().info(f'  Dynamic tau:      {self.policy.uncertainty.use_dynamic_tau}')
        self.get_logger().info(f'  Dynamic e_track:  {self.policy.uncertainty.use_dynamic_e_track}')
        self.get_logger().info('=' * 60)

    def _init_ltl_monitors(self):
        """Initialize SafetyChip and SELP from geofence zones."""
        constraints = []

        for zone in self.policy.zones:
            if zone.zone_type == ZoneType.FORBIDDEN:
                # Create G(!in_zone) constraint for each forbidden zone
                zone_name = zone.name.replace(' ', '_').lower()
                constraint = LTLConstraint(
                    original_nl=f"Never enter {zone.name}",
                    ltl_formula=f"G(!in_{zone_name})",
                    constraint_type=ConstraintType.ALWAYS_AVOID,
                    propositions={f"in_{zone_name}"},
                    parameters={"avoid_zone": zone_name, "vertices": zone.vertices}
                )
                constraints.append(constraint)

        if constraints:
            # SafetyChip Proper: LTL Monitor with Reprompting
            self.safetychip_monitor = SafetyChipMonitor(constraints)

            # SELP Proper: LTL Automaton (original paper - no noisy perception)
            # Noisy perception disabled for fair comparison with original SELP paper
            self.selp_automaton = SELPAutomaton(
                constraints,
                perception_noise_sigma=0.0  # Disabled - matches original SELP
            )
            self.get_logger().info(f'Created {len(constraints)} LTL constraints from forbidden zones')
            self.get_logger().info('SELP: Noisy perception disabled (original paper)')

    def _init_safety_baselines(self):
        """
        Initialize new safety baselines: CBF, Static Margin, SSM, SELP-proper.

        These provide fair academic comparisons with properly cited methods:
        - CBF: Ames et al., TAC 2017
        - Static Margin: Lu et al., IROS 2014 (costmap inflation)
        - SSM: ISO/TS 15066 (Speed and Separation Monitoring)
        - SELP-proper: Wu et al., ICRA 2025 (adapted for spatial safety)
        """
        zone_names = [z.name.replace(' ', '_').lower()
                      for z in self.policy.zones if z.zone_type == ZoneType.FORBIDDEN]

        # SELP-proper: LTL automaton adapted for spatial safety
        self.selp_proper = SELPSpatialSafety(zone_names)

        # CBF: Control Barrier Functions with configurable margin
        # Default margin = 0.3m (can be tuned for comparison)
        self.cbf_checker = CBFSpatialSafety(gamma=1.0, margin=0.3)

        # Static Margin: Fixed costmap-style inflation
        # Default margin = 0.5m (typical ROS Nav2 inflation_radius)
        self.static_margin_checker = StaticMarginSafety(margin=0.5)

        # SSM: Speed and Separation Monitoring per ISO 15066
        self.ssm_checker = SSMSpatialSafety(
            reaction_time=0.1,      # System reaction time (s)
            stopping_time=0.2,      # Time to initiate stop (s)
            max_deceleration=1.0,   # Max deceleration (m/s²)
            intrusion_distance=0.1, # Uncertainty margin (m)
            base_margin=0.2         # Minimum margin when stationary (m)
        )

        self.get_logger().info('Initialized safety baselines: CBF, Static Margin, SSM, SELP-proper')
        self.get_logger().info(f'  - CBF margin: {self.cbf_checker.margin}m')
        self.get_logger().info(f'  - Static margin: {self.static_margin_checker.margin}m')
        self.get_logger().info(f'  - SSM base margin: {self.ssm_checker.base_margin}m')

    def _get_propositions_for_point(self, x: float, y: float,
                                     include_near_boundary: bool = False) -> Dict[str, bool]:
        """
        Convert a point to propositions dict for LTL checking.

        Args:
            x, y: Point coordinates
            include_near_boundary: If True, include "near_zone" props for SELP noisy perception

        Returns:
            Dictionary of proposition names to boolean values
        """
        props = {}

        for zone in self.policy.zones:
            if zone.zone_type == ZoneType.FORBIDDEN:
                zone_name = zone.name.replace(' ', '_').lower()
                prop_name = f"in_{zone_name}"

                # Check if point is inside zone (using geofence's point-in-polygon)
                in_zone = self.policy._point_in_polygon((x, y), zone.vertices)
                props[prop_name] = in_zone

                # For SELP noisy perception: check if near zone boundary
                if include_near_boundary:
                    near_prop_name = f"near_{zone_name}"
                    # "Near" = within safety margin of the zone boundary
                    dist_to_zone = self.policy.get_distance_to_zone((x, y), zone)
                    safety_margin = self.policy.get_safety_margin()
                    props[near_prop_name] = (dist_to_zone < safety_margin)

        return props

    def _evaluate_goal_point(self, x: float, y: float) -> PolicyDecision:
        """
        Evaluate a goal point using the selected safety method.

        Returns PolicyDecision with action (ALLOW/REJECT/PROJECT) and details.
        Enhanced with Proper version features:
        - SafetyChip: Includes reprompt_text for LLM feedback
        - SELP: Includes noisy perception simulation
        """
        point = (x, y)

        # no_guard: Allow everything
        if self.safety_method == 'no_guard':
            return PolicyDecision(
                action=PolicyAction.ALLOW,
                reason="no_guard: All goals allowed",
                original_point=point,
                min_distance_to_forbidden=self.policy.get_min_distance_to_forbidden(x, y)
            )

        # geofence (and geofence_no_margin): Use geometric checking with safety margin
        # geofence_no_margin uses the same code but with k_sigma=0, localization_sigma=0, etc.
        if self.safety_method in ('geofence', 'geofence_no_margin'):
            return self.policy.evaluate_point(x, y)

        # safetychip: Use LTL monitor with Reprompting
        if self.safety_method == 'safetychip':
            if self.safetychip_monitor:
                # Get propositions for current and proposed state
                current_props = {}  # Robot's current position (assume outside all zones)
                proposed_props = self._get_propositions_for_point(x, y)

                would_violate, violations = self.safetychip_monitor.would_violate(
                    current_props, proposed_props
                )

                if would_violate:
                    # Generate detailed explanation with reprompt text
                    explanation = self.safetychip_monitor.explain_violation(
                        violations, current_props, proposed_props
                    )

                    violated_zone = violations[0].constraint.parameters.get('avoid_zone', 'unknown')

                    # Log the detailed explanation (useful for paper)
                    self.get_logger().warn(f"SafetyChip violation explanation:\n{explanation.reprompt_text}")

                    return PolicyDecision(
                        action=PolicyAction.REJECT,
                        reason=f"SafetyChip violation: {explanation.summary}",
                        original_point=point,
                        min_distance_to_forbidden=0.0,
                        violated_zone=violated_zone,
                        # Store reprompt text for potential LLM feedback
                        extra_info={
                            "reprompt_text": explanation.reprompt_text,
                            "suggestions": explanation.suggestions,
                            "reprompt_count": self.safetychip_monitor.reprompt_count
                        }
                    )

            return PolicyDecision(
                action=PolicyAction.ALLOW,
                reason="SafetyChip: No constraint violation",
                original_point=point,
                min_distance_to_forbidden=self.policy.get_min_distance_to_forbidden(x, y)
            )

        # selp: Use automaton checking with Noisy Perception
        if self.safety_method == 'selp':
            if self.selp_automaton:
                # Include near_boundary props for noisy perception
                props = self._get_propositions_for_point(x, y, include_near_boundary=True)

                # Use noisy perception (simulates LLM's imperfect zone understanding)
                would_violate, reason, mask_info = self.selp_automaton.would_violate(
                    props, use_noisy_perception=True
                )

                if would_violate:
                    # Extract violated zone from mask_info
                    violated_zone = "unknown"
                    for prop in mask_info.proposition_changes.keys():
                        if prop.startswith("in_"):
                            violated_zone = prop[3:]  # Remove "in_" prefix
                            break

                    return PolicyDecision(
                        action=PolicyAction.REJECT,
                        reason=f"SELP rejection: {reason}",
                        original_point=point,
                        min_distance_to_forbidden=0.0,
                        violated_zone=violated_zone,
                        extra_info={
                            "automaton_state": mask_info.automaton_status_before.name,
                            "proposition_changes": {k: str(v) for k, v in mask_info.proposition_changes.items()},
                            "noisy_perception": self.selp_automaton.perception_noise_sigma > 0  # Only True if actually enabled
                        }
                    )

            return PolicyDecision(
                action=PolicyAction.ALLOW,
                reason="SELP: Automaton accepts",
                original_point=point,
                min_distance_to_forbidden=self.policy.get_min_distance_to_forbidden(x, y)
            )

        # ============================================================
        # NEW BASELINES: Properly implemented academic methods
        # ============================================================

        # selp_proper: SELP adapted for spatial safety (Wu et al., ICRA 2025)
        if self.safety_method == 'selp_proper':
            forbidden_zones = [z for z in self.policy.zones if z.zone_type == ZoneType.FORBIDDEN]
            result = self.selp_proper.evaluate(point, forbidden_zones)

            if result.decision == SafetyDecision.REJECT:
                return PolicyDecision(
                    action=PolicyAction.REJECT,
                    reason=result.reason,
                    original_point=point,
                    min_distance_to_forbidden=result.distance_to_boundary,
                    violated_zone=result.extra_info.get("violated_zones", ["unknown"])[0] if result.extra_info.get("violated_zones") else "unknown",
                    extra_info={
                        "automaton_state": result.automaton_state,
                        "method": "selp_proper",
                        "reference": "Wu et al., ICRA 2025"
                    }
                )
            return PolicyDecision(
                action=PolicyAction.ALLOW,
                reason=result.reason,
                original_point=point,
                min_distance_to_forbidden=result.distance_to_boundary,
                extra_info={"automaton_state": result.automaton_state, "method": "selp_proper"}
            )

        # cbf: Control Barrier Functions (Ames et al., TAC 2017)
        # CBF evaluates safety using barrier function h(x) = distance - margin.
        # If h(x) < 0, the goal position violates the safety constraint.
        if self.safety_method == 'cbf':
            forbidden_zones = [z for z in self.policy.zones if z.zone_type == ZoneType.FORBIDDEN]
            result = self.cbf_checker.evaluate(point, forbidden_zones)

            # CBF rejects goals where barrier function h(x) < 0 (inside safety margin)
            if result.barrier_value < 0:
                return PolicyDecision(
                    action=PolicyAction.REJECT,
                    reason=f"CBF barrier h(x)={result.barrier_value:.3f} < 0 (goal inside safety margin {result.margin_used}m)",
                    original_point=point,
                    min_distance_to_forbidden=result.distance_to_boundary,
                    violated_zone=result.extra_info.get("closest_zone", "unknown") if result.extra_info else "forbidden_zone",
                    extra_info={
                        "barrier_value": result.barrier_value,
                        "margin": result.margin_used,
                        "method": "cbf",
                        "reference": "Ames et al., TAC 2017"
                    }
                )
            return PolicyDecision(
                action=PolicyAction.ALLOW,
                reason=f"CBF allows goal. Barrier h(x)={result.barrier_value:.3f} >= 0, distance: {result.distance_to_boundary:.2f}m",
                original_point=point,
                min_distance_to_forbidden=result.distance_to_boundary,
                extra_info={
                    "barrier_value": result.barrier_value,
                    "margin": result.margin_used,
                    "method": "cbf",
                    "reference": "Ames et al., TAC 2017"
                }
            )

        # static_margin: Fixed costmap inflation (Lu et al., IROS 2014)
        if self.safety_method == 'static_margin':
            forbidden_zones = [z for z in self.policy.zones if z.zone_type == ZoneType.FORBIDDEN]
            result = self.static_margin_checker.evaluate(point, forbidden_zones)

            if result.decision == SafetyDecision.REJECT:
                return PolicyDecision(
                    action=PolicyAction.REJECT,
                    reason=result.reason,
                    original_point=point,
                    min_distance_to_forbidden=result.distance_to_boundary,
                    violated_zone=result.extra_info.get("closest_zone", "unknown"),
                    extra_info={
                        "margin": result.margin_used,
                        "method": "static_margin",
                        "reference": "Lu et al., IROS 2014"
                    }
                )
            return PolicyDecision(
                action=PolicyAction.ALLOW,
                reason=result.reason,
                original_point=point,
                min_distance_to_forbidden=result.distance_to_boundary,
                extra_info={"margin": result.margin_used, "method": "static_margin"}
            )

        # ssm: Speed and Separation Monitoring (ISO 15066)
        # SSM uses velocity-dependent safety margins. For goal checking, use max velocity.
        if self.safety_method == 'ssm':
            forbidden_zones = [z for z in self.policy.zones if z.zone_type == ZoneType.FORBIDDEN]
            # Use max velocity for goal-level checking (conservative approach)
            # At runtime, actual velocity could be used for more precise margins
            max_velocity = 0.5  # Use v_max for goal-level checking
            if hasattr(self, '_approach_velocity_override') and self._approach_velocity_override is not None:
                max_velocity = self._approach_velocity_override
                self._approach_velocity_override = None
            result = self.ssm_checker.evaluate(point, forbidden_zones, velocity=max_velocity)

            # SSM rejects goals where distance < required margin for max velocity
            if result.distance_to_boundary < result.margin_used:
                return PolicyDecision(
                    action=PolicyAction.REJECT,
                    reason=f"SSM: distance {result.distance_to_boundary:.3f}m < margin {result.margin_used:.3f}m (at v_max={max_velocity}m/s)",
                    original_point=point,
                    min_distance_to_forbidden=result.distance_to_boundary,
                    violated_zone=result.extra_info.get("closest_zone", "forbidden_zone") if result.extra_info else "forbidden_zone",
                    extra_info={
                        "margin": result.margin_used,
                        "velocity": max_velocity,
                        "method": "ssm",
                        "reference": "ISO/TS 15066:2016"
                    }
                )
            return PolicyDecision(
                action=PolicyAction.ALLOW,
                reason=f"SSM allows goal. Distance {result.distance_to_boundary:.2f}m >= margin {result.margin_used:.2f}m",
                original_point=point,
                min_distance_to_forbidden=result.distance_to_boundary,
                extra_info={
                    "margin": result.margin_used,
                    "velocity": max_velocity,
                    "method": "ssm",
                    "reference": "ISO/TS 15066:2016"
                }
            )

        # Fallback to geofence
        self.get_logger().warn(f'Unknown safety method: {self.safety_method}, using geofence')
        return self.policy.evaluate_point(x, y)

    def set_method_callback(self, request, response):
        """Handle safety method change (triggered via parameter change)."""
        # Re-read the safety_method parameter
        try:
            self.safety_method = self.get_parameter('safety_method').get_parameter_value().string_value
            self.get_logger().info(f'Safety method changed to: {self.safety_method}')
            response.success = True
            response.message = f'Safety method set to: {self.safety_method}'
        except Exception as e:
            response.success = False
            response.message = f'Failed to change method: {e}'
        return response

    def _check_line_through_zones(self, goal_x: float, goal_y: float,
                                    num_samples: int = 50) -> Optional[Tuple[str, Tuple[float, float]]]:
        """
        Check if straight line from current position to goal passes through any forbidden zone.

        This is a simplified path check that samples points along the line.
        Used for S5 odom spoofing attack detection - if robot's perceived position
        is spoofed, this check may incorrectly allow dangerous paths.

        Args:
            goal_x, goal_y: Goal coordinates
            num_samples: Number of points to sample along the path

        Returns:
            Tuple of (zone_name, (x, y)) if path passes through zone, None if safe
        """
        start_x = self._current_x
        start_y = self._current_y

        # Include safety margin in the check
        margin = self.policy.uncertainty.compute_margin() if hasattr(self.policy, 'uncertainty') else 0.0

        for i in range(num_samples + 1):
            t = i / num_samples
            px = start_x + t * (goal_x - start_x)
            py = start_y + t * (goal_y - start_y)

            # Check against all forbidden zones
            for zone in self.policy.zones:
                if zone.zone_type != ZoneType.FORBIDDEN:
                    continue

                # Check if point is inside zone (with margin)
                # Expand zone by margin for conservative check
                expanded_vertices = []
                cx = sum(v[0] for v in zone.vertices) / len(zone.vertices)
                cy = sum(v[1] for v in zone.vertices) / len(zone.vertices)
                for vx, vy in zone.vertices:
                    dx, dy = vx - cx, vy - cy
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist > 0:
                        expanded_vertices.append((vx + margin * dx / dist, vy + margin * dy / dist))
                    else:
                        expanded_vertices.append((vx, vy))

                if self.policy._point_in_polygon((px, py), expanded_vertices):
                    self.get_logger().warn(
                        f'[PATH CHECK] Line from ({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f}) '
                        f'passes through {zone.name} at ({px:.2f}, {py:.2f})'
                    )
                    return (zone.name, (px, py))

        return None  # Path is safe

    async def _check_path_through_zones(self, goal_pose: PoseStamped) -> Optional[Tuple[str, Tuple[float, float]]]:
        """
        Check if the planned path to goal passes through any forbidden zone.

        NOTE: This feature is disabled. ComputePathToPose is an action, not a service,
        and proper async handling requires more work.

        Returns:
            None (always) - feature disabled
        """
        # Feature disabled - client is None
        if self._compute_path_client is None:
            return None

        # Check if service is available
        if not self._compute_path_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('ComputePathToPose service not available, skipping path check')
            return None

        # NOTE: This code is unreachable since client is None
        # Keeping it for reference - would need to use ActionClient instead of service client
        request = None  # ComputePathToPose.Request()
        request.goal = goal_pose
        request.planner_id = ''  # Use default planner
        request.use_start = False  # Use robot's current position

        try:
            # Call service synchronously with spin_until_future_complete
            future = self._compute_path_client.call_async(request)

            # Wait for result with timeout using polling
            start_time = self.get_clock().now()
            timeout_sec = 5.0
            while not future.done():
                elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
                if elapsed > timeout_sec:
                    self.get_logger().warn('ComputePathToPose service timed out')
                    return None
                await asyncio.sleep(0.1)

            result = future.result()

            if not result or not result.path.poses:
                self.get_logger().warn('No path returned from planner')
                return None

            # Check each point in the path against forbidden zones
            for pose in result.path.poses:
                px = pose.pose.position.x
                py = pose.pose.position.y

                # Check against all forbidden zones
                for zone in self.policy.zones:
                    if zone.zone_type != ZoneType.FORBIDDEN:
                        continue

                    # Check if point is inside zone polygon
                    if self.policy._point_in_polygon((px, py), zone.vertices):
                        return (zone.name, (px, py))

            return None  # Path is safe

        except Exception as e:
            self.get_logger().warn(f'Path check failed: {e}')
            return None

    def goal_callback(self, goal_request) -> GoalResponse:
        """Accept all goals initially; policy check happens in execute."""
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        """Allow cancellation."""
        return CancelResponse.ACCEPT

    async def execute_nav_callback(self, goal_handle: ServerGoalHandle):
        """Execute NavigateToPose with policy enforcement."""
        self.get_logger().info('Received NavigateToPose goal')

        request = goal_handle.request
        pose = request.pose

        # Evaluate against policy using selected safety method
        x = pose.pose.position.x
        y = pose.pose.position.y
        decision = self._evaluate_goal_point(x, y)

        self.stats['total_goals'] += 1

        # Log the decision
        self.audit_log(decision, 'NavigateToPose', pose)
        self.publish_metrics(decision)

        if decision.action == PolicyAction.REJECT:
            self.stats['rejected'] += 1
            self.get_logger().warn(f'REJECTED goal ({x:.2f}, {y:.2f}): {decision.reason}')
            goal_handle.abort()
            result = NavigateToPose.Result()
            return result

        elif decision.action == PolicyAction.PROJECT and self.enable_projection:
            self.stats['projected'] += 1
            if decision.projected_point:
                new_x, new_y = decision.projected_point
                self.get_logger().warn(
                    f'PROJECTED goal ({x:.2f}, {y:.2f}) -> ({new_x:.2f}, {new_y:.2f}): {decision.reason}'
                )
                pose.pose.position.x = new_x
                pose.pose.position.y = new_y
            else:
                self.get_logger().warn(f'REJECTED (projection failed): {decision.reason}')
                goal_handle.abort()
                result = NavigateToPose.Result()
                return result

        else:
            self.stats['allowed'] += 1
            self.get_logger().info(f'ALLOWED goal ({x:.2f}, {y:.2f})')

        # Path-based zone checking using line sampling
        # Check if straight line from current position to goal passes through forbidden zone
        # NOTE: This uses self._current_x/y from odom, so odom spoofing attacks can bypass this!
        #
        # Methods that need path checking:
        # - geofence/geofence_hw: Our method with full path checking and uncertainty margins
        #
        # Methods that ONLY check goal point (NO path checking per their papers):
        # - selp/selp_proper: Goal-only check (Wu et al., ICRA 2025)
        # - cbf: Control Barrier Functions - goal point only (Ames et al., TAC 2017)
        # - ssm: Speed and Separation Monitoring - goal point only (ISO 15066)
        # NOTE: CBF/SSM are intentionally EXCLUDED - they evaluate goal safety, not path safety
        path_aware_methods = ['geofence', 'geofence_hw']
        if self.safety_method in path_aware_methods:
            # Use the (possibly projected) goal coordinates
            check_x = pose.pose.position.x
            check_y = pose.pose.position.y
            path_check_result = self._check_line_through_zones(check_x, check_y)
            if path_check_result is not None:
                violated_zone, violation_point = path_check_result
                self.get_logger().warn(
                    f'PATH REJECTED: Route from ({self._current_x:.2f}, {self._current_y:.2f}) to '
                    f'({check_x:.2f}, {check_y:.2f}) passes through {violated_zone} '
                    f'at ({violation_point[0]:.2f}, {violation_point[1]:.2f})'
                )
                # Update stats
                self.stats['rejected'] += 1
                self.stats['allowed'] -= 1  # Undo the earlier allowed count

                # Log the path rejection for experiment analysis
                self.audit_log(PolicyDecision(
                    action=PolicyAction.REJECT,
                    reason=f"Path crosses forbidden zone {violated_zone} at ({violation_point[0]:.2f}, {violation_point[1]:.2f})",
                    original_point=(check_x, check_y),
                    min_distance_to_forbidden=0.0,
                    violated_zone=violated_zone,
                    extra_info={"rejection_type": "path_check", "violation_point": violation_point}
                ), 'NavigateToPose', pose)

                goal_handle.abort()
                result = NavigateToPose.Result()
                return result

        # Forward to Nav2
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 action server not available')
            goal_handle.abort()
            return NavigateToPose.Result()

        # Send goal to Nav2
        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = pose
        nav_goal.behavior_tree = request.behavior_tree

        send_goal_future = self._nav_client.send_goal_async(nav_goal)
        nav_goal_handle = await send_goal_future

        if not nav_goal_handle.accepted:
            self.get_logger().error('Nav2 rejected the goal')
            goal_handle.abort()
            return NavigateToPose.Result()

        # Notify cmd_vel_guard that navigation is authorized
        nav_state_msg = String()
        nav_state_msg.data = json.dumps({'state': 'active', 'goal_x': x, 'goal_y': y})
        self._nav_state_pub.publish(nav_state_msg)

        # Wait for result with optional runtime monitoring
        result_future = nav_goal_handle.get_result_async()

        if self.enable_runtime_monitoring and self.safety_method in ['ssm', 'cbf']:
            # Real-time monitoring during navigation
            monitoring_active = True
            monitoring_interval = 1.0 / self.runtime_monitoring_rate
            runtime_violation_logged = False
            monitoring_check_count = 0

            self.get_logger().info(f'[RUNTIME] Starting monitoring loop for {self.safety_method} at {self.runtime_monitoring_rate}Hz')

            while monitoring_active and not result_future.done():
                try:
                    # Get current state
                    current_velocity = math.sqrt(self._current_vx**2 + self._current_vy**2)
                    current_x = self._current_x
                    current_y = self._current_y
                    monitoring_check_count += 1

                    # CBF/SSM runtime safety check - directly evaluate barrier function
                    # instead of using _evaluate_goal_point which always returns ALLOW
                    forbidden_zones = [z for z in self.policy.zones if z.zone_type == ZoneType.FORBIDDEN]
                    point = (current_x, current_y)
                    runtime_violation = False
                    violation_reason = ""

                    if self.safety_method == 'cbf':
                        # CBF: Check barrier function h(x) = distance - margin
                        result = self.cbf_checker.evaluate(point, forbidden_zones)

                        # Debug log every 10 checks
                        if monitoring_check_count % 10 == 0:
                            self.get_logger().info(
                                f'[RUNTIME CHECK #{monitoring_check_count}] pos=({current_x:.2f}, {current_y:.2f}), '
                                f'h(x)={result.barrier_value:.3f}, dist={result.distance_to_boundary:.3f}m'
                            )

                        # If barrier value is negative, robot is in danger zone
                        if result.barrier_value < 0:
                            runtime_violation = True
                            violation_reason = f"CBF barrier h(x)={result.barrier_value:.2f} < 0 (inside safety margin)"
                    elif self.safety_method == 'ssm':
                        # SSM: Check velocity-dependent margin
                        result = self.ssm_checker.evaluate(point, forbidden_zones, velocity=current_velocity)
                        # If distance < required margin, robot is too close
                        if result.distance_to_boundary < result.margin_used:
                            runtime_violation = True
                            violation_reason = f"SSM distance {result.distance_to_boundary:.2f}m < margin {result.margin_used:.2f}m"

                    # Check for violations
                    if runtime_violation:
                        self.get_logger().warn(
                            f'[RUNTIME VIOLATION] Position ({current_x:.2f}, {current_y:.2f}), '
                            f'Velocity: {current_velocity:.2f} m/s - {violation_reason}'
                        )

                        # Log runtime violation
                        if not runtime_violation_logged:
                            self._runtime_violations.append({
                                'timestamp': datetime.now().isoformat(),
                                'position': (current_x, current_y),
                                'velocity': current_velocity,
                                'original_goal': (x, y),
                                'reason': violation_reason,
                                'method': self.safety_method
                            })
                            self.stats['runtime_violations'] = self.stats.get('runtime_violations', 0) + 1
                            runtime_violation_logged = True

                        # Cancel navigation
                        self.get_logger().warn('[RUNTIME] Canceling navigation due to safety violation')
                        await nav_goal_handle.cancel_goal_async()
                        goal_handle.abort()
                        monitoring_active = False
                        return NavigateToPose.Result()

                    # Sleep until next check
                    await asyncio.sleep(monitoring_interval)

                except Exception as e:
                    self.get_logger().error(f'Runtime monitoring error: {e}')
                    monitoring_active = False

            self.get_logger().info(f'[RUNTIME] Monitoring loop ended after {monitoring_check_count} checks')

        nav_result = await result_future

        # Navigation completed — notify cmd_vel_guard
        nav_state_msg = String()
        nav_state_msg.data = json.dumps({'state': 'idle'})
        self._nav_state_pub.publish(nav_state_msg)

        # Forward result
        if nav_result.status == 4:  # SUCCEEDED
            goal_handle.succeed()
        else:
            goal_handle.abort()

        return nav_result.result

    async def execute_waypoints_callback(self, goal_handle: ServerGoalHandle):
        """Execute FollowWaypoints with policy enforcement."""
        self.get_logger().info('Received FollowWaypoints goal')

        request = goal_handle.request
        poses = request.poses

        # Evaluate all waypoints
        filtered_poses = []
        any_rejected = False

        for i, pose in enumerate(poses):
            x = pose.pose.position.x
            y = pose.pose.position.y
            decision = self._evaluate_goal_point(x, y)

            self.stats['total_goals'] += 1
            self.audit_log(decision, f'FollowWaypoints[{i}]', pose)

            if decision.action == PolicyAction.REJECT:
                self.stats['rejected'] += 1
                self.get_logger().warn(f'REJECTED waypoint {i} ({x:.2f}, {y:.2f}): {decision.reason}')
                any_rejected = True
                continue  # Skip this waypoint

            elif decision.action == PolicyAction.PROJECT and self.enable_projection:
                self.stats['projected'] += 1
                if decision.projected_point:
                    new_x, new_y = decision.projected_point
                    self.get_logger().warn(
                        f'PROJECTED waypoint {i} ({x:.2f}, {y:.2f}) -> ({new_x:.2f}, {new_y:.2f})'
                    )
                    pose.pose.position.x = new_x
                    pose.pose.position.y = new_y
                    filtered_poses.append(pose)
                else:
                    self.get_logger().warn(f'REJECTED waypoint {i} (projection failed)')
                    any_rejected = True
            else:
                self.stats['allowed'] += 1
                filtered_poses.append(pose)

        if not filtered_poses:
            self.get_logger().error('All waypoints rejected')
            goal_handle.abort()
            return FollowWaypoints.Result()

        # Forward to Nav2
        if not self._waypoints_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 waypoints action server not available')
            goal_handle.abort()
            return FollowWaypoints.Result()

        wp_goal = FollowWaypoints.Goal()
        wp_goal.poses = filtered_poses

        send_goal_future = self._waypoints_client.send_goal_async(wp_goal)
        wp_goal_handle = await send_goal_future

        if not wp_goal_handle.accepted:
            self.get_logger().error('Nav2 rejected waypoints goal')
            goal_handle.abort()
            return FollowWaypoints.Result()

        result_future = wp_goal_handle.get_result_async()
        wp_result = await result_future

        if wp_result.status == 4:  # SUCCEEDED
            goal_handle.succeed()
        else:
            goal_handle.abort()

        return wp_result.result

    def audit_log(self, decision: PolicyDecision, action_type: str, pose: PoseStamped):
        """Write audit log entry with enhanced Proper version info and ablation data."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'safety_method': self.safety_method,
            'action_type': action_type,
            'decision': decision.action.value,
            'reason': decision.reason,
            'original_x': decision.original_point[0],
            'original_y': decision.original_point[1],
            'projected_x': decision.projected_point[0] if decision.projected_point else None,
            'projected_y': decision.projected_point[1] if decision.projected_point else None,
            'min_distance_to_forbidden': decision.min_distance_to_forbidden,
            'violated_zone': decision.violated_zone,
            # Ablation study data
            'ablation_condition': self._ablation_condition,
            'margin_breakdown': self.policy.uncertainty.get_margin_breakdown(),
            'ablation_state': self.policy.uncertainty.get_ablation_state(),
            'measured_params': self.policy.uncertainty.get_measured_params(),
        }

        # Include Proper version extra info (reprompt_text, automaton_state, etc.)
        if decision.extra_info:
            entry['extra_info'] = decision.extra_info

        try:
            with open(self.audit_log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            self.get_logger().warn(f'Failed to write audit log: {e}')

    def publish_metrics(self, decision: PolicyDecision):
        """Publish metrics event."""
        msg = String()
        msg.data = json.dumps({
            'safety_method': self.safety_method,
            'action': decision.action.value,
            'min_dist': decision.min_distance_to_forbidden,
            'stats': self.stats
        })
        self.metrics_pub.publish(msg)

    def _odom_callback(self, msg: Odometry):
        """Process odometry for dynamic v_max, latency, and tracking error estimation."""
        import math

        # Extract velocity - store for SSM velocity-dependent margin
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._current_vx = vx
        self._current_vy = vy
        velocity = math.sqrt(vx*vx + vy*vy)

        # Extract position
        self._current_x = msg.pose.pose.position.x
        self._current_y = msg.pose.pose.position.y

        # Get timestamp for latency measurement
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Record velocity for dynamic v_max
        self._velocity_monitor.record_velocity(velocity)

        # Record velocity for latency measurement
        self._latency_monitor.record_odom_velocity(timestamp, velocity)

        # Record position for tracking error measurement
        self._tracking_error_monitor.record_position(self._current_x, self._current_y)

        # Update policy with dynamic v_max if enabled
        if self.policy.uncertainty.use_dynamic_v_max:
            v_stats = self._velocity_monitor.get_stats()
            if v_stats.get('sample_count', 0) >= 20:
                self.policy.uncertainty.v_max_observed = v_stats['v_max_95']
                self.policy.uncertainty.velocity_samples = v_stats['sample_count']

        # Update policy with dynamic tau if enabled
        if self.policy.uncertainty.use_dynamic_tau:
            tau_stats = self._latency_monitor.get_stats()
            if tau_stats.get('sample_count', 0) >= 10:
                self.policy.uncertainty.tau_observed = tau_stats['tau_95']
                self.policy.uncertainty.tau_samples = tau_stats['sample_count']

        # Update policy with dynamic e_track if enabled
        if self.policy.uncertainty.use_dynamic_e_track:
            e_track_stats = self._tracking_error_monitor.get_stats()
            if e_track_stats.get('sample_count', 0) >= 20:
                # Cap e_track to reasonable maximum (0.2m) to prevent excessive margins
                e_track_value = min(e_track_stats['e_track_95'], 0.2)
                self.policy.uncertainty.e_track_observed = e_track_value
                self.policy.uncertainty.e_track_samples = e_track_stats['sample_count']

    def _cmd_vel_callback(self, msg: Twist):
        """Record cmd_vel for latency measurement."""
        # Get current time
        timestamp = self.get_clock().now().nanoseconds * 1e-9
        self._latency_monitor.record_cmd_vel(timestamp, msg.linear.x, msg.linear.y)

    def _approach_velocity_callback(self, msg: Float64):
        """Handle approach velocity override for S4 attack scenario."""
        self._approach_velocity_override = msg.data
        self.get_logger().info(f'[S4] Approach velocity override set to {msg.data:.2f} m/s')

    def _plan_callback(self, msg: Path):
        """Update tracking error monitor with new planned path."""
        # Extract path points as (x, y) tuples
        path_points = [
            (pose.pose.position.x, pose.pose.position.y)
            for pose in msg.poses
        ]
        self._tracking_error_monitor.update_path(path_points)

    def publish_visualization(self):
        """Publish geofence visualization markers."""
        if self.policy.zones:
            markers = self.policy.get_forbidden_zones_as_marker_array()
            for m in markers.markers:
                m.header.stamp = self.get_clock().now().to_msg()
            self.viz_pub.publish(markers)

        # Publish status
        status_msg = String()
        status_msg.data = json.dumps({
            'status': 'active',
            'safety_method': self.safety_method,
            'zones_count': len(self.policy.zones),
            'safety_margin': self.policy.get_safety_margin(),
            'stats': self.stats
        })
        self.status_pub.publish(status_msg)

        # Publish margin breakdown for experiment data collection
        margin_msg = String()
        margin_msg.data = json.dumps({
            'margin_breakdown': self.policy.uncertainty.get_margin_breakdown(),
            'ablation': self.policy.uncertainty.get_ablation_state(),
            'ablation_condition': self._ablation_condition,
            'measured_params': self.policy.uncertainty.get_measured_params(),
        })
        self.margin_pub.publish(margin_msg)

    def reload_callback(self, request, response):
        """Handle geofence reload service request."""
        if self.policy.reload():
            # Re-initialize LTL monitors with new zones
            self._init_ltl_monitors()
            self.get_logger().info('Geofence configuration reloaded')
            response.success = True
            response.message = f'Reloaded {len(self.policy.zones)} zones'
        else:
            self.get_logger().error('Failed to reload geofence configuration')
            response.success = False
            response.message = 'Reload failed'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GoalGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Goal Gate node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
