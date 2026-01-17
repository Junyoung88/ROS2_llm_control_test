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

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose, FollowWaypoints
from std_srvs.srv import Trigger, SetBool
from visualization_msgs.msg import MarkerArray

from tf_transformations import euler_from_quaternion, quaternion_from_euler

from .geofence_core import GeofencePolicy, PolicyAction, PolicyDecision, ZoneType


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

        # Load geofence policy
        config_path = self.get_parameter('geofence_config').get_parameter_value().string_value
        self.policy = GeofencePolicy(config_path if config_path else None)

        self.enable_projection = self.get_parameter('enable_projection').get_parameter_value().bool_value
        self.audit_log_file = self.get_parameter('audit_log_file').get_parameter_value().string_value
        self.safety_method = self.get_parameter('safety_method').get_parameter_value().string_value

        # Initialize SafetyChip and SELP from geofence zones
        self.safetychip_monitor = None
        self.selp_automaton = None
        self._init_ltl_monitors()

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

        self.get_logger().info('Goal Gate node initialized')
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

            # SELP Proper: LTL Automaton with Noisy Perception
            # perception_noise_sigma matches the localization uncertainty
            perception_sigma = self.policy.uncertainty.localization_sigma
            self.selp_automaton = SELPAutomaton(
                constraints,
                perception_noise_sigma=perception_sigma
            )
            self.get_logger().info(f'Created {len(constraints)} LTL constraints from forbidden zones')
            self.get_logger().info(f'SELP perception noise sigma: {perception_sigma:.3f}m')

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

        # geofence: Use geometric checking with safety margin
        if self.safety_method == 'geofence':
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
                            "noisy_perception": True
                        }
                    )

            return PolicyDecision(
                action=PolicyAction.ALLOW,
                reason="SELP: Automaton accepts",
                original_point=point,
                min_distance_to_forbidden=self.policy.get_min_distance_to_forbidden(x, y)
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

        # Wait for result
        result_future = nav_goal_handle.get_result_async()
        nav_result = await result_future

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
        """Write audit log entry with enhanced Proper version info."""
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
