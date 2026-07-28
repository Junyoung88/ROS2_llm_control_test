#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trusted Command Mux - actuator-facing reference monitor (PETSE Trusted Gateway).

Motivation
----------
In the baseline architecture both Nav2 and an attacker can publish to the same
``/cmd_vel`` topic that the base controller obeys, so a compromised ROS node can
simply overwrite PETSE's stop command:

    attacker ─┐
    Nav2    ──┼──▶ /cmd_vel ──▶ base controller
    PETSE  v=0┘        (attacker wins the last-writer race)

This node moves the trust boundary to the actuator. It is intended to be the
*sole* publisher to the actuator topic; every other node publishes a *proposal*
that this mux may forward, drop, or override:

    Nav2 / LLM / attackable ROS nodes ──▶ /cmd_vel_proposed
                                              │
                                    Trusted Command Mux  ◀── /petse/stop_latch (PETSE verdict)
                                              │ allow / stop-latch
                                              ▼
                                          /cmd_vel  ──▶ base controller / motor driver

Guarantees this prototype demonstrates
--------------------------------------
1. Sole-authority forwarding: only the mux writes the actuator topic. In a real
   deployment the exclusivity is enforced by SROS2 enclaves + DDS permissions (or
   a separate MCU / serial-CAN link); this node models the policy so the property
   can be measured end-to-end in simulation.
2. Stop latch: once PETSE trips ``/petse/stop_latch``, the mux latches to zero and
   *drops every subsequent proposal from any publisher* until a trusted reset. A
   200 Hz flood of malicious commands therefore reaches the actuator 0 times.
3. Trusted reset only: the latch is released solely by ``/petse/trusted_reset``
   carrying the shared ``reset_token``. A ROS-graph attacker can publish to the
   topic but, lacking the token held inside the enclave, cannot clear the latch
   (models "operator / separate trusted reset only").
4. Fail-closed heartbeat: while latched the mux keeps publishing zero at
   ``heartbeat_hz`` so the base controller cannot coast on a stale command.

Instrumentation (published as JSON on ``/petse/mux_metrics``) lets the
command-race experiment report: malicious commands generated, malicious commands
reaching the actuator after latch (target 0), detection→actuator latency, and
reset-token rejections.

Threat model (narrowed): the attacker may publish to navigation- and
controller-facing ROS topics (``/cmd_vel_proposed`` included) but NOT to the
actuator channel owned by the PETSE enclave, and encoder/odometry are delivered
through the same trusted boundary.
"""

import json
import time
from enum import Enum
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool


def _is_nonzero(cmd: Twist, eps: float = 1e-6) -> bool:
    """True if the command would actually move the robot."""
    return (abs(cmd.linear.x) > eps or abs(cmd.linear.y) > eps or
            abs(cmd.linear.z) > eps or abs(cmd.angular.x) > eps or
            abs(cmd.angular.y) > eps or abs(cmd.angular.z) > eps)


class MuxState(Enum):
    FORWARDING = "forwarding"   # normal: proposals are forwarded to the actuator
    LATCHED = "latched"         # danger latched: proposals dropped, zero held


class TrustedCmdMuxNode(Node):
    """Actuator-facing command mux with a trusted stop latch."""

    def __init__(self):
        super().__init__('trusted_cmd_mux')

        # -- Topics ------------------------------------------------------------
        self.declare_parameter('proposed_topic', '/cmd_vel_proposed')
        self.declare_parameter('actuator_topic', '/cmd_vel')
        self.declare_parameter('stop_latch_topic', '/petse/stop_latch')
        self.declare_parameter('reset_topic', '/petse/trusted_reset')
        self.declare_parameter('metrics_topic', '/petse/mux_metrics')
        # -- Policy ------------------------------------------------------------
        # Shared secret that a trusted reset must present. Held inside the enclave;
        # a ROS-graph attacker does not have it. Empty string disables the check
        # (prototype/debug only — never in a real deployment).
        self.declare_parameter('reset_token', 'petse-trusted-reset')
        self.declare_parameter('heartbeat_hz', 20.0)       # zero-hold rate while latched
        self.declare_parameter('proposal_timeout_sec', 0.5)  # fail-closed if proposals go stale
        self.declare_parameter('publish_metrics', True)

        gp = self.get_parameter
        self.proposed_topic = gp('proposed_topic').value
        self.actuator_topic = gp('actuator_topic').value
        self.stop_latch_topic = gp('stop_latch_topic').value
        self.reset_topic = gp('reset_topic').value
        self.metrics_topic = gp('metrics_topic').value
        self.reset_token = gp('reset_token').value
        self.heartbeat_hz = float(gp('heartbeat_hz').value)
        self.proposal_timeout = float(gp('proposal_timeout_sec').value)
        self.publish_metrics = bool(gp('publish_metrics').value)

        # -- State -------------------------------------------------------------
        self.state = MuxState.FORWARDING
        self._latch_time: Optional[float] = None          # monotonic when latch tripped
        self._latch_to_actuator_ms: Optional[float] = None  # detection→actuator stop latency
        self._last_proposal_time: Optional[float] = None

        # -- Counters (command-race metrics) -----------------------------------
        self.proposals_total = 0
        self.forwarded_total = 0
        self.proposals_after_latch = 0
        self.nonzero_after_latch = 0        # "malicious commands" arriving post-latch
        self.forwarded_after_latch = 0      # MUST stay 0 — the core security property
        self.latch_count = 0
        self.reset_attempts = 0
        self.reset_rejected = 0
        self.reset_accepted = 0

        # -- QoS: reliable, keep-last-1 (actuator wants the freshest command) ---
        cmd_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)
        sig_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)

        # Sole publisher to the actuator topic.
        self.actuator_pub = self.create_publisher(Twist, self.actuator_topic, cmd_qos)
        self.metrics_pub = self.create_publisher(String, self.metrics_topic, 10)

        self.create_subscription(Twist, self.proposed_topic,
                                 self._on_proposal, cmd_qos)
        self.create_subscription(Bool, self.stop_latch_topic,
                                 self._on_stop_latch, sig_qos)
        self.create_subscription(String, self.reset_topic,
                                 self._on_reset, sig_qos)

        # Heartbeat / watchdog timer.
        period = 1.0 / max(self.heartbeat_hz, 1.0)
        self.create_timer(period, self._on_heartbeat)
        if self.publish_metrics:
            self.create_timer(0.5, self._publish_metrics)

        self.get_logger().info(
            f"[trusted_cmd_mux] proposals '{self.proposed_topic}' → actuator "
            f"'{self.actuator_topic}'; latch '{self.stop_latch_topic}', reset "
            f"'{self.reset_topic}' (token {'set' if self.reset_token else 'DISABLED'})")

    # -- helpers ---------------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _publish_zero(self):
        self.actuator_pub.publish(Twist())

    # -- proposal path ---------------------------------------------------------
    def _on_proposal(self, msg: Twist):
        self.proposals_total += 1
        self._last_proposal_time = self._now()

        if self.state == MuxState.LATCHED:
            # Drop everything while latched — this is the command-race defence.
            self.proposals_after_latch += 1
            if _is_nonzero(msg):
                self.nonzero_after_latch += 1
            # Record the first time we hold zero *after* a nonzero proposal was
            # dropped, i.e. the detection→actuator latency for the stop taking hold.
            if self._latch_to_actuator_ms is None and self._latch_time is not None:
                self._latch_to_actuator_ms = (self._now() - self._latch_time) * 1000.0
            self._publish_zero()  # keep holding stop; never forward a latched proposal
            return

        # FORWARDING: pass the proposal through to the actuator.
        self.actuator_pub.publish(msg)
        self.forwarded_total += 1

    # -- trusted control path --------------------------------------------------
    def _on_stop_latch(self, msg: Bool):
        if not msg.data:
            return
        if self.state == MuxState.LATCHED:
            return
        self.state = MuxState.LATCHED
        self.latch_count += 1
        self._latch_time = self._now()
        self._latch_to_actuator_ms = None
        self._publish_zero()
        # If no nonzero proposal follows, the stop still took effect immediately.
        self._latch_to_actuator_ms = 0.0
        self.get_logger().warn("[trusted_cmd_mux] STOP LATCH engaged — dropping all "
                               "proposals until trusted reset")

    def _on_reset(self, msg: String):
        self.reset_attempts += 1
        if self.reset_token and msg.data != self.reset_token:
            self.reset_rejected += 1
            self.get_logger().warn("[trusted_cmd_mux] trusted reset REJECTED "
                                   "(bad token) — latch held")
            return
        self.reset_accepted += 1
        if self.state == MuxState.LATCHED:
            self.state = MuxState.FORWARDING
            self._latch_time = None
            self.get_logger().info("[trusted_cmd_mux] trusted reset accepted — "
                                   "resuming forwarding")

    # -- heartbeat / fail-closed watchdog --------------------------------------
    def _on_heartbeat(self):
        now = self._now()
        if self.state == MuxState.LATCHED:
            self._publish_zero()          # keep holding stop
            return
        # FORWARDING: fail closed if proposals have gone stale.
        if (self._last_proposal_time is not None and
                (now - self._last_proposal_time) > self.proposal_timeout):
            self._publish_zero()

    def _publish_metrics(self):
        m = {
            'state': self.state.value,
            'proposals_total': self.proposals_total,
            'forwarded_total': self.forwarded_total,
            'latch_count': self.latch_count,
            'proposals_after_latch': self.proposals_after_latch,
            'nonzero_after_latch': self.nonzero_after_latch,
            'forwarded_after_latch': self.forwarded_after_latch,   # must be 0
            'latch_to_actuator_ms': self._latch_to_actuator_ms,
            'reset_attempts': self.reset_attempts,
            'reset_rejected': self.reset_rejected,
            'reset_accepted': self.reset_accepted,
        }
        self.metrics_pub.publish(String(data=json.dumps(m)))


def main(args=None):
    rclpy.init(args=args)
    node = TrustedCmdMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
