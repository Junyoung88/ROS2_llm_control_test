#!/usr/bin/env python3
"""
Smooth pick-and-place for box stack demo.

Uses joint-space waypoints + Pilz PTP planner for natural motion.
Avoids the flipping/ceiling-looking issue of Cartesian IK goals.

Usage:
    python3 smooth_pick.py              # pick top box
    python3 smooth_pick.py --test-home  # just go to home position
"""

import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger
from geometry_msgs.msg import PoseStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, MultiPipelinePlanRequestParameters
from moveit.core.kinematic_constraints import construct_joint_constraint
from moveit_msgs.msg import CollisionObject, Constraints, JointConstraint
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


def plan_and_execute(robot, planning_component, logger, sleep_time=0.5):
    """Plan with Pilz PTP (smooth industrial motion), fallback to OMPL."""
    # Try Pilz first (smooth, predictable), then OMPL
    multi_params = MultiPipelinePlanRequestParameters(
        robot, ["pilz_lin", "ompl_rrtc"]
    )

    plan_result = planning_component.plan(
        multi_plan_parameters=multi_params
    )

    if not plan_result:
        logger.error("Planning failed")
        return False

    trajectory = plan_result.trajectory
    if trajectory:
        robot.execute(trajectory, controllers=[])
        time.sleep(sleep_time)
        return True
    return False


class SmoothPickController(Node):
    def __init__(self):
        super().__init__('smooth_pick')
        self.logger = self.get_logger()
        self.logger.info("Initializing SmoothPickController...")

        self.ur = MoveItPy(node_name="smooth_pick_moveit")
        self.arm = self.ur.get_planning_component("ur_manipulator")
        self.hand = self.ur.get_planning_component("robotiq_2f_85_gripper")

        robot_model = self.ur.get_robot_model()
        self.robot_state = RobotState(robot_model)

        self.pose_goal = PoseStamped()
        self.pose_goal.header.frame_id = "base_link"

        time.sleep(3)
        self.logger.info("SmoothPickController ready!")

    def _set_elbow_down_constraint(self):
        """Set path constraints to keep arm in natural pose."""
        constraints = Constraints()
        constraints.name = "elbow_down"

        # Shoulder: keep pointing down (-90° ± 30°)
        sc = JointConstraint()
        sc.joint_name = "shoulder_lift_joint"
        sc.position = -np.pi / 2
        sc.tolerance_above = np.pi / 6
        sc.tolerance_below = np.pi / 6
        sc.weight = 1.0
        constraints.joint_constraints.append(sc)

        # Elbow: keep positive (90° ± 60°)
        ec = JointConstraint()
        ec.joint_name = "elbow_joint"
        ec.position = np.pi / 2
        ec.tolerance_above = np.pi / 3
        ec.tolerance_below = np.pi / 3
        ec.weight = 1.0
        constraints.joint_constraints.append(ec)

        self.arm.set_path_constraints(constraints)

    def go_home(self):
        """Move to the 'ready' named configuration."""
        self.logger.info("Moving to HOME position...")
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(configuration_name="ready")
        return plan_and_execute(self.ur, self.arm, self.logger, sleep_time=1.0)

    def move_joints(self, joint_values: dict, description: str = ""):
        """Move to specific joint angles (most reliable, no IK issues)."""
        self.logger.info(f"Moving joints: {description}")
        self.arm.set_start_state_to_current_state()
        self.robot_state.joint_positions = joint_values
        joint_constraint = construct_joint_constraint(
            robot_state=self.robot_state,
            joint_model_group=self.ur.get_robot_model().get_joint_model_group(
                "ur_manipulator"),
        )
        self.arm.set_goal_state(motion_plan_constraints=[joint_constraint])
        return plan_and_execute(self.ur, self.arm, self.logger, sleep_time=0.5)

    def move_cartesian(self, x, y, z, qx=0., qy=0., qz=0., qw=1.):
        """Move to Cartesian pose with elbow-down constraint."""
        self.logger.info(f"Moving to ({x:.3f}, {y:.3f}, {z:.3f})")
        self.arm.set_start_state_to_current_state()
        self._set_elbow_down_constraint()

        self.pose_goal.pose.position.x = x
        self.pose_goal.pose.position.y = y
        self.pose_goal.pose.position.z = z
        self.pose_goal.pose.orientation.x = qx
        self.pose_goal.pose.orientation.y = qy
        self.pose_goal.pose.orientation.z = qz
        self.pose_goal.pose.orientation.w = qw
        self.arm.set_goal_state(
            pose_stamped_msg=self.pose_goal, pose_link="tool0")
        return plan_and_execute(self.ur, self.arm, self.logger, sleep_time=0.5)

    def gripper(self, action: str):
        """Open or close gripper."""
        self.logger.info(f"Gripper: {action}")
        self.hand.set_start_state_to_current_state()
        val = 0.01 if action == "open" else 0.65
        self.robot_state.joint_positions = {
            "robotiq_85_left_knuckle_joint": val}
        jc = construct_joint_constraint(
            robot_state=self.robot_state,
            joint_model_group=self.ur.get_robot_model().get_joint_model_group(
                "robotiq_2f_85_gripper"),
        )
        self.hand.set_goal_state(motion_plan_constraints=[jc])
        return plan_and_execute(self.ur, self.hand, self.logger, sleep_time=0.3)

    def pick_top_box(self):
        """
        Pick the top box from the stack using predefined joint waypoints.

        Box top is at base_link (0.5, 0.0, 0.14).
        We use joint-space goals for critical waypoints to avoid IK issues.
        """
        self.logger.info("=== PICKING TOP BOX ===")
        ok = True

        # 1. Home position (known good joint config)
        ok = ok and self.go_home()
        if not ok:
            self.logger.error("Failed to reach home")
            return False

        # 2. Open gripper
        ok = ok and self.gripper("open")

        # 3. Pre-approach: above the box stack
        #    Joint values for tool0 at roughly (0.5, 0.0, 0.35) pointing down
        ok = ok and self.move_joints({
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.2,
            "elbow_joint": 1.0,
            "wrist_1_joint": -1.37,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
        }, "pre-approach above box")

        # 4. Lower to grasp position
        #    Cartesian move straight down (with elbow constraint)
        ok = ok and self.move_cartesian(0.5, 0.0, 0.20, qw=1.0)

        # 5. Close gripper
        ok = ok and self.gripper("close")
        time.sleep(0.5)

        # 6. Lift up
        ok = ok and self.move_cartesian(0.5, 0.0, 0.35, qw=1.0)

        # 7. Move to place position (offset to the side)
        ok = ok and self.move_cartesian(0.5, -0.3, 0.35, qw=1.0)

        # 8. Lower to place
        ok = ok and self.move_cartesian(0.5, -0.3, 0.20, qw=1.0)

        # 9. Release
        ok = ok and self.gripper("open")

        # 10. Retreat up
        ok = ok and self.move_cartesian(0.5, -0.3, 0.35, qw=1.0)

        # 11. Back to home
        ok = ok and self.go_home()

        self.logger.info(f"=== PICK COMPLETE: {'SUCCESS' if ok else 'FAILED'} ===")
        return ok


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-home', action='store_true',
                        help='Just test moving to home position')
    args = parser.parse_args()

    rclpy.init()
    controller = SmoothPickController()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(controller)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    if args.test_home:
        controller.go_home()
    else:
        controller.pick_top_box()

    time.sleep(2)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
