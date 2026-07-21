#!/usr/bin/env python3

import time
import math
import threading
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.logging import get_logger
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, String
from moveit.core.robot_state import RobotState  # Correct import
from moveit.planning import MoveItPy, MultiPipelinePlanRequestParameters
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from moveit.core.kinematic_constraints import construct_joint_constraint
from geometry_msgs.msg import Pose
from moveit_msgs.msg import Constraints, JointConstraint



def plan_and_execute(robot, planning_component, logger,
					single_plan_parameters=None,
					multi_plan_parameters=None,
					sleep_time=0.0,
					use_constraints=False):

	logger.info("Planning trajectory")

	if use_constraints:
		# Keep arm in natural "elbow down" configuration
		# Wider range to accommodate all reachable positions
		constraints = Constraints()
		constraints.name = "natural_pose"

		shoulder_c = JointConstraint()
		shoulder_c.joint_name = "shoulder_lift_joint"
		shoulder_c.position = -1.2  # center of working range
		shoulder_c.tolerance_above = 0.7  # allows up to -0.5
		shoulder_c.tolerance_below = 0.7  # allows down to -1.9
		shoulder_c.weight = 1.0
		constraints.joint_constraints.append(shoulder_c)

		elbow_c = JointConstraint()
		elbow_c.joint_name = "elbow_joint"
		elbow_c.position = 2.0  # center of working range
		elbow_c.tolerance_above = 0.7  # allows up to 2.7
		elbow_c.tolerance_below = 0.7  # allows down to 1.3
		elbow_c.weight = 1.0
		constraints.joint_constraints.append(elbow_c)

		planning_component.set_path_constraints(constraints)

	plan_result = False
	if multi_plan_parameters is not None:
			plan_result = planning_component.plan(
					multi_plan_parameters=multi_plan_parameters
			)
	elif single_plan_parameters is not None:
			plan_result = planning_component.plan(
					single_plan_parameters=single_plan_parameters
			)
	else:
			plan_result = planning_component.plan()

	if not plan_result:
		logger.error("Planning failed. No valid trajectory generated.")
		return False
	logger.info("Executing plan")
	robot_trajectory = plan_result.trajectory
	if robot_trajectory:
		robot.execute(robot_trajectory, controllers=[])
		time.sleep(sleep_time)
		return True
	else:
		logger.error("Execution failed. No valid trajectory.")
		return False

class Controller(Node):
	def __init__(self):
		super().__init__('commander')
		self.get_logger().info("\n\n\nInitialize Manipulator\n\n\n")

		self.subscription = self.create_subscription(
			Float64MultiArray, '/target_point', self.listener_callback, 10
		)
		self.status_pub = self.create_publisher(String, '/pick_place_status', 10)

		self.pose_goal = PoseStamped()
		self.pose_goal.header.frame_id = "base_link"

		self.ur = MoveItPy(node_name="moveit_py")
		self.get_logger().info("\n\n\nSuccessfully called moveit py node\n\n\n")
		self.ur_arm = self.ur.get_planning_component("ur_manipulator")
		self.ur_hand = self.ur.get_planning_component("robotiq_2f_85_gripper")
		self.logger = get_logger("moveit_py.pose_goal")

		robot_model = self.ur.get_robot_model()
		self.robot_state = RobotState(robot_model)

		# Heights and offsets
		self.backoff_dist = 0.4
		self.grasp_dist = 0.15
		self.carrying_height = 0.4

		# Side grasping orientation (w, x, y, z)
		self.side_grasp_orientation = (0.7071067811865476, 0.0, 0.7071067811865475, 0.0)
		self.top_grasp_orientation = (1., 0., 0., 0.)

		time.sleep(10)
		self.setup_workspace(self.ur)

	def setup_workspace(self, moveit):
		with moveit.get_planning_scene_monitor().read_write() as scene:
			# No collision objects — let the planner reach freely
			# The table is handled by Gazebo physics
			scene.current_state.update()
			self.get_logger().info("Workspace setup (no collision objects)")

	def add_box(self, scene, name, size, position, base=False):
		co = CollisionObject()
		co.header.frame_id = 'world'
		co.id = name
		box = SolidPrimitive()
		box.type = SolidPrimitive.BOX
		box.dimensions = size
		co.primitives.append(box)
		pose = Pose()
		pose.position.x, pose.position.y, pose.position.z = position
		co.primitive_poses.append(pose)
		co.operation = CollisionObject.ADD
		scene.apply_collision_object(co)

	def publish_status(self, status_msg):
		status = String()
		status.data = status_msg
		self.status_pub.publish(status)
		self.get_logger().info(f"[DEBUG] Publishing pick-place completion: {status_msg}")

	def move_to(self, x, y, z, xo, yo, zo, wo):
		self.ur_arm.set_start_state_to_current_state()
		self.pose_goal.pose.position.x = x
		self.pose_goal.pose.position.y = y
		self.pose_goal.pose.position.z = z
		self.pose_goal.pose.orientation.x = xo
		self.pose_goal.pose.orientation.y = yo
		self.pose_goal.pose.orientation.z = zo
		self.pose_goal.pose.orientation.w = wo
		self.ur_arm.set_goal_state(pose_stamped_msg=self.pose_goal, pose_link="tool0")
		multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
				self.ur, ["ompl_rrtc", "pilz_lin", "chomp_planner", "ompl_rrt_star", "stomp_planner"]
		)
		return plan_and_execute(self.ur, self.ur_arm, self.logger, multi_plan_parameters=multi_pipeline_plan_request_params, sleep_time=1.)

	def go_named(self, name):
		"""Move to a named configuration (e.g., 'ready')."""
		self.get_logger().info(f"Moving to named state: {name}")
		self.ur_arm.set_start_state_to_current_state()
		self.ur_arm.set_goal_state(configuration_name=name)
		return plan_and_execute(self.ur, self.ur_arm, self.logger, sleep_time=1.)

	def move_joints(self, joint_dict):
		"""Move to specific joint angles via joint constraint goal."""
		self.get_logger().info(f"Moving joints: {joint_dict}")
		self.ur_arm.set_start_state_to_current_state()
		self.robot_state.joint_positions = joint_dict
		jc = construct_joint_constraint(
			robot_state=self.robot_state,
			joint_model_group=self.ur.get_robot_model().get_joint_model_group("ur_manipulator"),
		)
		self.ur_arm.set_goal_state(motion_plan_constraints=[jc])
		return plan_and_execute(self.ur, self.ur_arm, self.logger, sleep_time=1.0)

	def move_to_down(self, x, y, z, linear=False, tilt=False):
		"""Move tool0 to (x,y,z) with tool pointing down.
		If linear=True, use Pilz LIN only for straight-line Cartesian path.
		If tilt=True, tilt tool 15° forward for better fingertip reach."""
		self.ur_arm.set_start_state_to_current_state()
		self.pose_goal.pose.position.x = x
		self.pose_goal.pose.position.y = y
		self.pose_goal.pose.position.z = z
		if tilt:
			# 15° tilt from vertical — fingertips reach ~3cm lower
			self.pose_goal.pose.orientation.x = 0.0
			self.pose_goal.pose.orientation.y = 0.6088
			self.pose_goal.pose.orientation.z = 0.0
			self.pose_goal.pose.orientation.w = 0.7934
		else:
			# Straight down
			self.pose_goal.pose.orientation.x = 0.0
			self.pose_goal.pose.orientation.y = 0.7071068
			self.pose_goal.pose.orientation.z = 0.0
			self.pose_goal.pose.orientation.w = 0.7071068
		self.ur_arm.set_goal_state(pose_stamped_msg=self.pose_goal, pose_link="tool0")
		if linear:
			# Pilz LIN = straight line in Cartesian space (won't bump objects)
			multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
					self.ur, ["pilz_lin"]
			)
		else:
			multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
					self.ur, ["ompl_rrtc", "pilz_lin", "chomp_planner", "stomp_planner"]
			)
		return plan_and_execute(self.ur, self.ur_arm, self.logger,
			multi_plan_parameters=multi_pipeline_plan_request_params,
			sleep_time=1.0)

	def gripper_action(self, action):
		self.ur_hand.set_start_state_to_current_state()
		if action == 'open':
			joint_values = {"robotiq_85_left_knuckle_joint": 0.01}
		elif action == 'close':
			joint_values = {"robotiq_85_left_knuckle_joint": 0.65}
		else:
			self.get_logger().info("No such action")
			return False
		self.robot_state.joint_positions = joint_values
		joint_constraint = construct_joint_constraint(
			robot_state=self.robot_state,
			joint_model_group=self.ur.get_robot_model().get_joint_model_group("robotiq_2f_85_gripper"),
		)
		self.ur_hand.set_goal_state(motion_plan_constraints=[joint_constraint])
		return plan_and_execute(self.ur, self.ur_hand, self.logger, sleep_time=1.)

	def listener_callback(self, data):
		self.get_logger().info(f"Received target point: {[f'{x:.2f}' for x in data.data]}")
		self.publish_status("IN_PROGRESS")
		valid = False  # default for unknown modes

		if data.data[-1] == 1.0:
			# Load mode
			start_x, start_y, start_z, start_xo, start_yo, start_zo, start_wo = data.data[0:7]
			goal_x, goal_y, goal_z, goal_xo, goal_yo, goal_zo, goal_wo = data.data[7:14]
			start_orientation = (start_xo, start_yo, start_zo, start_wo)
			goal_orientation = (goal_xo, goal_yo, goal_zo, goal_wo)

			valid = True

			valid = valid and self.move_to(0.4, 0.0, self.carrying_height, 1., 0., 0., 0.)

			# Backoff position (approach from side, along X-axis)
			start_backoff_x = start_x - self.backoff_dist
			self.get_logger().info(f"\n\nMoving to backoff position {start_backoff_x, start_y, start_z}\n\n")
			valid = valid and self.move_to(start_backoff_x, start_y, start_z, *start_orientation)

			print("\n\n\n\n\n\n")
			# Open gripper
			self.get_logger().info("Opening gripper")
			valid = valid and self.gripper_action("open")

			# Move to pick position (side grasp)
			start_grasp_x = start_x - 0.1
			self.get_logger().info(f"\n\nMoving to pick position {start_grasp_x, start_y, start_z}")
			valid = valid and self.move_to(start_grasp_x, start_y, start_z, *start_orientation)

			# Grasp (close gripper)
			self.get_logger().info("Grasping object")
			valid = valid and self.gripper_action("close")

			# Backup after grasping
			self.get_logger().info(f"Back up after grasp {start_backoff_x, start_y, start_z}")
			valid = valid and self.move_to(start_backoff_x, start_y, start_z, *start_orientation)

			# Move to goal backoff position
			goal_backoff_z = goal_z + self.backoff_dist
			self.get_logger().info("Moving to goal backoff position")
			valid = valid and self.move_to(goal_x, goal_y, goal_backoff_z, *goal_orientation)

			# Lower to place position
			goal_grasp_z = goal_z + self.grasp_dist
			self.get_logger().info("Lowering to place position")
			valid = valid and self.move_to(goal_x, goal_y, goal_grasp_z, *goal_orientation)

			# Release (open gripper)
			self.get_logger().info("Releasing object")
			valid = valid and self.gripper_action("open")

			# Backup after placing
			self.get_logger().info("Backing up after place")
			valid = valid and self.move_to(goal_x, goal_y, goal_backoff_z, *goal_orientation)

			valid = valid and self.move_to(0.4, 0.0, self.carrying_height, 1., 0., 0., 0.)

		elif data.data[-1] == 2.0:
			# Unload mode
			start_x, start_y, start_z, start_xo, start_yo, start_zo, start_wo = data.data[0:7]
			goal_x, goal_y, goal_z, goal_xo, goal_yo, goal_zo, goal_wo = data.data[7:14]
			start_orientation = (start_xo, start_yo, start_zo, start_wo)
			goal_orientation = (goal_xo, goal_yo, goal_zo, goal_wo)

			valid = True

			valid = valid and self.move_to(0.4, 0.0, self.carrying_height, 1., 0., 0., 0.)

			# Backoff position (approach from side, along Z-axis)
			start_backoff_z = start_z + self.backoff_dist
			self.get_logger().info("Moving to backoff position")
			valid = valid and self.move_to(start_x, start_y, start_backoff_z, *start_orientation)

			# Open gripper
			self.get_logger().info("Opening gripper")
			valid = valid and self.gripper_action("open")

			# Move to pick position (top grasp)
			self.get_logger().info("Moving to pick position")
			start_graspf_z = start_z + self.grasp_dist
			valid = valid and self.move_to(start_x, start_y, start_graspf_z, *start_orientation)

			# Grasp (close gripper)
			self.get_logger().info("Grasping object")
			valid = valid and self.gripper_action("close")

			# Backoff position (approach from side, along Z-axis)
			self.get_logger().info("Moving to backoff position")
			valid = valid and self.move_to(start_x, start_y, start_backoff_z, *start_orientation)

			# Backup after grasping
			backoff_x = goal_x - self.backoff_dist
			self.get_logger().info("Moving to backoff position")
			valid = valid and self.move_to(backoff_x, goal_y, goal_z, *goal_orientation)

			# Move to release position (side grasp)
			self.get_logger().info("Moving to pick position")
			goal_grasp_x = goal_x - self.grasp_dist
			valid = valid and self.move_to(goal_grasp_x, goal_y, goal_z, *goal_orientation)

			# Release (release gripper)
			self.get_logger().info("Grasping object")
			valid = valid and self.gripper_action("open")

			# Backup after unloading
			self.get_logger().info("Moving to backoff position")
			valid = valid and self.move_to(backoff_x, goal_y, goal_z, *goal_orientation)

			valid = valid and self.move_to(0.4, 0.0, self.carrying_height, 1., 0., 0., 0.)

		elif data.data[-1] == 3.0:
			# Simple Pick and Place
			start_x, start_y, start_z, start_xo, start_yo, start_zo, start_wo = data.data[0:7]
			goal_x, goal_y, goal_z, goal_xo, goal_yo, goal_zo, goal_wo = data.data[7:14]
			start_orientation = (start_xo, start_yo, start_zo, start_wo)
			goal_orientation = (goal_xo, goal_yo, goal_zo, goal_wo)

			valid = True

			valid = valid and self.move_to(0.4, 0.0, self.carrying_height, 1., 0., 0., 0.)

			# Backoff position (approach from side, along X-axis)
			start_backoff_z = start_z + 0.25
			self.get_logger().info(f"\n\nMoving to backoff position {start_x, start_y, start_backoff_z}\n\n")
			valid = valid and self.move_to(start_x, start_y, start_backoff_z, *start_orientation)

			print("\n\n\n\n\n\n")
			# Open gripper
			self.get_logger().info("Opening gripper")
			valid = valid and self.gripper_action("open")

			# Move to pick position (top grasp)
			start_grasp_z = start_z + 0.15
			self.get_logger().info(f"\n\nMoving to pick position {start_x, start_y, start_grasp_z}")
			valid = valid and self.move_to(start_x, start_y, start_grasp_z, *start_orientation)

			# Grasp (close gripper)
			self.get_logger().info("Grasping object")
			valid = valid and self.gripper_action("close")

			# Backup after grasping
			self.get_logger().info(f"Back up after grasp {start_x, start_y, start_backoff_z}")
			valid = valid and self.move_to(start_x, start_y, start_backoff_z, *start_orientation)

			# Move to goal backoff position
			goal_backoff_z = goal_z + 0.25
			self.get_logger().info("Moving to goal backoff position")
			valid = valid and self.move_to(goal_x, goal_y, goal_backoff_z, *goal_orientation)

			# Lower to place position
			goal_grasp_z = goal_z + 0.15
			self.get_logger().info("Lowering to place position")
			valid = valid and self.move_to(goal_x, goal_y, goal_grasp_z, *goal_orientation)

			# Release (open gripper)
			self.get_logger().info("Releasing object")
			valid = valid and self.gripper_action("open")

			# Backup after placing
			self.get_logger().info("Backing up after place")
			valid = valid and self.move_to(goal_x, goal_y, goal_backoff_z, *goal_orientation)

			valid = valid and self.move_to(0.4, 0.0, self.carrying_height, 1., 0., 0., 0.)

		elif data.data[-1] == 4.0:
			# Mode 4: Side-grasp pick-and-place
			# Approaches box from the SIDE (along X in base_link = robot forward)
			# Gripper horizontal, fingers wrap around box sides
			# Same approach as mode 1 (Load) which works in workstation demo
			#
			# Side grasp orientation: tool0 points along +X (toward box)
			# Quaternion (w,x,y,z) = (0.707, 0, 0.707, 0)
			pick_x, pick_y, pick_z = data.data[0], data.data[1], data.data[2]
			place_x, place_y, place_z = data.data[7], data.data[8], data.data[9]

			valid = True
			self.get_logger().info(f"\n=== SIDE-GRASP PICK: ({pick_x:.3f},{pick_y:.3f},{pick_z:.3f}) -> ({place_x:.3f},{place_y:.3f},{place_z:.3f}) ===\n")

			# Side grasp: tool0 approaches along X axis
			# backoff = 0.30m before box (along -X), grasp = 0.14m before box
			BACKOFF = 0.30
			GRASP_OFFSET = 0.14  # tool0 to fingertip ~0.15m, box half-width 0.03
			LIFT_HEIGHT = 0.35

			so = self.side_grasp_orientation  # (w, x, y, z) = (0.707, 0, 0.707, 0)

			# --- Step 1: Go to ready ---
			self.get_logger().info("Step 1: Go to ready")
			valid = valid and self.go_named("ready")

			# --- Step 2: Open gripper ---
			if valid:
				self.get_logger().info("Step 2: Open gripper")
				valid = valid and self.gripper_action("open")

			# --- Step 3: Move to carrying height ---
			if valid:
				self.get_logger().info("Step 3: Carrying height")
				valid = valid and self.move_to(0.4, 0.0, self.carrying_height, 1., 0., 0., 0.)

			# --- Step 4: Backoff position (behind box, same height) ---
			if valid:
				backoff_x = pick_x - BACKOFF
				self.get_logger().info(f"Step 4: Backoff ({backoff_x:.2f}, {pick_y:.2f}, {pick_z:.2f})")
				valid = valid and self.move_to(backoff_x, pick_y, pick_z, *so)

			# --- Step 5: Approach box (linear, same height) ---
			if valid:
				grasp_x = pick_x - GRASP_OFFSET
				self.get_logger().info(f"Step 5: Approach ({grasp_x:.2f}, {pick_y:.2f}, {pick_z:.2f})")
				valid = valid and self.move_to(grasp_x, pick_y, pick_z, *so)

			# --- Step 6: Close gripper ---
			if valid:
				self.get_logger().info("Step 6: Close gripper")
				valid = valid and self.gripper_action("close")
				time.sleep(0.5)

			# --- Step 7: Pull back with box ---
			if valid:
				backoff_x = pick_x - BACKOFF
				self.get_logger().info("Step 7: Pull back")
				valid = valid and self.move_to(backoff_x, pick_y, pick_z, *so)

			# --- Step 8: Lift to carrying height ---
			if valid:
				self.get_logger().info("Step 8: Lift")
				valid = valid and self.move_to(0.4, 0.0, self.carrying_height, 1., 0., 0., 0.)

			# --- Step 9: Move to place backoff ---
			if valid:
				place_backoff_z = place_z + BACKOFF
				self.get_logger().info("Step 9: Above place")
				valid = valid and self.move_to(place_x, place_y, place_backoff_z, 1., 0., 0., 0.)

			# --- Step 10: Lower to place ---
			if valid:
				place_grasp_z = place_z + GRASP_OFFSET
				self.get_logger().info("Step 10: Lower to place")
				valid = valid and self.move_to(place_x, place_y, place_grasp_z, 1., 0., 0., 0.)

			# --- Step 11: Release ---
			if valid:
				self.get_logger().info("Step 11: Release")
				valid = valid and self.gripper_action("open")

			# --- Step 12: Retreat and home ---
			if valid:
				self.get_logger().info("Step 12: Retreat and home")
				place_backoff_z = place_z + BACKOFF
				valid = valid and self.move_to(place_x, place_y, place_backoff_z, 1., 0., 0., 0.)
				valid = valid and self.move_to(0.4, 0.0, self.carrying_height, 1., 0., 0., 0.)

		elif data.data[-1] == 5.0:
			# Mode 5: 6-DOF grasp from Contact-GraspNet
			# Input: [pick_x, pick_y, pick_z, qx, qy, qz, qw,
			#         place_x, place_y, place_z, 0, 0, 0, 0, 5.0]
			# The grasp pose is a full 6-DOF pose for tool0 in base_link frame.
			# Approach: backoff along the grasp approach direction (-Z of grasp frame),
			# then linear approach to grasp, close, retract, move to place.
			pick_x, pick_y, pick_z = data.data[0], data.data[1], data.data[2]
			qx, qy, qz, qw = data.data[3], data.data[4], data.data[5], data.data[6]
			place_x, place_y, place_z = data.data[7], data.data[8], data.data[9]

			valid = True
			self.get_logger().info(
				f"\n=== 6-DOF GRASP: pos=({pick_x:.3f},{pick_y:.3f},{pick_z:.3f}) "
				f"quat=({qw:.3f},{qx:.3f},{qy:.3f},{qz:.3f}) ===\n")

			BACKOFF = 0.20   # approach distance along grasp axis
			LIFT_HEIGHT = 0.35

			# Compute approach direction from quaternion (grasp frame -Z in base_link)
			# The grasp frame Z axis points away from the object surface.
			# Approach from behind: move along -Z of grasp frame.
			import numpy as _np
			# Quaternion to rotation matrix (scipy-free)
			x2, y2, z2 = qx*2, qy*2, qz*2
			xx, xy, xz = qx*x2, qx*y2, qx*z2
			yy, yz, zz = qy*y2, qy*z2, qz*z2
			wx, wy, wz = qw*x2, qw*y2, qw*z2
			R = _np.array([
				[1-(yy+zz), xy-wz,     xz+wy    ],
				[xy+wz,     1-(xx+zz), yz-wx    ],
				[xz-wy,     yz+wx,     1-(xx+yy)],
			])
			# Approach direction = +Z of grasp frame (towards object)
			# Backoff direction = -Z of grasp frame (away from object)
			approach_dir = R[:, 2]  # Z column of rotation matrix
			backoff_dir = -approach_dir

			# Backoff position (start further from object)
			backoff_x = pick_x + backoff_dir[0] * BACKOFF
			backoff_y = pick_y + backoff_dir[1] * BACKOFF
			backoff_z = pick_z + backoff_dir[2] * BACKOFF

			# --- Step 1: Go to ready ---
			self.get_logger().info("Step 1: Go to ready")
			valid = valid and self.go_named("ready")

			# --- Step 2: Open gripper ---
			if valid:
				self.get_logger().info("Step 2: Open gripper")
				valid = valid and self.gripper_action("open")

			# --- Step 3: Move to carrying height ---
			if valid:
				self.get_logger().info("Step 3: Carrying height")
				valid = valid and self.move_to(0.4, 0.0, self.carrying_height, 1., 0., 0., 0.)

			# --- Step 4: Move to backoff position (away from grasp) ---
			if valid:
				self.get_logger().info(f"Step 4: Backoff ({backoff_x:.3f}, {backoff_y:.3f}, {backoff_z:.3f})")
				valid = valid and self.move_to(backoff_x, backoff_y, backoff_z, qx, qy, qz, qw)

			# --- Step 5: Approach grasp position ---
			if valid:
				self.get_logger().info(f"Step 5: Approach grasp ({pick_x:.3f}, {pick_y:.3f}, {pick_z:.3f})")
				valid = valid and self.move_to(pick_x, pick_y, pick_z, qx, qy, qz, qw)

			# --- Step 6: Close gripper ---
			if valid:
				self.get_logger().info("Step 6: Close gripper")
				valid = valid and self.gripper_action("close")
				time.sleep(0.5)

			# --- Step 7: Retract to backoff ---
			if valid:
				self.get_logger().info("Step 7: Retract")
				valid = valid and self.move_to(backoff_x, backoff_y, backoff_z, qx, qy, qz, qw)

			# --- Step 8: Lift to carrying height ---
			if valid:
				self.get_logger().info("Step 8: Lift to carrying height")
				valid = valid and self.move_to(0.4, 0.0, self.carrying_height, 1., 0., 0., 0.)

			# --- Step 9: Move above place position ---
			if valid:
				place_above_z = place_z + BACKOFF
				self.get_logger().info(f"Step 9: Above place ({place_x:.3f}, {place_y:.3f}, {place_above_z:.3f})")
				valid = valid and self.move_to(place_x, place_y, place_above_z, 1., 0., 0., 0.)

			# --- Step 10: Lower to place ---
			if valid:
				place_grasp_z = place_z + 0.14
				self.get_logger().info("Step 10: Lower to place")
				valid = valid and self.move_to(place_x, place_y, place_grasp_z, 1., 0., 0., 0.)

			# --- Step 11: Release ---
			if valid:
				self.get_logger().info("Step 11: Release")
				valid = valid and self.gripper_action("open")

			# --- Step 12: Retreat and home ---
			if valid:
				self.get_logger().info("Step 12: Retreat and home")
				place_above_z = place_z + BACKOFF
				valid = valid and self.move_to(place_x, place_y, place_above_z, 1., 0., 0., 0.)
				valid = valid and self.move_to(0.4, 0.0, self.carrying_height, 1., 0., 0., 0.)

		if valid:
			self.publish_status("DONE")
		else:
			self.publish_status("FAILED")

if __name__ == '__main__':
	rclpy.init(args=None)
	controller = Controller()
	executor = rclpy.executors.MultiThreadedExecutor()
	executor.add_node(controller)
	executor_thread = threading.Thread(target=executor.spin, daemon=True)
	executor_thread.start()
	rate = controller.create_rate(2)
	try:
		while rclpy.ok():
			rate.sleep()
	except KeyboardInterrupt:
		pass
	rclpy.shutdown()
	executor_thread.join()