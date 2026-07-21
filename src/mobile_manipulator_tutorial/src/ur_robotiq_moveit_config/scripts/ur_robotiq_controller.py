#!/usr/bin/env python3

import time
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
					sleep_time=0.0):
	
	logger.info("Planning trajectory")

	# Keep arm in natural "elbow down" configuration
	constraints = Constraints()
	constraints.name = "natural_pose"

	# Shoulder lift: keep around -90° (pointing down)
	shoulder_c = JointConstraint()
	shoulder_c.joint_name = "shoulder_lift_joint"
	shoulder_c.position = -np.pi/2
	shoulder_c.tolerance_above = np.pi/6  # ±30°
	shoulder_c.tolerance_below = np.pi/6
	shoulder_c.weight = 1.0
	constraints.joint_constraints.append(shoulder_c)

	# Elbow: keep positive (elbow down, not flipped)
	elbow_c = JointConstraint()
	elbow_c.joint_name = "elbow_joint"
	elbow_c.position = np.pi/2
	elbow_c.tolerance_above = np.pi/3  # ±60°
	elbow_c.tolerance_below = np.pi/3
	elbow_c.weight = 1.0
	constraints.joint_constraints.append(elbow_c)

	planning_component.set_path_constraints(constraints)


	# plan_result = planning_component.plan()
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
			objects_to_add = [
				("base", [1.5, 1.5, 0.5], [0., 0., 0.75]),
			]
			for name, size, position in objects_to_add:
				self.add_box(scene, name, size, position, base=True)
				self.get_logger().info(f"\tSuccessful placement of {name}")
			scene.current_state.update()

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
		"""Move to specific joint angles (avoids IK flip issues)."""
		self.get_logger().info(f"Moving joints: {joint_dict}")
		self.ur_arm.set_start_state_to_current_state()
		self.robot_state.joint_positions = joint_dict
		jc = construct_joint_constraint(
			robot_state=self.robot_state,
			joint_model_group=self.ur.get_robot_model().get_joint_model_group("ur_manipulator"),
		)
		self.ur_arm.set_goal_state(motion_plan_constraints=[jc])
		return plan_and_execute(self.ur, self.ur_arm, self.logger, sleep_time=0.5)

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
			# Mode 4: ALL joint-space pick-and-place (no Cartesian goals)
			# Avoids IK flip issues entirely by using only joint-space waypoints
			# data = [pick_x, pick_y, pick_z, 0,0,0,0, place_x, place_y, place_z, 0,0,0,0, 4.0]
			pick_x, pick_y, pick_z = data.data[0], data.data[1], data.data[2]
			place_x, place_y, place_z = data.data[7], data.data[8], data.data[9]

			valid = True
			self.get_logger().info(f"\n=== JOINT-SPACE PICK: ({pick_x:.2f},{pick_y:.2f},{pick_z:.2f}) → ({place_x:.2f},{place_y:.2f},{place_z:.2f}) ===\n")

			# UR5e ready position: tool0 at ~(0.49, 0.13, 0.49) in base_link
			# Ready joints: SP=0, SL=-pi/2, EL=pi/2, W1=-pi/2, W2=-pi/2, W3=0
			# Tool points DOWN at ready position.
			#
			# Key kinematic insight:
			# - shoulder_pan rotates the entire arm around Z (base)
			# - shoulder_lift + elbow control reach and height
			# - wrist_1 = -(SL + EL) - pi/2 keeps tool vertical (pointing down)
			# - wrist_2 = -pi/2 for top-down grasp
			# - wrist_3 = 0 (no end-effector rotation needed)
			#
			# To reach lower (smaller z): make SL less negative (tilt arm forward)
			#   and adjust EL accordingly
			# To change y: rotate shoulder_pan
			#
			# Target: box_top at (0.50, 0.00, 0.14) in base_link
			# The y=0.13 offset from wrist means SP≈0 naturally reaches y≈0.13
			# For y≈0 need SP≈-0.26 (small correction)
			# For y≈-0.30 (place) need SP≈-0.55

			import math

			# Compute shoulder_pan for pick and place y-positions
			# At ready, reach ≈ 0.49m in XZ plane, wrist y-offset ≈ 0.13m
			pick_sp = math.atan2(-(pick_y - 0.133), 0.49)   # compensate wrist offset
			place_sp = math.atan2(-(place_y - 0.133), 0.49)

			self.get_logger().info(f"  pick_sp={pick_sp:.3f}, place_sp={place_sp:.3f}")

			# --- Step 1: Go to ready ---
			self.get_logger().info("Step 1: Go to ready")
			valid = valid and self.go_named("ready")
			if not valid:
				self.get_logger().error("FAILED at step 1 (ready)")

			# --- Step 2: Open gripper ---
			self.get_logger().info("Step 2: Open gripper")
			valid = valid and self.gripper_action("open")
			if not valid:
				self.get_logger().error("FAILED at step 2 (open gripper)")

			# --- Step 3: Pre-approach above pick (high, safe) ---
			# Slightly lower than ready to start descending
			self.get_logger().info("Step 3: Pre-approach above pick")
			valid = valid and self.move_joints({
				"shoulder_pan_joint": pick_sp,
				"shoulder_lift_joint": -1.40,
				"elbow_joint": 1.20,
				"wrist_1_joint": -1.37,  # -(−1.40+1.20)−pi/2 = 0.20−1.57 = −1.37
				"wrist_2_joint": -1.5707,
				"wrist_3_joint": 0.0,
			})
			if not valid:
				self.get_logger().error("FAILED at step 3 (pre-approach)")

			# --- Step 4: Lower to grasp approach ---
			# Tilt arm more forward to descend
			self.get_logger().info("Step 4: Approach grasp height")
			valid = valid and self.move_joints({
				"shoulder_pan_joint": pick_sp,
				"shoulder_lift_joint": -1.10,
				"elbow_joint": 1.20,
				"wrist_1_joint": -1.67,  # -(−1.10+1.20)−pi/2 = −0.10−1.57 = −1.67
				"wrist_2_joint": -1.5707,
				"wrist_3_joint": 0.0,
			})
			if not valid:
				self.get_logger().error("FAILED at step 4 (approach)")

			# --- Step 5: Final grasp position (lowest) ---
			self.get_logger().info("Step 5: Grasp position")
			valid = valid and self.move_joints({
				"shoulder_pan_joint": pick_sp,
				"shoulder_lift_joint": -0.85,
				"elbow_joint": 1.10,
				"wrist_1_joint": -1.82,  # -(−0.85+1.10)−pi/2 = −0.25−1.57 = −1.82
				"wrist_2_joint": -1.5707,
				"wrist_3_joint": 0.0,
			})
			if not valid:
				self.get_logger().error("FAILED at step 5 (grasp pos)")

			# --- Step 6: Close gripper ---
			self.get_logger().info("Step 6: Close gripper")
			valid = valid and self.gripper_action("close")
			time.sleep(0.5)
			if not valid:
				self.get_logger().error("FAILED at step 6 (close gripper)")

			# --- Step 7: Lift up ---
			self.get_logger().info("Step 7: Lift with object")
			valid = valid and self.move_joints({
				"shoulder_pan_joint": pick_sp,
				"shoulder_lift_joint": -1.40,
				"elbow_joint": 1.20,
				"wrist_1_joint": -1.37,
				"wrist_2_joint": -1.5707,
				"wrist_3_joint": 0.0,
			})
			if not valid:
				self.get_logger().error("FAILED at step 7 (lift)")

			# --- Step 8: Transit to above place (rotate shoulder) ---
			self.get_logger().info("Step 8: Transit to place position")
			valid = valid and self.move_joints({
				"shoulder_pan_joint": place_sp,
				"shoulder_lift_joint": -1.40,
				"elbow_joint": 1.20,
				"wrist_1_joint": -1.37,
				"wrist_2_joint": -1.5707,
				"wrist_3_joint": 0.0,
			})
			if not valid:
				self.get_logger().error("FAILED at step 8 (transit)")

			# --- Step 9: Lower to place ---
			self.get_logger().info("Step 9: Lower to place")
			valid = valid and self.move_joints({
				"shoulder_pan_joint": place_sp,
				"shoulder_lift_joint": -0.85,
				"elbow_joint": 1.10,
				"wrist_1_joint": -1.82,
				"wrist_2_joint": -1.5707,
				"wrist_3_joint": 0.0,
			})
			if not valid:
				self.get_logger().error("FAILED at step 9 (lower)")

			# --- Step 10: Release ---
			self.get_logger().info("Step 10: Release object")
			valid = valid and self.gripper_action("open")
			if not valid:
				self.get_logger().error("FAILED at step 10 (release)")

			# --- Step 11: Retreat up ---
			self.get_logger().info("Step 11: Retreat")
			valid = valid and self.move_joints({
				"shoulder_pan_joint": place_sp,
				"shoulder_lift_joint": -1.40,
				"elbow_joint": 1.20,
				"wrist_1_joint": -1.37,
				"wrist_2_joint": -1.5707,
				"wrist_3_joint": 0.0,
			})
			if not valid:
				self.get_logger().error("FAILED at step 11 (retreat)")

			# --- Step 12: Return home ---
			self.get_logger().info("Step 12: Return to ready")
			valid = valid and self.go_named("ready")


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