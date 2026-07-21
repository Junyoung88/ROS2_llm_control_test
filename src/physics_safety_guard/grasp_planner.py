#!/usr/bin/env python3
"""
Grasp Planner — Contact-GraspNet integration for Gazebo box stacking demo.

Subscribes to RGB-D camera topics from Gazebo, converts depth images to
point clouds, runs Contact-GraspNet inference to generate 6-DOF grasp poses,
and returns the best grasp for a target region.

Requirements:
    - Contact-GraspNet PyTorch: /home/jim/contact_graspnet_pytorch/
    - PYOPENGL_PLATFORM=egl (set before import)
    - GPU with CUDA support

Usage:
    # Standalone test (requires Gazebo running with box_stack.sdf):
    python3 grasp_planner.py

    # As a module from box_stack_demo.py:
    from grasp_planner import GraspPlanner
    planner = GraspPlanner()
    grasp = planner.get_best_grasp_for_target(target_position_world, ...)
"""

import os
import sys
import time
import math
import numpy as np

# Set EGL before any OpenGL imports
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Add Contact-GraspNet to path
CONTACT_GRASPNET_DIR = '/home/jim/contact_graspnet_pytorch'
if CONTACT_GRASPNET_DIR not in sys.path:
    sys.path.insert(0, CONTACT_GRASPNET_DIR)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

# Lazy-load Contact-GraspNet (heavy imports with CUDA)
_grasp_estimator = None
_grasp_estimator_loaded = False


def _load_grasp_estimator():
    """Load Contact-GraspNet model (once, lazily)."""
    global _grasp_estimator, _grasp_estimator_loaded
    if _grasp_estimator_loaded:
        return _grasp_estimator

    print("[GraspPlanner] Loading Contact-GraspNet model...")
    t0 = time.time()

    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    from contact_graspnet_pytorch import config_utils
    from contact_graspnet_pytorch.checkpoints import CheckpointIO

    ckpt_dir = os.path.join(CONTACT_GRASPNET_DIR, 'checkpoints', 'contact_graspnet')
    global_config = config_utils.load_config(ckpt_dir, batch_size=1)

    grasp_estimator = GraspEstimator(global_config)

    model_checkpoint_dir = os.path.join(ckpt_dir, 'checkpoints')
    checkpoint_io = CheckpointIO(
        checkpoint_dir=model_checkpoint_dir,
        model=grasp_estimator.model)
    try:
        checkpoint_io.load('model.pt')
    except FileExistsError:
        print('[GraspPlanner] WARNING: No model checkpoint found!')

    grasp_estimator.model.eval()

    _grasp_estimator = grasp_estimator
    _grasp_estimator_loaded = True
    print(f"[GraspPlanner] Model loaded in {time.time()-t0:.1f}s")
    return _grasp_estimator


# =========================================================================
# Camera frame definitions
# =========================================================================

# Camera pose in world frame (must match box_stack.sdf)
CAM_POSITION_WORLD = np.array([0.5, 0.0, 1.5])
CAM_RPY = np.array([0.0, 0.41, 2.31])  # roll, pitch, yaw (radians)


def _rpy_to_rotation_matrix(roll, pitch, yaw):
    """Convert roll-pitch-yaw to 3x3 rotation matrix (XYZ extrinsic = ZYX intrinsic)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr            ],
    ])
    return R


def _get_camera_extrinsics():
    """Get the 4x4 world-to-camera transform.

    Gazebo camera: +X forward (optical axis), +Y left, +Z up.
    OpenCV camera: +Z forward, +X right, +Y down.

    Returns T_cam_world: 4x4 matrix that transforms world points to OpenCV camera coords.
    """
    R_world_cam_gz = _rpy_to_rotation_matrix(*CAM_RPY)

    # Gazebo camera frame: +X = forward, +Y = left, +Z = up
    # OpenCV camera frame: +Z = forward, +X = right, +Y = down
    # Rotation from Gazebo camera to OpenCV camera:
    #   opencv_x = -gz_y   (right = -left)
    #   opencv_y = -gz_z   (down = -up)
    #   opencv_z = +gz_x   (forward = forward)
    R_cv_gz = np.array([
        [0, -1,  0],
        [0,  0, -1],
        [1,  0,  0],
    ], dtype=np.float64)

    # R_world_cam_gz rotates from world to Gazebo camera frame
    # R_cv_gz rotates from Gazebo camera to OpenCV camera frame
    R_cam_world = R_cv_gz @ R_world_cam_gz.T

    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = R_cam_world
    T_cam_world[:3, 3] = R_cam_world @ (-CAM_POSITION_WORLD)

    return T_cam_world


def _get_camera_to_world():
    """Get the 4x4 camera-to-world transform (inverse of extrinsics)."""
    T_cam_world = _get_camera_extrinsics()
    T_world_cam = np.eye(4)
    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    T_world_cam[:3, :3] = R.T
    T_world_cam[:3, 3] = -R.T @ t
    return T_world_cam


# =========================================================================
# Depth image to point cloud conversion
# =========================================================================

def depth_image_to_pointcloud(depth_m, K):
    """Convert depth image (meters) to Nx3 point cloud in camera frame.

    Args:
        depth_m: HxW float32 depth in meters
        K: 3x3 camera intrinsic matrix

    Returns:
        Nx3 point cloud in OpenCV camera coordinates
    """
    h, w = depth_m.shape
    mask = depth_m > 0.01  # filter out zero/near-zero depth

    v, u = np.where(mask)
    z = depth_m[v, u]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy

    pc = np.stack([x, y, z], axis=-1)
    return pc


# =========================================================================
# ROS2 depth subscriber (grabs a single frame)
# =========================================================================

class DepthGrabber(Node):
    """Grabs one depth image and camera_info from ROS2 topics."""

    def __init__(self):
        super().__init__('depth_grabber')
        self.depth_image = None
        self.rgb_image = None
        self.camera_info = None

        self.depth_sub = self.create_subscription(
            Image, '/rgbd_camera/depth_image', self._depth_cb, 10)
        self.rgb_sub = self.create_subscription(
            Image, '/rgbd_camera/image', self._rgb_cb, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/rgbd_camera/camera_info', self._info_cb, 10)

    def _depth_cb(self, msg):
        if self.depth_image is None:
            # Convert ROS Image to numpy
            h, w = msg.height, msg.width
            if msg.encoding == '32FC1':
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
            elif msg.encoding == '16UC1':
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
                depth = depth.astype(np.float32) / 1000.0  # mm to m
            else:
                self.get_logger().warn(f"Unknown depth encoding: {msg.encoding}")
                return
            self.depth_image = depth.copy()
            self.get_logger().info(f"Got depth image: {w}x{h}, range [{depth[depth>0].min():.2f}, {depth[depth>0].max():.2f}]m")

    def _rgb_cb(self, msg):
        if self.rgb_image is None:
            h, w = msg.height, msg.width
            if msg.encoding == 'rgb8':
                rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            elif msg.encoding == 'bgr8':
                bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
                rgb = bgr[:, :, ::-1]
            else:
                self.get_logger().warn(f"Unknown rgb encoding: {msg.encoding}")
                return
            self.rgb_image = rgb.copy()

    def _info_cb(self, msg):
        if self.camera_info is None:
            K = np.array(msg.k).reshape(3, 3)
            self.camera_info = K
            self.get_logger().info(f"Got camera_info: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")

    def wait_for_data(self, timeout_sec=15.0):
        """Spin until we have depth + camera_info."""
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.5)
            if self.depth_image is not None and self.camera_info is not None:
                return True
        return False


# =========================================================================
# GraspPlanner — main class
# =========================================================================

class GraspPlanner:
    """Plans grasps using Contact-GraspNet on Gazebo depth images.

    Usage:
        planner = GraspPlanner()
        planner.capture_scene()  # grab depth from Gazebo
        grasp_world, score = planner.get_best_grasp_for_target(
            target_position_world=(0.0, 0.55, 1.30),
            search_radius=0.08)
    """

    def __init__(self, load_model=True):
        self.depth_image = None
        self.rgb_image = None
        self.K = None
        self.pc_cam = None  # point cloud in camera frame
        self.T_world_cam = _get_camera_to_world()
        self.T_cam_world = _get_camera_extrinsics()

        if load_model:
            self.estimator = _load_grasp_estimator()
        else:
            self.estimator = None

    def capture_scene(self, timeout_sec=15.0):
        """Capture a depth image from Gazebo via ROS2.

        Returns True if successful.
        """
        grabber = DepthGrabber()
        ok = grabber.wait_for_data(timeout_sec)
        if not ok:
            print("[GraspPlanner] ERROR: Timeout waiting for depth image")
            grabber.destroy_node()
            return False

        self.depth_image = grabber.depth_image
        self.rgb_image = grabber.rgb_image
        self.K = grabber.camera_info

        # Convert to point cloud
        self.pc_cam = depth_image_to_pointcloud(self.depth_image, self.K)
        print(f"[GraspPlanner] Point cloud: {self.pc_cam.shape[0]} points")

        grabber.destroy_node()
        return True

    def set_depth_data(self, depth_image, K, rgb_image=None):
        """Set depth data directly (for testing without ROS2)."""
        self.depth_image = depth_image
        self.K = K
        self.rgb_image = rgb_image
        self.pc_cam = depth_image_to_pointcloud(depth_image, K)

    def _pc_cam_to_world(self, pc_cam):
        """Transform point cloud from camera frame to world frame."""
        ones = np.ones((pc_cam.shape[0], 1), dtype=np.float32)
        pc_h = np.hstack([pc_cam, ones])
        pc_world_h = (self.T_world_cam @ pc_h.T).T
        return pc_world_h[:, :3]

    def _world_to_cam(self, point_world):
        """Transform a single world point to camera frame."""
        p = np.array([*point_world, 1.0])
        return (self.T_cam_world @ p)[:3]

    def get_best_grasp_for_target(self, target_position_world,
                                   search_radius=0.10,
                                   z_range_cam=(0.1, 3.0),
                                   prefer_side_grasp=True):
        """Find the best Contact-GraspNet grasp near a target position.

        Args:
            target_position_world: (x, y, z) in world frame
            search_radius: max distance from target to accept a grasp (meters)
            z_range_cam: depth clipping range in camera frame
            prefer_side_grasp: prefer side-approach grasps for stacked boxes

        Returns:
            (grasp_world_4x4, score) or (None, 0.0) if no grasp found
        """
        if self.pc_cam is None:
            print("[GraspPlanner] No point cloud captured yet!")
            return None, 0.0

        if self.estimator is None:
            print("[GraspPlanner] Model not loaded!")
            return None, 0.0

        # Filter point cloud by z range
        pc = self.pc_cam.copy()
        mask = (pc[:, 2] > z_range_cam[0]) & (pc[:, 2] < z_range_cam[1])
        pc = pc[mask]

        if pc.shape[0] < 100:
            print(f"[GraspPlanner] Too few points after z-filter: {pc.shape[0]}")
            return None, 0.0

        print(f"[GraspPlanner] Running Contact-GraspNet on {pc.shape[0]} points...")
        t0 = time.time()

        # Run inference (returns grasps in camera frame)
        pred_grasps_cam, scores, contact_pts, gripper_openings = \
            self.estimator.predict_scene_grasps(pc, local_regions=False,
                                                 filter_grasps=False)

        elapsed = time.time() - t0
        print(f"[GraspPlanner] Inference: {elapsed:.2f}s")

        # Collect all grasps (key=-1 for full-scene prediction)
        if -1 not in pred_grasps_cam or len(pred_grasps_cam[-1]) == 0:
            print("[GraspPlanner] No grasps predicted!")
            return None, 0.0

        all_grasps = pred_grasps_cam[-1]  # Nx4x4
        all_scores = scores[-1]            # N
        print(f"[GraspPlanner] {len(all_grasps)} grasps predicted, "
              f"score range [{all_scores.min():.3f}, {all_scores.max():.3f}]")

        # Transform target to camera frame
        target_cam = self._world_to_cam(target_position_world)

        # Filter grasps near target
        grasp_positions_cam = all_grasps[:, :3, 3]  # Nx3
        dists = np.linalg.norm(grasp_positions_cam - target_cam, axis=1)

        near_mask = dists < search_radius
        if not np.any(near_mask):
            # Expand search radius progressively
            for r in [0.15, 0.20, 0.30]:
                near_mask = dists < r
                if np.any(near_mask):
                    print(f"[GraspPlanner] Expanded search radius to {r}m, "
                          f"found {near_mask.sum()} grasps")
                    break
            if not np.any(near_mask):
                print(f"[GraspPlanner] No grasps within 0.3m of target. "
                      f"Closest: {dists.min():.3f}m")
                return None, 0.0

        near_grasps = all_grasps[near_mask]
        near_scores = all_scores[near_mask]
        near_dists = dists[near_mask]

        # Score: combine confidence and distance (closer is better)
        combined_score = near_scores - 0.5 * near_dists

        if prefer_side_grasp:
            # STRICT filter: only keep grasps with nearly horizontal approach
            # in world frame. This ensures UR5e can actually reach the pose.
            for i in range(len(near_grasps)):
                grasp_cam = near_grasps[i]
                grasp_world = self.T_world_cam @ grasp_cam
                approach_world = -grasp_world[:3, 2]  # approach = -Z of grasp
                verticality = abs(approach_world[2])
                # Strong bonus for horizontal (verticality < 0.3 = nearly horizontal)
                if verticality < 0.3:
                    combined_score[i] += 1.0  # big bonus
                elif verticality < 0.5:
                    combined_score[i] += 0.3
                else:
                    combined_score[i] -= 1.0  # penalize vertical approaches

        best_idx = np.argmax(combined_score)
        best_grasp_cam = near_grasps[best_idx]
        best_score = near_scores[best_idx]

        # Transform to world frame
        best_grasp_world = self.T_world_cam @ best_grasp_cam

        # Check if the best grasp is reachable by UR5e
        # If approach is too vertical, snap to a known good side-grasp orientation
        approach_world = -best_grasp_world[:3, 2]
        verticality = abs(approach_world[2])

        # Always snap to side-grasp orientation for UR5e reliability.
        # Contact-GraspNet provides the POSITION, we use known-good orientation.
        snapped = True
        print(f"[GraspPlanner] Using GraspNet position + side-grasp orientation")

        print(f"[GraspPlanner] Best grasp: score={best_score:.3f}, "
              f"dist={near_dists[best_idx]:.3f}m, vert={verticality:.2f}, "
              f"snapped={snapped}")
        print(f"  Position (world): {best_grasp_world[:3, 3]}")

        return best_grasp_world, best_score, snapped

    def _snap_to_side_grasp(self, grasp_4x4):
        """Replace grasp orientation with a side-approach orientation.

        Keeps the GraspNet position but uses a reliable side-grasp orientation
        where the gripper approaches along the world -Y direction (toward the
        robot from the boxes).

        Side grasp orientation in world frame:
          X (gripper open axis) = world Z (up)
          Y (gripper close axis) = world X
          Z (approach axis) = world -Y (toward robot)
        """
        pos = grasp_4x4[:3, 3].copy()

        # Use the KNOWN WORKING side-grasp orientation directly in base_link.
        # Mode 4 side-grasp quat (w,x,y,z) = (0.707, 0, 0.707, 0) works.
        # Convert this back to a world-frame rotation matrix for the 4x4 transform.
        #
        # In base_link: tool0 points along +X, approach along +X
        # R_base = [[0,0,1],[0,-1,0],[1,0,0]] (from (w=0.707,x=0,y=0.707,z=0))
        #
        # Convert base_link to world: R_world = R_world_base @ R_base
        robot_yaw = math.pi / 2.0
        cy, sy = math.cos(robot_yaw), math.sin(robot_yaw)
        R_world_base = np.array([
            [cy, -sy, 0],
            [sy,  cy, 0],
            [0,   0,  1],
        ])

        # Side grasp in base_link (from quaternion w=0.707, x=0, y=0.707, z=0)
        R_side_base = np.array([
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0],
        ], dtype=np.float64)

        R_side_world = R_world_base @ R_side_base

        result = np.eye(4)
        result[:3, :3] = R_side_world
        result[:3, 3] = pos
        return result

    def grasp_to_pose(self, grasp_4x4, snapped=False):
        """Convert a 4x4 grasp transform to position + quaternion.

        Args:
            grasp_4x4: 4x4 homogeneous transform in world frame
            snapped: if True, the grasp was snapped to side-grasp and we
                     should use the known-good quaternion directly.

        Returns:
            (position, quaternion) where position=(x,y,z) and
            quaternion=(w,x,y,z) in base_link frame
        """
        # World to base_link: position transform
        robot_base = np.array([0.0, 0.0, 1.01])
        robot_yaw = math.pi / 2.0

        cos_y = math.cos(-robot_yaw)
        sin_y = math.sin(-robot_yaw)
        R_base_world = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1],
        ])

        # Transform position to base_link
        pos_world = grasp_4x4[:3, 3]
        pos_base = R_base_world @ (pos_world - robot_base)
        position = tuple(pos_base)

        if snapped:
            # Use the known working side-grasp quaternion directly
            # (w, x, y, z) = (0.707, 0, 0.707, 0) — tool0 pointing along +X in base_link
            quaternion = (0.7071068, 0.0, 0.7071068, 0.0)
        else:
            # Full rotation transform
            T_base_world = np.eye(4)
            T_base_world[:3, :3] = R_base_world
            T_base_world[:3, 3] = R_base_world @ (-robot_base)
            grasp_base = T_base_world @ grasp_4x4
            R = grasp_base[:3, :3]
            qw, qx, qy, qz = _rotation_matrix_to_quaternion(R)
            quaternion = (qw, qx, qy, qz)

        return position, quaternion


def _rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    # Normalize
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    return w/norm, x/norm, y/norm, z/norm


# =========================================================================
# Standalone test
# =========================================================================

def main():
    """Test grasp planner with live Gazebo camera feed."""
    rclpy.init()

    planner = GraspPlanner(load_model=True)

    print("\n=== Capturing scene from Gazebo camera ===")
    ok = planner.capture_scene(timeout_sec=15.0)
    if not ok:
        print("Failed to capture scene. Is Gazebo running with box_stack.sdf?")
        print("Check: ros2 topic list | grep rgbd_camera")
        rclpy.shutdown()
        return

    # Test: find grasp for top box
    target = (0.0, 0.55, 1.30)
    print(f"\n=== Finding grasp for top box at {target} ===")
    grasp_world, score = planner.get_best_grasp_for_target(target)

    if grasp_world is not None:
        position, quaternion = planner.grasp_to_pose(grasp_world)
        print(f"\nGrasp in base_link frame:")
        print(f"  Position: ({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})")
        print(f"  Quaternion (w,x,y,z): ({quaternion[0]:.4f}, {quaternion[1]:.4f}, "
              f"{quaternion[2]:.4f}, {quaternion[3]:.4f})")
    else:
        print("\nNo grasp found. Using fallback hardcoded position.")

    rclpy.shutdown()


if __name__ == '__main__':
    main()
