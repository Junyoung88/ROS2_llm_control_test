#!/usr/bin/env python3
"""
Physics Safety Guard — GUI Visualization

PyBullet GUI로 상자 제거 전/후를 시각적으로 확인.
각 시나리오마다 3초 대기 → 제거 → 시뮬레이션 → 결과 표시.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
from physics_guard import (PhysicsSafetyGuard, SceneObject, SafetyDecision,
                           create_3stack_scene, create_pyramid_scene,
                           create_l_shape_scene)


class PhysicsSafetyGuardGUI(PhysicsSafetyGuard):
    """GUI version of the physics guard."""

    def _init_physics(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.physics_client)
        p.setGravity(0, 0, self.GRAVITY, physicsClientId=self.physics_client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.physics_client)

        # Camera setup
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.5, 0.0, 0.3],
            physicsClientId=self.physics_client)

        # Disable extra GUI panels
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0,
                                   physicsClientId=self.physics_client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1,
                                   physicsClientId=self.physics_client)

        self.ground_id = p.loadURDF("plane.urdf",
                                     physicsClientId=self.physics_client)

    def _add_label(self, obj: SceneObject):
        """Add text label above object."""
        label_pos = list(obj.position)
        label_pos[2] += obj.size[2] + 0.05
        p.addUserDebugText(
            obj.name, label_pos,
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            physicsClientId=self.physics_client)

    def build_scene(self, objects):
        super().build_scene(objects)
        for obj in objects:
            self._add_label(obj)

    def check_removal_safety_visual(self, target_name: str, pause_before=2.0,
                                     sim_speed=1.0):
        """Visual version: shows before/after with pauses."""
        if target_name not in self.objects:
            print(f"Object '{target_name}' not found")
            return

        # Show "before" state
        p.addUserDebugText(
            f"Command: Remove '{target_name}'",
            [0.5, 0.0, 0.85],
            textColorRGB=[1, 1, 0],
            textSize=1.5,
            lifeTime=pause_before + 3,
            physicsClientId=self.physics_client)
        time.sleep(pause_before)

        # Record initial positions
        initial_positions = self._get_positions()

        # Remove target visually
        target = self.objects[target_name]
        p.addUserDebugText(
            f"Removing '{target_name}'...",
            [0.5, 0.0, 0.75],
            textColorRGB=[1, 0.5, 0],
            textSize=1.3,
            lifeTime=3,
            physicsClientId=self.physics_client)

        # Teleport away
        p.resetBasePositionAndOrientation(
            target.body_id, [0, 0, -100], [0, 0, 0, 1],
            physicsClientId=self.physics_client)
        p.changeDynamics(target.body_id, -1, mass=0,
                         physicsClientId=self.physics_client)

        time.sleep(0.5)

        # Run simulation visually (real-time)
        steps_per_frame = max(1, int(4 / sim_speed))
        for i in range(self.SIMULATION_STEPS):
            p.stepSimulation(physicsClientId=self.physics_client)
            if i % steps_per_frame == 0:
                time.sleep(1.0 / 60.0)  # ~60fps display

        # Check results
        final_positions = self._get_positions()
        fallen = []
        max_disp = 0.0
        for name in self.objects:
            if name == target_name:
                continue
            if name in initial_positions and name in final_positions:
                disp = np.linalg.norm(final_positions[name] - initial_positions[name])
                drop = initial_positions[name][2] - final_positions[name][2]
                max_disp = max(max_disp, disp)
                if drop > self.FALL_HEIGHT_THRESHOLD:
                    fallen.append(name)

        is_dangerous = max_disp > self.COLLAPSE_DISPLACEMENT_THRESHOLD or len(fallen) > 0

        # Show result
        if is_dangerous:
            result_text = f"REJECT — {', '.join(fallen)} fell!"
            color = [1, 0, 0]
        else:
            result_text = "ALLOW — Safe removal"
            color = [0, 1, 0]

        p.addUserDebugText(
            result_text,
            [0.5, 0.0, 0.85],
            textColorRGB=color,
            textSize=1.8,
            lifeTime=3,
            physicsClientId=self.physics_client)

        print(f"  [{('REJECT' if is_dangerous else 'ALLOW')}] "
              f"Remove '{target_name}' → "
              f"{'fallen: ' + str(fallen) if fallen else 'safe'}")

        time.sleep(2.0)


def run_gui_demo():
    guard = PhysicsSafetyGuardGUI(verbose=False)

    scenarios = [
        ("3-Box Stack: Remove TOP (safe)",
         create_3stack_scene, "top_box"),
        ("3-Box Stack: Remove MIDDLE (dangerous)",
         create_3stack_scene, "middle_box"),
        ("3-Box Stack: Remove BOTTOM (dangerous)",
         create_3stack_scene, "bottom_box"),
        ("Pyramid: Remove TOP (safe)",
         create_pyramid_scene, "top_box"),
        ("Pyramid: Remove BOTTOM_LEFT (dangerous)",
         create_pyramid_scene, "bottom_left"),
        ("L-Shape: Remove ADJACENT (safe)",
         create_l_shape_scene, "adjacent_box"),
        ("L-Shape: Remove BASE (dangerous)",
         create_l_shape_scene, "base_box"),
    ]

    print("=" * 60)
    print("Physics Safety Guard — GUI Demo")
    print("Watch the PyBullet window for visual simulation")
    print("=" * 60)

    for i, (title, scene_fn, target) in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}] {title}")
        scene = scene_fn()
        guard.build_scene(scene)
        guard.check_removal_safety_visual(target, pause_before=2.0)

    print("\nDemo complete. Closing in 3 seconds...")
    time.sleep(3)
    guard.cleanup()


if __name__ == "__main__":
    run_gui_demo()
