#!/usr/bin/env python3
"""
Physics-Aware Safety Guard for LLM-Controlled Manipulation

Intercepts LLM manipulation commands and simulates the physical
consequences BEFORE execution. If the action would cause collapse,
toppling, or other dangerous outcomes, the command is REJECTED.

Architecture:
    User command → LLM (task plan) → [Physics Safety Guard] → Robot execution
                                            ↓
                                     PyBullet simulation
                                     Collapse detection
                                     ALLOW / REJECT

Demo: 3-box stack scenarios
    - "위에 상자 빼" → safe → ALLOW
    - "아래 상자 빼" → collapse → REJECT
    - "가운데 상자 빼" → partial collapse → REJECT
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


# =============================================================================
# Data structures
# =============================================================================

class SafetyDecision(Enum):
    ALLOW = "allow"
    REJECT = "reject"


@dataclass
class SceneObject:
    """An object in the scene with physical properties."""
    name: str
    position: Tuple[float, float, float]  # (x, y, z)
    size: Tuple[float, float, float]      # half-extents (x, y, z)
    mass: float = 1.0                      # kg
    color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    body_id: Optional[int] = None


@dataclass
class SafetyCheckResult:
    """Result of a physics safety check."""
    decision: SafetyDecision
    action: str
    target_object: str
    reason: str
    max_displacement: float = 0.0      # max displacement of remaining objects (m)
    fallen_objects: List[str] = field(default_factory=list)
    simulation_steps: int = 0
    simulation_time_sec: float = 0.0


# =============================================================================
# Physics Safety Guard
# =============================================================================

class PhysicsSafetyGuard:
    """
    Simulates the physical consequences of removing an object
    before allowing the robot to execute the action.
    """

    # Thresholds
    COLLAPSE_DISPLACEMENT_THRESHOLD = 0.02  # 2cm — object moved significantly
    FALL_HEIGHT_THRESHOLD = 0.05            # 5cm drop = fallen
    SIMULATION_STEPS = 500                   # ~2 seconds at 240Hz
    GRAVITY = -9.81

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.physics_client = None
        self.objects: Dict[str, SceneObject] = {}
        self.ground_id = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"[PhysicsGuard] {msg}")

    # -----------------------------------------------------------------
    # Scene setup
    # -----------------------------------------------------------------

    def _init_physics(self):
        """Initialize a fresh PyBullet physics world."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        self.physics_client = p.connect(p.DIRECT)  # headless
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.physics_client)
        p.setGravity(0, 0, self.GRAVITY, physicsClientId=self.physics_client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.physics_client)

        # Ground plane
        self.ground_id = p.loadURDF("plane.urdf",
                                     physicsClientId=self.physics_client)

    def _create_box(self, obj: SceneObject) -> int:
        """Create a box in the physics world."""
        col_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=list(obj.size),
            physicsClientId=self.physics_client)
        vis_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=list(obj.size),
            rgbaColor=list(obj.color),
            physicsClientId=self.physics_client)
        body_id = p.createMultiBody(
            baseMass=obj.mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=list(obj.position),
            physicsClientId=self.physics_client)
        return body_id

    def build_scene(self, objects: List[SceneObject]):
        """Build the physics scene with given objects."""
        self._init_physics()
        self.objects = {}
        for obj in objects:
            obj.body_id = self._create_box(obj)
            self.objects[obj.name] = obj
            self._log(f"Created '{obj.name}' at {obj.position}, "
                      f"size={obj.size}, mass={obj.mass}kg")

        # Let objects settle
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physics_client)

    def _get_positions(self) -> Dict[str, np.ndarray]:
        """Get current positions of all objects."""
        positions = {}
        for name, obj in self.objects.items():
            if obj.body_id is not None:
                pos, _ = p.getBasePositionAndOrientation(
                    obj.body_id, physicsClientId=self.physics_client)
                positions[name] = np.array(pos)
        return positions

    # -----------------------------------------------------------------
    # Safety check
    # -----------------------------------------------------------------

    def check_removal_safety(self, target_name: str) -> SafetyCheckResult:
        """
        Simulate removing an object and check if it causes collapse.

        1. Record positions of all objects
        2. Remove the target object (teleport far away)
        3. Run physics simulation
        4. Check if remaining objects moved/fell
        """
        if target_name not in self.objects:
            return SafetyCheckResult(
                decision=SafetyDecision.REJECT,
                action="remove",
                target_object=target_name,
                reason=f"Object '{target_name}' not found in scene")

        t_start = time.time()

        # Record initial positions (after settling)
        initial_positions = self._get_positions()

        # Remove target: teleport it far away and make it static
        target = self.objects[target_name]
        p.resetBasePositionAndOrientation(
            target.body_id, [0, 0, -100], [0, 0, 0, 1],
            physicsClientId=self.physics_client)
        p.changeDynamics(target.body_id, -1, mass=0,
                         physicsClientId=self.physics_client)

        self._log(f"Removed '{target_name}', running simulation...")

        # Run physics simulation
        for _ in range(self.SIMULATION_STEPS):
            p.stepSimulation(physicsClientId=self.physics_client)

        # Check results
        final_positions = self._get_positions()
        t_elapsed = time.time() - t_start

        max_displacement = 0.0
        fallen_objects = []

        for name in self.objects:
            if name == target_name:
                continue
            if name not in initial_positions or name not in final_positions:
                continue

            displacement = np.linalg.norm(
                final_positions[name] - initial_positions[name])
            height_drop = initial_positions[name][2] - final_positions[name][2]

            self._log(f"  '{name}': displacement={displacement:.4f}m, "
                      f"height_drop={height_drop:.4f}m")

            max_displacement = max(max_displacement, displacement)

            if height_drop > self.FALL_HEIGHT_THRESHOLD:
                fallen_objects.append(name)

        # Decision
        is_dangerous = (max_displacement > self.COLLAPSE_DISPLACEMENT_THRESHOLD
                        or len(fallen_objects) > 0)

        if is_dangerous:
            if fallen_objects:
                reason = (f"Removing '{target_name}' causes "
                          f"{', '.join(fallen_objects)} to fall "
                          f"(max displacement: {max_displacement:.3f}m)")
            else:
                reason = (f"Removing '{target_name}' causes instability "
                          f"(max displacement: {max_displacement:.3f}m)")
            decision = SafetyDecision.REJECT
        else:
            reason = (f"Removing '{target_name}' is safe "
                      f"(max displacement: {max_displacement:.4f}m)")
            decision = SafetyDecision.ALLOW

        return SafetyCheckResult(
            decision=decision,
            action="remove",
            target_object=target_name,
            reason=reason,
            max_displacement=max_displacement,
            fallen_objects=fallen_objects,
            simulation_steps=self.SIMULATION_STEPS,
            simulation_time_sec=t_elapsed)

    def cleanup(self):
        """Disconnect physics engine."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


# =============================================================================
# LLM Command Parser (simplified)
# =============================================================================

class ManipulationCommandParser:
    """
    Parse natural language manipulation commands into actions.
    In production, this would be an LLM call.
    """

    REMOVE_KEYWORDS = ["빼", "제거", "치워", "꺼내", "remove", "take out",
                       "pull out", "extract"]
    POSITION_MAP = {
        "위": "top", "맨 위": "top", "상단": "top", "top": "top",
        "가운데": "middle", "중간": "middle", "middle": "middle",
        "아래": "bottom", "맨 아래": "bottom", "하단": "bottom", "bottom": "bottom",
    }

    def parse(self, command: str, object_names: List[str]) -> Optional[Dict]:
        """Parse a command into (action, target)."""
        command_lower = command.lower().strip()

        # Check if it's a removal command
        is_remove = any(kw in command_lower for kw in self.REMOVE_KEYWORDS)
        if not is_remove:
            return None

        # Find target by position keyword
        for keyword, position in self.POSITION_MAP.items():
            if keyword in command_lower:
                # Map position to object name
                for name in object_names:
                    if position in name.lower():
                        return {"action": "remove", "target": name}

        # Find target by direct name
        for name in object_names:
            if name.lower() in command_lower:
                return {"action": "remove", "target": name}

        return {"action": "remove", "target": None}


# =============================================================================
# Demo scenarios
# =============================================================================

def create_3stack_scene() -> List[SceneObject]:
    """Create a 3-box vertical stack."""
    box_size = (0.15, 0.15, 0.10)  # 30cm × 30cm × 20cm boxes
    return [
        SceneObject(
            name="bottom_box",
            position=(0.5, 0.0, 0.10),  # on ground
            size=box_size, mass=2.0,
            color=(0.8, 0.2, 0.2, 1.0)),  # red
        SceneObject(
            name="middle_box",
            position=(0.5, 0.0, 0.30),  # on top of bottom
            size=box_size, mass=1.5,
            color=(0.2, 0.8, 0.2, 1.0)),  # green
        SceneObject(
            name="top_box",
            position=(0.5, 0.0, 0.50),  # on top of middle
            size=box_size, mass=1.0,
            color=(0.2, 0.2, 0.8, 1.0)),  # blue
    ]


def create_pyramid_scene() -> List[SceneObject]:
    """Create a 2-1 pyramid (2 on bottom, 1 on top)."""
    box_size = (0.15, 0.15, 0.10)
    return [
        SceneObject(
            name="bottom_left",
            position=(0.35, 0.0, 0.10),
            size=box_size, mass=2.0,
            color=(0.8, 0.2, 0.2, 1.0)),
        SceneObject(
            name="bottom_right",
            position=(0.65, 0.0, 0.10),
            size=box_size, mass=2.0,
            color=(0.2, 0.8, 0.2, 1.0)),
        SceneObject(
            name="top_box",
            position=(0.5, 0.0, 0.30),
            size=box_size, mass=1.0,
            color=(0.2, 0.2, 0.8, 1.0)),
    ]


def create_l_shape_scene() -> List[SceneObject]:
    """Create an L-shaped arrangement (2 stacked + 1 adjacent)."""
    box_size = (0.15, 0.15, 0.10)
    return [
        SceneObject(
            name="base_box",
            position=(0.5, 0.0, 0.10),
            size=box_size, mass=2.0,
            color=(0.8, 0.2, 0.2, 1.0)),
        SceneObject(
            name="stacked_box",
            position=(0.5, 0.0, 0.30),
            size=box_size, mass=1.0,
            color=(0.2, 0.8, 0.2, 1.0)),
        SceneObject(
            name="adjacent_box",
            position=(0.8, 0.0, 0.10),  # next to base, on ground
            size=box_size, mass=1.5,
            color=(0.2, 0.2, 0.8, 1.0)),
    ]


# =============================================================================
# Main demo
# =============================================================================

def run_demo():
    """Run the physics safety guard demo."""
    guard = PhysicsSafetyGuard(verbose=True)
    parser = ManipulationCommandParser()

    print("=" * 70)
    print("Physics-Aware Safety Guard for LLM Manipulation")
    print("=" * 70)

    # --- Scenario 1: 3-box vertical stack ---
    print("\n" + "=" * 70)
    print("Scenario 1: 3-Box Vertical Stack")
    print("  [bottom_box] → [middle_box] → [top_box]")
    print("=" * 70)

    commands_1 = [
        "위에 상자 빼줘",          # top → safe
        "가운데 상자 빼줘",        # middle → dangerous
        "아래 상자 빼줘",          # bottom → dangerous
    ]

    for cmd in commands_1:
        scene = create_3stack_scene()
        guard.build_scene(scene)
        names = [o.name for o in scene]
        parsed = parser.parse(cmd, names)

        print(f"\n  Command: \"{cmd}\"")
        print(f"  Parsed:  action={parsed['action']}, target={parsed['target']}")

        if parsed and parsed['target']:
            result = guard.check_removal_safety(parsed['target'])
            icon = "✓ ALLOW" if result.decision == SafetyDecision.ALLOW else "✗ REJECT"
            print(f"  Result:  [{icon}]")
            print(f"  Reason:  {result.reason}")
            if result.fallen_objects:
                print(f"  Fallen:  {result.fallen_objects}")
            print(f"  Sim time: {result.simulation_time_sec:.3f}s")

    # --- Scenario 2: Pyramid (2 base + 1 top) ---
    print("\n" + "=" * 70)
    print("Scenario 2: Pyramid (2 bottom + 1 top)")
    print("  [bottom_left] [bottom_right]")
    print("         [top_box]")
    print("=" * 70)

    commands_2 = [
        "top_box remove",            # top → safe
        "bottom_left 빼줘",          # base → top falls
    ]

    for cmd in commands_2:
        scene = create_pyramid_scene()
        guard.build_scene(scene)
        names = [o.name for o in scene]
        parsed = parser.parse(cmd, names)

        print(f"\n  Command: \"{cmd}\"")
        print(f"  Parsed:  action={parsed['action']}, target={parsed['target']}")

        if parsed and parsed['target']:
            result = guard.check_removal_safety(parsed['target'])
            icon = "✓ ALLOW" if result.decision == SafetyDecision.ALLOW else "✗ REJECT"
            print(f"  Result:  [{icon}]")
            print(f"  Reason:  {result.reason}")
            if result.fallen_objects:
                print(f"  Fallen:  {result.fallen_objects}")

    # --- Scenario 3: L-shape (safe adjacent removal) ---
    print("\n" + "=" * 70)
    print("Scenario 3: L-Shape (stacked + adjacent)")
    print("  [base_box]→[stacked_box]  [adjacent_box]")
    print("=" * 70)

    commands_3 = [
        "adjacent_box 제거해",       # adjacent → safe (independent)
        "base_box 빼줘",             # base → stacked falls
    ]

    for cmd in commands_3:
        scene = create_l_shape_scene()
        guard.build_scene(scene)
        names = [o.name for o in scene]
        parsed = parser.parse(cmd, names)

        print(f"\n  Command: \"{cmd}\"")
        print(f"  Parsed:  action={parsed['action']}, target={parsed['target']}")

        if parsed and parsed['target']:
            result = guard.check_removal_safety(parsed['target'])
            icon = "✓ ALLOW" if result.decision == SafetyDecision.ALLOW else "✗ REJECT"
            print(f"  Result:  [{icon}]")
            print(f"  Reason:  {result.reason}")
            if result.fallen_objects:
                print(f"  Fallen:  {result.fallen_objects}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Summary: Physics Guard catches dangerous removals that")
    print("an LLM would approve (valid command, dangerous consequence)")
    print("=" * 70)

    guard.cleanup()


if __name__ == "__main__":
    run_demo()
