"""Five-bar leg kinematics read directly from the packaged loop-cut URDF.

Pure NumPy, no Isaac Lab import, so it can run in unit tests and in the G3
scripts.  It is used to

* give the passive knee joints closure-consistent initial angles for a hip
  stance (``ArticulationCfg.init_state``), and
* express hip joint angles as the crank angles used in the servo derivation
  (docs/ACTION_RANGE_THROUGHPUT_FIX_PLAN.md section 3A).

Crank angle convention (section 3A): in the leg's sagittal plane, with the
horizontal axis pointing away from the body centre along the robot's forward
axis for front legs (backward for rear legs) and the vertical axis pointing
down, a crank angle of 90 deg points straight down.  At the URDF zero pose the
outer crank is at about 154 deg and the inner crank at about 6.5 deg.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


LEGS = ("FR", "FL", "RL", "RR")
DEFAULT_URDF = (
    Path(__file__).resolve().parents[5]
    / "assets/robots/stackforce_quadrupedal_wheeled_robot/urdf/stackforce_quadrupedal_wheeled_robot.urdf"
)


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = map(float, (np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)))
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


class LegKinematics:
    """Forward kinematics for the outer and inner chains of the four legs."""

    def __init__(self, urdf_path: Path | str = DEFAULT_URDF) -> None:
        root = ET.parse(str(urdf_path)).getroot()
        self.joints: dict[str, _Joint] = {}
        for element in root.findall("joint"):
            origin = element.find("origin")
            axis = element.find("axis")
            self.joints[element.get("name")] = _Joint(
                name=element.get("name"),
                parent=element.find("parent").get("link"),
                child=element.find("child").get("link"),
                xyz=np.array([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]),
                rpy=np.array([float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]),
                axis=np.array([float(v) for v in (axis.get("xyz") if axis is not None else "1 0 0").split()]),
            )

    # ------------------------------------------------------------------ FK
    def _chain(self, names: tuple[str, ...], angles: tuple[float, ...]) -> tuple[list[np.ndarray], np.ndarray]:
        """Return joint origins in base coordinates and the final frame origin."""
        rotation = np.eye(3)
        position = np.zeros(3)
        origins = []
        for index, name in enumerate(names):
            joint = self.joints[name]
            position = position + rotation @ joint.xyz
            rotation = rotation @ _rpy_matrix(joint.rpy)
            origins.append(position.copy())
            if index < len(angles):
                rotation = rotation @ _axis_angle(joint.axis, angles[index])
        return origins, position

    def outer_chain(self, leg: str, hip: float, knee: float) -> tuple[list[np.ndarray], np.ndarray]:
        names = (f"{leg}_Outer_Hip_Joint", f"{leg}_Outer_Knee_Joint", f"{leg}_Wheel_Joint", f"{leg}_Wheel_Frame_Joint")
        return self._chain(names, (hip, knee, 0.0))

    def inner_chain(self, leg: str, hip: float, knee: float) -> tuple[list[np.ndarray], np.ndarray]:
        names = (f"{leg}_Inner_Hip_Joint", f"{leg}_Inner_Knee_Joint", f"{leg}_Closure_Frame_Joint")
        return self._chain(names, (hip, knee))

    def closure_error(self, leg: str, q_outer: float, q_inner: float, k_outer: float, k_inner: float) -> np.ndarray:
        """Wheel-frame minus closure-frame position in the sagittal (Y, Z) plane."""
        _, wheel = self.outer_chain(leg, q_outer, k_outer)
        _, closure = self.inner_chain(leg, q_inner, k_inner)
        return (wheel - closure)[1:]

    # ------------------------------------------------------- loop closure
    def solve_knees(
        self, leg: str, q_outer: float, q_inner: float, *, steps: int = 32, tolerance: float = 1e-7
    ) -> tuple[float, float, float]:
        """Return (outer_knee, inner_knee, residual_m) by continuation from the zero pose."""
        knees = np.zeros(2)
        for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
            qo, qi = fraction * q_outer, fraction * q_inner
            for _ in range(20):
                error = self.closure_error(leg, qo, qi, *knees)
                jacobian = np.zeros((2, 2))
                for column in range(2):
                    delta = np.zeros(2)
                    delta[column] = 1e-7
                    jacobian[:, column] = (self.closure_error(leg, qo, qi, *(knees + delta)) - error) / 1e-7
                update = np.linalg.solve(jacobian, error)
                knees = knees - update
                if float(np.abs(update).max()) < tolerance:
                    break
        residual = float(np.linalg.norm(self.closure_error(leg, q_outer, q_inner, *knees)))
        return float(knees[0]), float(knees[1]), residual

    # ------------------------------------------------------- crank angles
    def _sagittal_sign(self, leg: str) -> float:
        return 1.0 if self.joints[f"{leg}_Outer_Hip_Joint"].xyz[1] > 0 else -1.0

    def crank_angles_deg(self, leg: str, q_outer: float, q_inner: float) -> tuple[float, float]:
        """Return (alpha, beta) crank angles in degrees for the given hip angles."""
        sign = self._sagittal_sign(leg)
        angles = []
        for chain, q in ((self.outer_chain, q_outer), (self.inner_chain, q_inner)):
            origins, _ = chain(leg, q, 0.0)
            delta = origins[1] - origins[0]
            angles.append(math.degrees(math.atan2(-delta[2], sign * delta[1])))
        return angles[0], angles[1]

    def foot_depth_below_hips_m(self, leg: str, q_outer: float, q_inner: float) -> float:
        """Vertical distance from the mean hip-axis height down to the wheel centre."""
        k_outer, k_inner, _ = self.solve_knees(leg, q_outer, q_inner)
        outer, wheel = self.outer_chain(leg, q_outer, k_outer)
        inner, _ = self.inner_chain(leg, q_inner, k_inner)
        hip_z = 0.5 * (outer[0][2] + inner[0][2])
        return float(hip_z - wheel[2])

    def wheel_centre_z_m(self, leg: str, q_outer: float, q_inner: float) -> float:
        """Wheel-centre height in the base frame for the given hip angles."""
        k_outer, _, _ = self.solve_knees(leg, q_outer, q_inner)
        _, wheel = self.outer_chain(leg, q_outer, k_outer)
        return float(wheel[2])

    def hips_for_crank_angles(self, leg: str, alpha_deg: float, beta_deg: float) -> tuple[float, float]:
        """Invert ``crank_angles_deg`` (crank angles are affine in the hip angles)."""
        a0, b0 = self.crank_angles_deg(leg, 0.0, 0.0)
        s_outer, s_inner = self.hip_signs(leg)
        return s_outer * math.radians(alpha_deg - a0), s_inner * math.radians(beta_deg - b0)

    def hip_signs(self, leg: str) -> tuple[int, int]:
        """Return (d alpha / dq, d beta / dq) signs; crank angles move one-for-one with the hip angle."""
        a0, b0 = self.crank_angles_deg(leg, 0.0, 0.0)
        a1, _ = self.crank_angles_deg(leg, 0.1, 0.0)
        _, b1 = self.crank_angles_deg(leg, 0.0, 0.1)
        return (1 if a1 > a0 else -1), (1 if b1 > b0 else -1)


__all__ = ["DEFAULT_URDF", "LEGS", "LegKinematics"]
