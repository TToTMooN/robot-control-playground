"""Wrapper for Pinocchio kinematics and dynamics on the FR3 robot.

This module defines a `PinocchioModel` class that loads a URDF model via
Pinocchio's `RobotWrapper` and exposes convenience methods to compute
forward and inverse dynamics.  It can be used in conjunction with the
MuJoCo wrapper to cross‑check dynamics or to perform gravity
compensation for controllers.

Requirements:
    - pinocchio (>=3.1)

Note:
    The URDF file for the FR3 robot must exist at the path supplied
    during initialisation.  The URDF itself is not provided in this
    repository; please download it from the manufacturer or your
    robotic middleware.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Optional

import numpy as np

try:
    import pinocchio as pin
    from pinocchio.robot_wrapper import RobotWrapper
except ImportError as e:  # pragma: no cover - optional dependency
    raise ImportError(
        "PinocchioModel requires the pinocchio package. Install pinocchio>=3.1 to use this class.") from e


@dataclasses.dataclass
class PinocchioModel:
    """Thin wrapper around Pinocchio for FR3 kinematics and dynamics."""
    urdf_path: str
    mesh_dir: Optional[str] = None
    robot: RobotWrapper = dataclasses.field(init=False)
    model: pin.Model = dataclasses.field(init=False)
    data: pin.Data = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.robot = RobotWrapper.BuildFromURDF(self.urdf_path, self.mesh_dir)
        self.model = self.robot.model
        self.data = self.robot.data

    def inverse_dynamics(self, q: Iterable[float], v: Iterable[float], a: Iterable[float]) -> np.ndarray:
        """Compute joint torques using the Recursive Newton–Euler Algorithm (RNEA).

        Args:
            q: Joint positions of length `model.nq`.
            v: Joint velocities of length `model.nv`.
            a: Joint accelerations of length `model.nv`.

        Returns:
            A numpy array of joint torques.
        """
        q_arr = np.asarray(q, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        a_arr = np.asarray(a, dtype=float)
        assert q_arr.shape[0] == self.model.nq
        assert v_arr.shape[0] == self.model.nv
        assert a_arr.shape[0] == self.model.nv
        tau = pin.rnea(self.model, self.data, q_arr, v_arr, a_arr)
        return tau

    def mass_matrix(self, q: Iterable[float]) -> np.ndarray:
        """Compute the joint‑space inertia matrix using CRBA."""
        q_arr = np.asarray(q, dtype=float)
        assert q_arr.shape[0] == self.model.nq
        M = pin.crba(self.model, self.data, q_arr)
        return M

    def jacobian(self, q: Iterable[float], frame_name: str) -> np.ndarray:
        """Compute the spatial Jacobian of a frame expressed in the world frame.

        Args:
            q: Joint positions of length `model.nq`.
            frame_name: Name of the frame whose Jacobian is desired.

        Returns:
            A (6 × n) Jacobian matrix.
        """
        q_arr = np.asarray(q, dtype=float)
        assert q_arr.shape[0] == self.model.nq
        frame_id = self.model.getFrameId(frame_name)
        if frame_id < 0:
            raise ValueError(f"Frame '{frame_name}' not found in URDF")
        pin.computeJointJacobians(self.model, self.data, q_arr)
        pin.updateFramePlacement(self.model, self.data, frame_id)
        J6 = pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.WORLD)
        return J6