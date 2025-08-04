"""Cartesian impedance controller.

This module implements a simple Cartesian impedance controller that can
command the end–effector of a manipulator to follow a target pose.  The
controller computes a desired wrench based on proportional (stiffness)
and derivative (damping) gains on position and velocity errors and
optionally adds gravity compensation.  It does not handle orientation
control explicitly; this can be added by extending the state vectors
and gain matrices to 6D space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class CartesianImpedanceController:
    """Simple impedance controller acting in Cartesian space.

    Args:
        kp: Proportional gains (stiffness) for each Cartesian axis.
        kd: Derivative gains (damping) for each Cartesian axis.
        target_pos: Target Cartesian position.
        target_vel: Target Cartesian velocity (default zeros).
    """
    kp: np.ndarray
    kd: np.ndarray
    target_pos: np.ndarray
    target_vel: np.ndarray

    def __init__(
        self,
        kp: Iterable[float],
        kd: Iterable[float],
        target_pos: Iterable[float],
        target_vel: Optional[Iterable[float]] = None,
    ) -> None:
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.target_pos = np.asarray(target_pos, dtype=float)
        if target_vel is None:
            self.target_vel = np.zeros_like(self.kp)
        else:
            self.target_vel = np.asarray(target_vel, dtype=float)
        assert self.kp.shape == self.target_pos.shape
        assert self.kd.shape == self.target_pos.shape
        assert self.target_vel.shape == self.target_pos.shape

    def compute(
        self,
        current_pos: Iterable[float],
        current_vel: Iterable[float],
        gravity_torque: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        """Compute the desired Cartesian wrench/torque.

        Args:
            current_pos: Current end–effector position.
            current_vel: Current end–effector velocity.
            gravity_torque: Optional feed‑forward term for gravity compensation
                (in joint space).  If provided, it will be added to the
                returned torque vector.

        Returns:
            Control torques.  If gravity_torque is provided the return
            dimension matches that array; otherwise a wrench in Cartesian
            space is returned (to be mapped to joint torques via a Jacobian).
        """
        pos_err = self.target_pos - np.asarray(current_pos, dtype=float)
        vel_err = self.target_vel - np.asarray(current_vel, dtype=float)
        # desired Cartesian wrench
        wrench = self.kp * pos_err + self.kd * vel_err
        if gravity_torque is not None:
            # If gravity compensation is provided, assume the wrench has been
            # mapped to joint space already and simply add it.
            wrench = np.asarray(wrench, dtype=float)
            torque = np.asarray(gravity_torque, dtype=float) + wrench
            return torque
        return wrench