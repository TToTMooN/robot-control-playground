"""Wrapper around MuJoCo simulation for the FR3 robot and simple systems.

This module defines a `MujocoSim` class that loads a MuJoCo model from an
MJCF file and exposes convenience methods for stepping the simulation,
computing inverse dynamics and updating physical parameters.  It is
designed to hide MuJoCo's internal data structures and provide a clean
Pythonic API.  The wrapper is intentionally minimal; advanced features
can be added as needed.

Requirements:
    - mujoco (>=2.3)

Example:
    >>> sim = MujocoSim("models/fr3.xml")
    >>> sim.reset()
    >>> q0, v0 = sim.data.qpos.copy(), sim.data.qvel.copy()
    >>> tau = sim.inverse_dynamics(q0, v0, np.zeros_like(v0))
    >>> sim.step(tau)
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Optional

import numpy as np

try:
    import mujoco
except ImportError as e:  # pragma: no cover - optional dependency
    raise ImportError(
        "MujocoSim requires the mujoco package. Install mujoco>=2.3 to use this class.") from e


@dataclasses.dataclass
class MujocoSim:
    """A thin wrapper around MuJoCo for simulation and dynamics queries.

    Attributes:
        model_path: Path to the MJCF (XML) file describing the system.  For
            the FR3 robot this should point to an MJCF model exported from
            the URDF or provided by the user.
        model: The underlying `mujoco.MjModel` instance.
        data: The underlying `mujoco.MjData` instance.

    """
    model_path: str
    model: mujoco.MjModel = dataclasses.field(init=False)
    data: mujoco.MjData = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

    # -- Simulation control -------------------------------------------------
    def reset(self, qpos: Optional[Iterable[float]] = None, qvel: Optional[Iterable[float]] = None) -> None:
        """Reset the simulator state.

        If `qpos` or `qvel` are provided, the corresponding state vectors are
        set after resetting the data structure.  After modifying positions
        directly the forward kinematics are recomputed using `mj_forward`.

        Args:
            qpos: An iterable of joint positions matching `model.nq`.
            qvel: An iterable of joint velocities matching `model.nv`.
        """
        mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            qpos_arr = np.asarray(qpos, dtype=float)
            assert qpos_arr.shape == (self.model.nq,)
            self.data.qpos[:] = qpos_arr
        if qvel is not None:
            qvel_arr = np.asarray(qvel, dtype=float)
            assert qvel_arr.shape == (self.model.nv,)
            self.data.qvel[:] = qvel_arr
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl: Iterable[float], nstep: int = 1) -> None:
        """Advance the simulation by ``nstep`` steps applying the provided control.

        Args:
            ctrl: A sequence of control inputs (torques or forces) applied to
                the actuators.  The length must match the number of actuated
                dofs (`model.nu`).
            nstep: Number of time steps to advance the simulation.
        """
        ctrl_arr = np.asarray(ctrl, dtype=float)
        assert ctrl_arr.shape == (self.model.nu,)
        for _ in range(nstep):
            self.data.ctrl[:] = ctrl_arr
            mujoco.mj_step(self.model, self.data)

    # -- Dynamics queries --------------------------------------------------
    def inverse_dynamics(self, q: Iterable[float], v: Iterable[float], a: Iterable[float]) -> np.ndarray:
        """Compute the joint torques required to achieve a given motion.

        This method sets the internal state to the provided configuration,
        velocity and acceleration, calls `mujoco.mj_inverse` and returns
        the resulting `qfrc_inverse` vector.  It does **not** advance the
        simulation.

        Args:
            q: Joint positions of length ``model.nq``.
            v: Joint velocities of length ``model.nv``.
            a: Joint accelerations of length ``model.nv``.

        Returns:
            A numpy array containing the computed joint torques.
        """
        q_arr = np.asarray(q, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        a_arr = np.asarray(a, dtype=float)
        assert q_arr.shape == (self.model.nq,)
        assert v_arr.shape == (self.model.nv,)
        assert a_arr.shape == (self.model.nv,)
        self.data.qpos[:] = q_arr
        self.data.qvel[:] = v_arr
        self.data.qacc[:] = a_arr
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse.copy()

    def set_body_mass(self, body_name: str, mass: float) -> None:
        """Modify the mass of a body in the model.

        This can be used to perform parameter sweeps or system identification.
        The change takes effect immediately and will be saved if you call
        ``save_model``.

        Args:
            body_name: Name of the body whose mass will be modified.
            mass: New mass value.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        self.model.body_mass[body_id] = float(mass)

    def save_model(self, xml_path: str) -> None:
        """Save the current model (including any parameter edits) to MJCF.

        Args:
            xml_path: Path to write the new MJCF file.
        """
        mujoco.mj_saveLastXML(xml_path, self.model)