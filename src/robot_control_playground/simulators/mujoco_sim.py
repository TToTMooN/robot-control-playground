import mujoco
import numpy as np

class MujocoSim:
    """Thin wrapper around MuJoCo simulation and inverse dynamics."""
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

    def reset(self, qpos=None, qvel=None):
        """Reset state; optionally set joint positions and velocities."""
        mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            self.data.qpos[:] = np.array(qpos, dtype=float)
        if qvel is not None:
            self.data.qvel[:] = np.array(qvel, dtype=float)
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl, nstep=1):
        """Advance simulation by nstep time steps with specified control torques."""
        self.data.ctrl[:] = np.array(ctrl, dtype=float)
        for _ in range(nstep):
            mujoco.mj_step(self.model, self.data)
        return self.data.qpos.copy(), self.data.qvel.copy()

    def inverse_dynamics(self, q, v, a):
        """Compute joint torques required to achieve given q, v, a."""
        self.data.qpos[:] = np.array(q, dtype=float)
        self.data.qvel[:] = np.array(v, dtype=float)
        self.data.qacc[:] = np.array(a, dtype=float)
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse.copy()

    def set_body_mass(self, body_name, new_mass):
        """Modify mass of a body; useful for parameter sweeps or system ID."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.model.body_mass[body_id] = new_mass

    def save_model(self, xml_out):
        """Persist any modified parameters back to XML."""
        mujoco.mj_saveLastXML(xml_out, self.model)
